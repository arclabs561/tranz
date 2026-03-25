//! tranz CLI: train, evaluate, and export KGE embeddings.
//!
//! ```sh
//! # Train on any TSV/CSV triple file
//! tranz train --triples data.tsv --model rotate --dim 200 --epochs 500 --output embeddings/
//!
//! # Train on WN18RR-format directory
//! tranz train --data data/WN18RR/ --model transe --dim 200 --epochs 500 --output embeddings/
//!
//! # Predict top-k tails for a query
//! tranz predict --embeddings embeddings/ --head "aspirin" --relation "treats" --k 10
//! ```

use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    match args[1].as_str() {
        "train" => cmd_train(&args[2..]),
        "help" | "--help" | "-h" => print_usage(),
        other => {
            eprintln!("Unknown command: {other}");
            print_usage();
            std::process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!(
        "tranz -- point-embedding knowledge graph completion

USAGE:
    tranz train [OPTIONS]

TRAIN OPTIONS:
    --data <DIR>          WN18RR-format directory (train.txt, valid.txt, test.txt)
    --triples <FILE>      Single TSV/CSV triple file (auto-split 80/10/10)
    --model <MODEL>       transe, rotate, complex, distmult (default: transe)
    --dim <N>             Embedding dimension (default: 200)
    --epochs <N>          Training epochs (default: 500)
    --batch-size <N>      Batch size (default: 512)
    --gamma <F>           Margin (default: 12.0)
    --lr <F>              Learning rate (default: 0.001)
    --negatives <N>       Negative samples per positive (default: 256)
    --alpha <F>           SANS adversarial temperature (default: 1.0)
    --n3 <F>              N3 regularization coefficient (default: 0.0)
    --reciprocals         Add reciprocal relations
    --output <DIR>        Output directory for embeddings (default: output/)
    --eval                Evaluate on test set after training"
    );
}

#[cfg(not(feature = "candle"))]
fn cmd_train(_args: &[str]) {
    eprintln!("Training requires the 'candle' feature. Build with: cargo install tranz --features candle");
    std::process::exit(1);
}

#[cfg(feature = "candle")]
fn cmd_train(args: &[String]) {
    use tranz::dataset;
    use tranz::eval::evaluate_link_prediction;
    use tranz::io::export_embeddings;
    use tranz::train::{self, ModelType, TrainConfig};
    use tranz::Scorer;

    let mut data_dir: Option<PathBuf> = None;
    let mut triples_file: Option<PathBuf> = None;
    let mut model_type = ModelType::TransE;
    let mut dim = 200_usize;
    let mut epochs = 500_usize;
    let mut batch_size = 512_usize;
    let mut gamma = 12.0_f32;
    let mut lr = 0.001_f64;
    let mut num_negatives = 256_usize;
    let mut alpha = 1.0_f32;
    let mut n3_reg = 0.0_f32;
    let mut reciprocals = false;
    let mut output_dir = PathBuf::from("output");
    let mut do_eval = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--data" => { i += 1; data_dir = Some(PathBuf::from(&args[i])); }
            "--triples" => { i += 1; triples_file = Some(PathBuf::from(&args[i])); }
            "--model" => {
                i += 1;
                model_type = match args[i].as_str() {
                    "transe" => ModelType::TransE,
                    "rotate" => ModelType::RotatE,
                    "complex" => ModelType::ComplEx,
                    "distmult" => ModelType::DistMult,
                    other => {
                        eprintln!("Unknown model: {other}");
                        std::process::exit(1);
                    }
                };
            }
            "--dim" => { i += 1; dim = args[i].parse().unwrap(); }
            "--epochs" => { i += 1; epochs = args[i].parse().unwrap(); }
            "--batch-size" => { i += 1; batch_size = args[i].parse().unwrap(); }
            "--gamma" => { i += 1; gamma = args[i].parse().unwrap(); }
            "--lr" => { i += 1; lr = args[i].parse().unwrap(); }
            "--negatives" => { i += 1; num_negatives = args[i].parse().unwrap(); }
            "--alpha" => { i += 1; alpha = args[i].parse().unwrap(); }
            "--n3" => { i += 1; n3_reg = args[i].parse().unwrap(); }
            "--reciprocals" => { reciprocals = true; }
            "--output" => { i += 1; output_dir = PathBuf::from(&args[i]); }
            "--eval" => { do_eval = true; }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // Load dataset.
    let ds = if let Some(dir) = &data_dir {
        eprintln!("Loading dataset from {}", dir.display());
        dataset::load_dataset(dir).unwrap_or_else(|e| {
            eprintln!("Failed: {e}");
            std::process::exit(1);
        })
    } else if let Some(file) = &triples_file {
        eprintln!("Loading triples from {}", file.display());
        let ds = dataset::load_triples(file).unwrap_or_else(|e| {
            eprintln!("Failed: {e}");
            std::process::exit(1);
        });
        eprintln!("Loaded {} triples, splitting 80/10/10", ds.train.len());
        ds.split(0.1, 0.1)
    } else {
        eprintln!("Specify --data <DIR> or --triples <FILE>");
        std::process::exit(1);
    };

    let mut interned = ds.into_interned();
    if reciprocals {
        eprintln!("Adding reciprocal relations");
        interned.add_reciprocals();
    }

    eprintln!(
        "Entities: {}, Relations: {}, Train: {}, Valid: {}, Test: {}",
        interned.num_entities(),
        interned.num_relations(),
        interned.train.len(),
        interned.valid.len(),
        interned.test.len(),
    );

    let config = TrainConfig {
        model_type,
        dim,
        num_negatives,
        gamma,
        adversarial_temperature: alpha,
        lr,
        n3_reg,
        batch_size,
        epochs,
        ..TrainConfig::default()
    };

    eprintln!("Training {model_type:?} dim={dim} gamma={gamma} lr={lr} epochs={epochs}");
    let device = candle_core::Device::Cpu;
    let start = Instant::now();

    let result = train::train(
        &interned.train,
        interned.num_entities(),
        interned.num_relations(),
        &config,
        &device,
    )
    .unwrap();

    eprintln!(
        "Training complete in {:.1}s, final loss: {:.4}",
        start.elapsed().as_secs_f32(),
        result.losses.last().unwrap(),
    );

    // Export embeddings.
    let entity_vecs = result.model.entity_vecs().unwrap();
    let relation_vecs = result.model.relation_vecs().unwrap();
    eprintln!("Exporting embeddings to {}", output_dir.display());
    export_embeddings(
        &output_dir,
        &interned.id_to_entity,
        &entity_vecs,
        &interned.id_to_relation,
        &relation_vecs,
    )
    .unwrap();
    eprintln!("Wrote entities.tsv and relations.tsv");

    // Optional evaluation.
    if do_eval && !interned.test.is_empty() {
        eprintln!("Evaluating on test set ({} triples)...", interned.test.len());
        let all_triples = interned.all_triples();
        let scorer: Box<dyn Scorer> = match model_type {
            ModelType::TransE => Box::new(result.model.to_transe().unwrap()),
            ModelType::RotatE => Box::new(result.model.to_rotate().unwrap()),
            ModelType::ComplEx => Box::new(result.model.to_complex().unwrap()),
            ModelType::DistMult => Box::new(result.model.to_distmult().unwrap()),
        };
        let metrics = evaluate_link_prediction(
            scorer.as_ref(),
            &interned.test,
            &all_triples,
            interned.num_entities(),
        );
        println!("MRR:      {:.4}", metrics.mrr);
        println!("Hits@1:   {:.4}", metrics.hits_at_1);
        println!("Hits@3:   {:.4}", metrics.hits_at_3);
        println!("Hits@10:  {:.4}", metrics.hits_at_10);
    }
}
