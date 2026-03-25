//! Train a KGE model on WN18RR and evaluate filtered link prediction.
//!
//! Expects a `data/WN18RR/` directory with `train.txt`, `valid.txt`, `test.txt`.
//! Download from: <https://github.com/TimDettmers/ConvE/tree/master/data/WN18RR>
//!
//! Usage:
//! ```sh
//! cargo run --release --features candle --example train_wn18rr -- \
//!     --model rotate --dim 200 --epochs 500 --batch-size 512 --gamma 12.0
//! ```

use std::path::PathBuf;
use std::time::Instant;

use tranz::dataset::load_dataset;
use tranz::eval::evaluate_link_prediction;
use tranz::train::{self, ModelType, TrainConfig};
use tranz::Scorer;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut data_path = PathBuf::from("data/WN18RR");
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

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data" => { i += 1; data_path = PathBuf::from(&args[i]); }
            "--model" => {
                i += 1;
                model_type = match args[i].as_str() {
                    "transe" => ModelType::TransE,
                    "rotate" => ModelType::RotatE,
                    "complex" => ModelType::ComplEx,
                    "distmult" => ModelType::DistMult,
                    other => {
                        eprintln!("Unknown model: {other}. Use: transe, rotate, complex, distmult");
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
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // Load dataset.
    eprintln!("Loading dataset from {}", data_path.display());
    let ds = load_dataset(&data_path).unwrap_or_else(|e| {
        eprintln!("Failed to load dataset: {e}");
        eprintln!("Download WN18RR to data/WN18RR/ with train.txt, valid.txt, test.txt");
        std::process::exit(1);
    });
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

    let elapsed = start.elapsed();
    eprintln!(
        "Training complete in {:.1}s, final loss: {:.4}",
        elapsed.as_secs_f32(),
        result.losses.last().unwrap(),
    );

    // Print loss curve (sampled).
    let n = result.losses.len();
    let sample_points = [0, n / 4, n / 2, 3 * n / 4, n - 1];
    eprintln!("Loss curve:");
    for &idx in &sample_points {
        if idx < n {
            eprintln!("  epoch {:>4}: {:.4}", idx + 1, result.losses[idx]);
        }
    }

    // Evaluate on test set.
    eprintln!("Evaluating on test set ({} triples)...", interned.test.len());
    let all_triples = interned.all_triples();
    let num_entities = interned.num_entities();

    let eval_start = Instant::now();
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
        num_entities,
    );

    eprintln!("Evaluation complete in {:.1}s", eval_start.elapsed().as_secs_f32());
    println!("MRR:      {:.4}", metrics.mrr);
    println!("Hits@1:   {:.4}", metrics.hits_at_1);
    println!("Hits@3:   {:.4}", metrics.hits_at_3);
    println!("Hits@10:  {:.4}", metrics.hits_at_10);
}
