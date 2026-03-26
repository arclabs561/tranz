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
        "predict" => cmd_predict(&args[2..]),
        "eval" => cmd_eval(&args[2..]),
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
    --model <MODEL>       complex, distmult, rotate, transe (default: transe)
                          Recommended: complex --1n --label-smoothing 0.1
    --dim <N>             Embedding dimension (default: 200)
    --epochs <N>          Training epochs (default: 500)
    --batch-size <N>      Batch size (default: 512)
    --gamma <F>           Margin (default: 12.0)
    --lr <F>              Learning rate (default: 0.001)
    --negatives <N>       Negative samples per positive (default: 256)
    --alpha <F>           SANS adversarial temperature (default: 1.0)
    --n3 <F>              N3 regularization coefficient (default: 0.0)
    --norm <N>            Distance norm: 1=L1, 2=L2 (default: 1)
    --dropout <F>         Embedding dropout rate (default: 0.0)
    --1n / --one-to-n     Use 1-N scoring with BCE loss (faster convergence)
    --label-smoothing <F> Label smoothing epsilon for 1-N mode (default: 0.0)
    --reciprocals         Add reciprocal relations
    --normalize           Normalize entity embeddings to unit L2 norm
    --subsampling         Apply entity frequency subsampling weights
    --warmup <N>          Linear LR warmup epochs (default: 0)
    --log-interval <N>    Print loss every N epochs (default: 10)
    --output <DIR>        Output directory for embeddings (default: output/)
    --eval                Evaluate on test set after training

USAGE:
    tranz predict [OPTIONS]

PREDICT OPTIONS:
    --embeddings <DIR>    Directory with entities.tsv and relations.tsv
    --model <MODEL>       transe, rotate, complex, distmult (default: transe)
    --head <NAME>         Head entity name (for tail prediction)
    --tail <NAME>         Tail entity name (for head prediction)
    --relation <NAME>     Relation name
    --k <N>               Number of predictions (default: 10)"
    );
}

fn cmd_predict(args: &[String]) {
    use std::collections::HashMap;
    use tranz::io::load_embeddings;
    use tranz::Scorer;

    let mut embeddings_dir = PathBuf::from("output");
    let mut model_name = "transe".to_string();
    let mut head: Option<String> = None;
    let mut tail: Option<String> = None;
    let mut relation: Option<String> = None;
    let mut k = 10_usize;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--embeddings" => {
                i += 1;
                embeddings_dir = PathBuf::from(&args[i]);
            }
            "--model" => {
                i += 1;
                model_name = args[i].clone();
            }
            "--head" => {
                i += 1;
                head = Some(args[i].clone());
            }
            "--tail" => {
                i += 1;
                tail = Some(args[i].clone());
            }
            "--relation" => {
                i += 1;
                relation = Some(args[i].clone());
            }
            "--k" => {
                i += 1;
                k = args[i].parse().unwrap();
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let rel_name = relation.unwrap_or_else(|| {
        eprintln!("--relation is required");
        std::process::exit(1);
    });

    // Load embeddings.
    let loaded = load_embeddings(&embeddings_dir).unwrap_or_else(|e| {
        eprintln!("Failed to load embeddings: {e}");
        std::process::exit(1);
    });
    let ent_names = loaded.entity_names;
    let ent_vecs = loaded.entity_vecs;
    let rel_names = loaded.relation_names;
    let rel_vecs = loaded.relation_vecs;

    // Build name-to-index maps.
    let ent_map: HashMap<&str, usize> = ent_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();
    let rel_map: HashMap<&str, usize> = rel_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();

    let rel_id = *rel_map.get(rel_name.as_str()).unwrap_or_else(|| {
        eprintln!("Unknown relation: {rel_name}");
        eprintln!("Available: {}", rel_names.join(", "));
        std::process::exit(1);
    });

    // Determine embedding dim.
    let emb_dim = ent_vecs[0].len();

    // Build model based on type.
    let scorer: Box<dyn Scorer + Sync> = match model_name.as_str() {
        "transe" => Box::new(tranz::TransE::from_vecs(ent_vecs, rel_vecs, emb_dim)),
        "distmult" => Box::new(tranz::DistMult::from_vecs(ent_vecs, rel_vecs, emb_dim)),
        "complex" => {
            let dim = emb_dim / 2;
            Box::new(tranz::ComplEx::from_vecs(ent_vecs, rel_vecs, dim))
        }
        "rotate" => {
            let dim = emb_dim / 2;
            Box::new(tranz::RotatE::from_vecs(ent_vecs, rel_vecs, dim, 12.0))
        }
        other => {
            eprintln!("Unknown model: {other}");
            std::process::exit(1);
        }
    };

    if let Some(head_name) = &head {
        // Tail prediction: (head, relation, ?)
        let head_id = *ent_map.get(head_name.as_str()).unwrap_or_else(|| {
            eprintln!("Unknown entity: {head_name}");
            std::process::exit(1);
        });
        let results = scorer.top_k_tails(head_id, rel_id, k);
        println!("Top-{k} tail predictions for ({head_name}, {rel_name}, ?):");
        for (rank, (ent_id, score)) in results.iter().enumerate() {
            println!(
                "  {:>3}. {:<30} score={:.4}",
                rank + 1,
                &ent_names[*ent_id],
                score
            );
        }
    } else if let Some(tail_name) = &tail {
        // Head prediction: (?, relation, tail)
        let tail_id = *ent_map.get(tail_name.as_str()).unwrap_or_else(|| {
            eprintln!("Unknown entity: {tail_name}");
            std::process::exit(1);
        });
        let results = scorer.top_k_heads(rel_id, tail_id, k);
        println!("Top-{k} head predictions for (?, {rel_name}, {tail_name}):");
        for (rank, (ent_id, score)) in results.iter().enumerate() {
            println!(
                "  {:>3}. {:<30} score={:.4}",
                rank + 1,
                &ent_names[*ent_id],
                score
            );
        }
    } else {
        eprintln!("Specify --head <NAME> for tail prediction or --tail <NAME> for head prediction");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "candle"))]
fn cmd_train(_args: &[str]) {
    eprintln!(
        "Training requires the 'candle' feature. Build with: cargo install tranz --features candle"
    );
    std::process::exit(1);
}

#[cfg(feature = "candle")]
fn cmd_train(args: &[String]) {
    use tranz::dataset;
    use tranz::io::export_embeddings;
    use tranz::train::{self, ModelType, TrainConfig};
    use tranz::Scorer;

    let mut data_dir: Option<PathBuf> = None;
    let mut triples_file: Option<PathBuf> = None;
    let mut model_type = ModelType::TransE;
    let mut optimizer_type = tranz::train::OptimizerType::AdamW;
    let mut dim = 200_usize;
    let mut init_scale = 1e-3_f64;
    let mut epochs = 500_usize;
    let mut batch_size = 512_usize;
    let mut gamma = 12.0_f32;
    let mut lr = 0.001_f64;
    let mut num_negatives = 256_usize;
    let mut alpha = 1.0_f32;
    let mut n3_reg = 0.0_f32;
    let mut dropout = 0.0_f32;
    let mut distance_norm = 1_u32;
    let mut subsampling = false;
    let mut reciprocals = false;
    let mut normalize = false;
    let mut warmup = 0_usize;
    let mut log_interval = 10_usize;
    let mut output_dir = PathBuf::from("output");
    let mut checkpoint_interval = 0_usize;
    let mut do_eval = false;
    let mut use_gpu = false;
    let mut one_to_n = false;
    let mut label_smoothing = 0.0_f32;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--data" => {
                i += 1;
                data_dir = Some(PathBuf::from(&args[i]));
            }
            "--triples" => {
                i += 1;
                triples_file = Some(PathBuf::from(&args[i]));
            }
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
            "--optimizer" => {
                i += 1;
                optimizer_type = match args[i].as_str() {
                    "adam" | "adamw" => tranz::train::OptimizerType::AdamW,
                    "adagrad" => tranz::train::OptimizerType::Adagrad,
                    other => {
                        eprintln!("Unknown optimizer: {other}. Use: adamw, adagrad");
                        std::process::exit(1);
                    }
                };
            }
            "--init-scale" => {
                i += 1;
                init_scale = args[i].parse().unwrap();
            }
            "--dim" => {
                i += 1;
                dim = args[i].parse().unwrap();
            }
            "--epochs" => {
                i += 1;
                epochs = args[i].parse().unwrap();
            }
            "--batch-size" => {
                i += 1;
                batch_size = args[i].parse().unwrap();
            }
            "--gamma" => {
                i += 1;
                gamma = args[i].parse().unwrap();
            }
            "--lr" => {
                i += 1;
                lr = args[i].parse().unwrap();
            }
            "--negatives" => {
                i += 1;
                num_negatives = args[i].parse().unwrap();
            }
            "--alpha" => {
                i += 1;
                alpha = args[i].parse().unwrap();
            }
            "--n3" => {
                i += 1;
                n3_reg = args[i].parse().unwrap();
            }
            "--norm" => {
                i += 1;
                distance_norm = args[i].parse().unwrap();
            }
            "--dropout" => {
                i += 1;
                dropout = args[i].parse().unwrap();
            }
            "--subsampling" => {
                subsampling = true;
            }
            "--reciprocals" => {
                reciprocals = true;
            }
            "--normalize" => {
                normalize = true;
            }
            "--warmup" => {
                i += 1;
                warmup = args[i].parse().unwrap();
            }
            "--log-interval" => {
                i += 1;
                log_interval = args[i].parse().unwrap();
            }
            "--output" => {
                i += 1;
                output_dir = PathBuf::from(&args[i]);
            }
            "--eval" => {
                do_eval = true;
            }
            "--gpu" => {
                use_gpu = true;
            }
            "--checkpoint" => {
                i += 1;
                checkpoint_interval = args[i].parse().unwrap();
            }
            "--1n" | "--one-to-n" => {
                one_to_n = true;
            }
            "--label-smoothing" => {
                i += 1;
                label_smoothing = args[i].parse().unwrap();
            }
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
        optimizer: optimizer_type,
        dim,
        init_scale,
        num_negatives,
        gamma,
        adversarial_temperature: alpha,
        lr,
        embedding_dropout: dropout,
        n3_reg,
        distance_norm,
        subsampling,
        one_to_n,
        label_smoothing,
        batch_size,
        epochs,
        normalize_entities: normalize,
        warmup_epochs: warmup,
        log_interval,
        checkpoint_dir: if checkpoint_interval > 0 {
            Some(output_dir.clone())
        } else {
            None
        },
        checkpoint_interval,
        ..TrainConfig::default()
    };

    eprintln!("Training {model_type:?} dim={dim} gamma={gamma} lr={lr} epochs={epochs}");
    let device = if use_gpu {
        candle_core::Device::new_cuda(0).unwrap_or_else(|e| {
            eprintln!("CUDA not available: {e}, falling back to CPU");
            candle_core::Device::Cpu
        })
    } else {
        candle_core::Device::Cpu
    };
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
        use tranz::eval::evaluate_link_prediction_detailed;

        eprintln!(
            "Evaluating on test set ({} triples)...",
            interned.test.len()
        );
        let all_triples = interned.all_triples();
        let scorer: Box<dyn Scorer + Sync> = match model_type {
            ModelType::TransE => Box::new(result.model.to_transe().unwrap()),
            ModelType::RotatE => Box::new(result.model.to_rotate().unwrap()),
            ModelType::ComplEx => Box::new(result.model.to_complex().unwrap()),
            ModelType::DistMult => Box::new(result.model.to_distmult().unwrap()),
        };
        let result = evaluate_link_prediction_detailed(
            scorer.as_ref(),
            &interned.test,
            &all_triples,
            interned.num_entities(),
        );
        let m = result.metrics;
        println!("MRR:      {:.4}", m.mrr);
        println!("Hits@1:   {:.4}", m.hits_at_1);
        println!("Hits@3:   {:.4}", m.hits_at_3);
        println!("Hits@10:  {:.4}", m.hits_at_10);

        if !result.per_relation.is_empty() {
            println!();
            println!("Per-relation MRR:");
            let mut rels: Vec<_> = result.per_relation.iter().collect();
            rels.sort_by_key(|&(id, _)| *id);
            for (&rel_id, metrics) in &rels {
                let name = interned
                    .id_to_relation
                    .get(rel_id)
                    .map(|s| s.as_str())
                    .unwrap_or("?");
                println!(
                    "  {name:<30} MRR={:.4}  H@10={:.4}",
                    metrics.mrr, metrics.hits_at_10
                );
            }
        }
    }
}

fn cmd_eval(args: &[String]) {
    use tranz::dataset;
    use tranz::eval::evaluate_link_prediction_detailed;
    use tranz::io::load_embeddings;
    use tranz::Scorer;

    let mut data_dir: Option<PathBuf> = None;
    let mut embeddings_dir = PathBuf::from("output");
    let mut model_name = "transe".to_string();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--data" => {
                i += 1;
                data_dir = Some(PathBuf::from(&args[i]));
            }
            "--embeddings" => {
                i += 1;
                embeddings_dir = PathBuf::from(&args[i]);
            }
            "--model" => {
                i += 1;
                model_name = args[i].clone();
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let data_path = data_dir.unwrap_or_else(|| {
        eprintln!("--data <DIR> is required for eval");
        std::process::exit(1);
    });

    eprintln!("Loading dataset from {}", data_path.display());
    let ds = dataset::load_dataset(&data_path).unwrap_or_else(|e| {
        eprintln!("Failed: {e}");
        std::process::exit(1);
    });
    let interned = ds.into_interned();

    eprintln!("Loading embeddings from {}", embeddings_dir.display());
    let loaded = load_embeddings(&embeddings_dir).unwrap_or_else(|e| {
        eprintln!("Failed: {e}");
        std::process::exit(1);
    });

    let emb_dim = loaded.entity_vecs[0].len();
    let scorer: Box<dyn Scorer + Sync> = match model_name.as_str() {
        "transe" => Box::new(tranz::TransE::from_vecs(
            loaded.entity_vecs,
            loaded.relation_vecs,
            emb_dim,
        )),
        "distmult" => Box::new(tranz::DistMult::from_vecs(
            loaded.entity_vecs,
            loaded.relation_vecs,
            emb_dim,
        )),
        "complex" => {
            let dim = emb_dim / 2;
            Box::new(tranz::ComplEx::from_vecs(
                loaded.entity_vecs,
                loaded.relation_vecs,
                dim,
            ))
        }
        "rotate" => {
            let dim = emb_dim / 2;
            Box::new(tranz::RotatE::from_vecs(
                loaded.entity_vecs,
                loaded.relation_vecs,
                dim,
                12.0,
            ))
        }
        other => {
            eprintln!("Unknown model: {other}");
            std::process::exit(1);
        }
    };

    eprintln!(
        "Evaluating on test set ({} triples)...",
        interned.test.len()
    );
    let all_triples = interned.all_triples();
    let result = evaluate_link_prediction_detailed(
        scorer.as_ref(),
        &interned.test,
        &all_triples,
        interned.num_entities(),
    );

    let m = result.metrics;
    println!("MRR:      {:.4}", m.mrr);
    println!("Hits@1:   {:.4}", m.hits_at_1);
    println!("Hits@3:   {:.4}", m.hits_at_3);
    println!("Hits@10:  {:.4}", m.hits_at_10);

    if !result.per_relation.is_empty() {
        println!();
        println!("Per-relation MRR:");
        let mut rels: Vec<_> = result.per_relation.iter().collect();
        rels.sort_by_key(|&(id, _)| *id);
        for (&rel_id, metrics) in &rels {
            let name = interned
                .id_to_relation
                .get(rel_id)
                .map(|s| s.as_str())
                .unwrap_or("?");
            println!(
                "  {name:<30} MRR={:.4}  H@10={:.4}",
                metrics.mrr, metrics.hits_at_10
            );
        }
    }
}
