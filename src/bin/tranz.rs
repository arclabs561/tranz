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

Training uses the Burn backend with 1-N (1vsAll) cross-entropy and AdamW.
Build with a backend: --features burn-ndarray (CPU) or --features burn-wgpu (GPU/Metal).

TRAIN OPTIONS:
    --data <DIR>          WN18RR-format directory (train.txt, valid.txt, test.txt)
    --triples <FILE>      Single TSV/CSV triple file (auto-split 80/10/10)
    --model <MODEL>       complex, distmult, rotate, transe (default: transe)
                          Recommended: complex --label-smoothing 0.1 --reciprocals
    --dim <N>             Embedding dimension (default: 200)
    --epochs <N>          Training epochs (default: 500)
    --batch-size <N>      Batch size (default: 512)
    --lr <F>              Learning rate (default: 0.001)
    --init-scale <F>      Init std for embeddings (default: 0.001)
    --label-smoothing <F> Label smoothing epsilon for 1-N CE (default: 0.0)
    --reciprocals         Add reciprocal relations before training
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

    let rel_id = relation.as_ref().map(|rel_name| {
        *rel_map.get(rel_name.as_str()).unwrap_or_else(|| {
            eprintln!("Unknown relation: {rel_name}");
            eprintln!("Available: {}", rel_names.join(", "));
            std::process::exit(1);
        })
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

    if let (Some(head_name), Some(rel_id)) = (&head, rel_id) {
        // Tail prediction: (head, relation, ?)
        let head_id = *ent_map.get(head_name.as_str()).unwrap_or_else(|| {
            eprintln!("Unknown entity: {head_name}");
            std::process::exit(1);
        });
        let rel_name = relation.as_ref().unwrap();
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
    } else if let (Some(tail_name), Some(rel_id)) = (&tail, rel_id) {
        // Head prediction: (?, relation, tail)
        let tail_id = *ent_map.get(tail_name.as_str()).unwrap_or_else(|| {
            eprintln!("Unknown entity: {tail_name}");
            std::process::exit(1);
        });
        let rel_name = relation.as_ref().unwrap();
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
    } else if let (Some(head_name), Some(tail_name)) = (&head, &tail) {
        // Relation prediction: (head, ?, tail) -- no relation specified.
        let head_id = *ent_map.get(head_name.as_str()).unwrap_or_else(|| {
            eprintln!("Unknown entity: {head_name}");
            std::process::exit(1);
        });
        let tail_id = *ent_map.get(tail_name.as_str()).unwrap_or_else(|| {
            eprintln!("Unknown entity: {tail_name}");
            std::process::exit(1);
        });
        let results = scorer.top_k_relations(head_id, tail_id, rel_names.len(), k);
        println!("Top-{k} relation predictions for ({head_name}, ?, {tail_name}):");
        for (rank, (rel_id, score)) in results.iter().enumerate() {
            println!(
                "  {:>3}. {:<30} score={:.4}",
                rank + 1,
                &rel_names[*rel_id],
                score
            );
        }
    } else {
        eprintln!("Specify --head + --relation for tail prediction, --tail + --relation for head prediction, or --head + --tail for relation prediction");
        std::process::exit(1);
    }
}

fn cmd_train(args: &[String]) {
    use tranz::burn_train::{train_kge, BurnModelType, BurnTrainConfig};
    use tranz::dataset::{self, DatasetExt, InternedDatasetExt};
    use tranz::io::{
        describe_embedding_artifact, export_embeddings, write_embedding_manifest,
        EmbeddingManifest, ManifestDataset, ManifestMetrics, ManifestTraining,
    };

    // Backend: prefer wgpu (Metal/Vulkan) when enabled, else ndarray (CPU).
    // The bin's required-features guarantees burn-ndarray, so `not(burn-wgpu)`
    // always resolves to an available backend.
    #[cfg(feature = "burn-wgpu")]
    type TrainB = burn::backend::Autodiff<burn_wgpu::Wgpu>;
    #[cfg(not(feature = "burn-wgpu"))]
    type TrainB = burn::backend::Autodiff<burn_ndarray::NdArray>;

    let mut data_dir: Option<PathBuf> = None;
    let mut triples_file: Option<PathBuf> = None;
    let mut model_type = BurnModelType::TransE;
    let mut dim = 200_usize;
    let mut init_scale = 1e-3_f64;
    let mut epochs = 500_usize;
    let mut batch_size = 512_usize;
    let mut lr = 0.001_f64;
    let mut label_smoothing = 0.0_f64;
    let mut reciprocals = false;
    let mut output_dir = PathBuf::from("output");
    let mut do_eval = false;

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
                    "transe" => BurnModelType::TransE,
                    "rotate" => BurnModelType::RotatE,
                    "complex" => BurnModelType::ComplEx,
                    "distmult" => BurnModelType::DistMult,
                    other => {
                        eprintln!("Unknown model: {other}");
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
            "--lr" => {
                i += 1;
                lr = args[i].parse().unwrap();
            }
            "--label-smoothing" => {
                i += 1;
                label_smoothing = args[i].parse().unwrap();
            }
            "--reciprocals" => {
                reciprocals = true;
            }
            "--output" => {
                i += 1;
                output_dir = PathBuf::from(&args[i]);
            }
            "--eval" => {
                do_eval = true;
            }
            // 1-N (1vsAll) CE is the only training mode; accepted for compatibility.
            "--1n" | "--one-to-n" => {}
            other => {
                eprintln!("Unknown argument: {other}");
                eprintln!("Run `tranz help` for the supported flags (Burn 1-N trainer).");
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
        let ds = dataset::Dataset::load_flexible(file).unwrap_or_else(|e| {
            eprintln!("Failed: {e}");
            std::process::exit(1);
        });
        eprintln!("Loaded {} triples, splitting 80/10/10", ds.train.len());
        ds.split(0.1, 0.1)
    } else {
        eprintln!("Specify --data <DIR> or --triples <FILE>");
        std::process::exit(1);
    };
    let (source_kind, source_path, split) = if let Some(dir) = &data_dir {
        (
            "directory".to_string(),
            dir.display().to_string(),
            "provided_train_valid_test".to_string(),
        )
    } else if let Some(file) = &triples_file {
        (
            "triple_file".to_string(),
            file.display().to_string(),
            "auto_80_10_10".to_string(),
        )
    } else {
        unreachable!("dataset source was validated above")
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

    let config = BurnTrainConfig {
        dim,
        init_scale,
        lr,
        label_smoothing,
        n3_reg: 0.0,
        batch_size,
        epochs,
        log_interval: 0,
    };

    // Print full command for reproducibility.
    eprintln!("Command: tranz train {}", args.join(" "));
    eprintln!("Training {model_type:?} dim={dim} lr={lr} epochs={epochs} (Burn 1-N / AdamW)");
    #[cfg(feature = "burn-wgpu")]
    let device = burn_wgpu::WgpuDevice::default();
    #[cfg(not(feature = "burn-wgpu"))]
    let device = burn_ndarray::NdArrayDevice::Cpu;
    let start = Instant::now();

    let result = train_kge::<TrainB>(
        &interned.train,
        interned.num_entities(),
        interned.num_relations(),
        model_type,
        &config,
        &device,
    );

    eprintln!(
        "Training complete in {:.1}s, final loss: {:.4}",
        start.elapsed().as_secs_f32(),
        result.losses.last().unwrap(),
    );

    // Export embeddings.
    let entity_vecs = result.entity_vecs.clone();
    let relation_vecs = result.relation_vecs.clone();
    eprintln!("Exporting embeddings to {}", output_dir.display());
    let ent_names: Vec<String> = (0..interned.num_entities())
        .map(|i| interned.entities.get(i).unwrap().to_string())
        .collect();
    let rel_names: Vec<String> = (0..interned.num_relations())
        .map(|i| interned.relations.get(i).unwrap().to_string())
        .collect();
    export_embeddings(
        &output_dir,
        &ent_names,
        &entity_vecs,
        &rel_names,
        &relation_vecs,
    )
    .unwrap();
    eprintln!("Wrote entities.tsv and relations.tsv");

    let mut manifest_metrics = None;

    // Optional evaluation.
    if do_eval && !interned.test.is_empty() {
        use tranz::dataset::FilterIndex;
        use tranz::eval::evaluate_link_prediction_detailed;

        eprintln!(
            "Evaluating on test set ({} triples)...",
            interned.test.len()
        );
        let filter = FilterIndex::from_dataset(&interned);
        let scorer = result.to_scorer();
        let eval = evaluate_link_prediction_detailed(scorer.as_ref(), &interned.test, &filter);
        let m = eval.metrics;
        println!("MRR:      {:.4}", m.mrr);
        println!("  head:   {:.4}", m.head_mrr);
        println!("  tail:   {:.4}", m.tail_mrr);
        println!("MR:       {:.1}", m.mean_rank);
        println!("Hits@1:   {:.4}", m.hits_at_1);
        println!("Hits@3:   {:.4}", m.hits_at_3);
        println!("Hits@10:  {:.4}", m.hits_at_10);
        manifest_metrics = Some(ManifestMetrics {
            mrr: m.mrr,
            head_mrr: m.head_mrr,
            tail_mrr: m.tail_mrr,
            mean_rank: m.mean_rank,
            hits_at_1: m.hits_at_1,
            hits_at_3: m.hits_at_3,
            hits_at_10: m.hits_at_10,
        });

        if !eval.per_relation.is_empty() {
            println!();
            println!("Per-relation MRR:");
            let mut rels: Vec<_> = eval.per_relation.iter().collect();
            rels.sort_by_key(|&(id, _)| *id);
            for (&rel_id, metrics) in &rels {
                let name = interned.relations.get(rel_id).unwrap_or("?");
                println!(
                    "  {name:<30} MRR={:.4}  H@10={:.4}",
                    metrics.mrr, metrics.hits_at_10
                );
            }
        }
    }

    let entity_dim = entity_vecs.first().map_or(0, Vec::len);
    let relation_dim = relation_vecs.first().map_or(0, Vec::len);
    let artifacts = vec![
        describe_embedding_artifact(
            &output_dir,
            "entities.tsv",
            "application/vnd.tranz.entity-embeddings+w2v-tsv",
            "w2v-tsv",
            ent_names.len(),
            entity_dim,
        ),
        describe_embedding_artifact(
            &output_dir,
            "relations.tsv",
            "application/vnd.tranz.relation-embeddings+w2v-tsv",
            "w2v-tsv",
            rel_names.len(),
            relation_dim,
        ),
    ]
    .into_iter()
    .collect::<std::io::Result<Vec<_>>>()
    .unwrap_or_else(|e| {
        eprintln!("Failed to describe exported embeddings: {e}");
        std::process::exit(1);
    });

    let manifest = EmbeddingManifest {
        schema: "tranz.embedding-export.v1".to_string(),
        model: model_type_name(model_type).to_string(),
        score_order: "lower_is_better".to_string(),
        artifacts,
        dataset: ManifestDataset {
            source_kind,
            source_path,
            split,
            entities: interned.num_entities(),
            relations: interned.num_relations(),
            train_triples: interned.train.len(),
            valid_triples: interned.valid.len(),
            test_triples: interned.test.len(),
        },
        training: ManifestTraining {
            trainer: "burn-1n-adamw".to_string(),
            backend: burn_backend_name().to_string(),
            dim,
            init_scale,
            lr,
            label_smoothing,
            n3_reg: config.n3_reg,
            batch_size,
            epochs,
            reciprocals,
            final_loss: result.losses.last().copied(),
        },
        metrics: manifest_metrics,
    };
    write_embedding_manifest(&output_dir, &manifest).unwrap_or_else(|e| {
        eprintln!("Failed to write manifest.json: {e}");
        std::process::exit(1);
    });
    eprintln!("Wrote manifest.json");
}

fn cmd_eval(args: &[String]) {
    use tranz::dataset::{self, FilterIndex};
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
    let filter = FilterIndex::from_dataset(&interned);
    let result = evaluate_link_prediction_detailed(scorer.as_ref(), &interned.test, &filter);

    let m = result.metrics;
    println!("MRR:      {:.4}", m.mrr);
    println!("  head:   {:.4}", m.head_mrr);
    println!("  tail:   {:.4}", m.tail_mrr);
    println!("MR:       {:.1}", m.mean_rank);
    println!("Hits@1:   {:.4}", m.hits_at_1);
    println!("Hits@3:   {:.4}", m.hits_at_3);
    println!("Hits@10:  {:.4}", m.hits_at_10);

    if !result.per_relation.is_empty() {
        println!();
        println!("Per-relation MRR:");
        let mut rels: Vec<_> = result.per_relation.iter().collect();
        rels.sort_by_key(|&(id, _)| *id);
        for (&rel_id, metrics) in &rels {
            let name = interned.relations.get(rel_id).unwrap_or("?");
            println!(
                "  {name:<30} MRR={:.4}  H@10={:.4}",
                metrics.mrr, metrics.hits_at_10
            );
        }
    }
}

fn model_type_name(model_type: tranz::burn_train::BurnModelType) -> &'static str {
    match model_type {
        tranz::burn_train::BurnModelType::TransE => "transe",
        tranz::burn_train::BurnModelType::RotatE => "rotate",
        tranz::burn_train::BurnModelType::ComplEx => "complex",
        tranz::burn_train::BurnModelType::DistMult => "distmult",
    }
}

fn burn_backend_name() -> &'static str {
    if cfg!(feature = "burn-wgpu") {
        "burn-wgpu"
    } else {
        "burn-ndarray"
    }
}
