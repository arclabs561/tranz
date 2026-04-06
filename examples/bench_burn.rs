//! Benchmark burn vs candle training on a small dataset.
//!
//! Run with: `cargo run --release --features "candle,burn-cpu" --example bench_burn`
#![allow(missing_docs)]
use std::time::Instant;
use tranz::dataset::{self, InternedDatasetExt};

fn main() {
    let ds = dataset::load_dataset("data/WN18RR".as_ref()).unwrap();
    let mut interned = ds.into_interned();
    interned.add_reciprocals();

    let n_ent = interned.num_entities();
    let n_rel = interned.num_relations();
    eprintln!(
        "Dataset: {n_ent} entities, {n_rel} relations, {} triples",
        interned.train.len()
    );

    let dim = 50; // small dim for quick comparison
    let epochs = 3;

    // Candle
    {
        use tranz::train::{train, ModelType, TrainConfig};
        let config = TrainConfig {
            model_type: ModelType::ComplEx,
            dim,
            one_to_n: true,
            label_smoothing: 0.1,
            lr: 0.001,
            batch_size: 512,
            epochs,
            ..TrainConfig::default()
        };
        let device = candle_core::Device::Cpu;
        let start = Instant::now();
        let result = train(&interned.train, n_ent, n_rel, &config, &device).unwrap();
        let elapsed = start.elapsed();
        eprintln!("\nCandle ComplEx 1-N dim={dim} ({epochs} epochs):");
        eprintln!(
            "  total: {:.1}s ({:.1}s/epoch)",
            elapsed.as_secs_f64(),
            elapsed.as_secs_f64() / epochs as f64,
        );
        eprintln!("  losses: {:?}", result.losses);
    }

    // Burn ndarray
    {
        use tranz::burn_train::{train_complex, BurnTrainConfig};
        type B = burn::backend::Autodiff<burn_ndarray::NdArray>;
        let config = BurnTrainConfig {
            dim,
            lr: 0.001,
            label_smoothing: 0.1,
            batch_size: 512,
            epochs,
            ..BurnTrainConfig::default()
        };
        let device = burn_ndarray::NdArrayDevice::Cpu;
        let start = Instant::now();
        let result = train_complex::<B>(&interned.train, n_ent, n_rel, &config, &device);
        let elapsed = start.elapsed();
        eprintln!("\nBurn ndarray ComplEx 1-N dim={dim} ({epochs} epochs):");
        eprintln!(
            "  total: {:.1}s ({:.1}s/epoch)",
            elapsed.as_secs_f64(),
            elapsed.as_secs_f64() / epochs as f64,
        );
        eprintln!("  losses: {:?}", result.losses);
    }
}
