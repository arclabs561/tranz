//! Benchmark training epoch cost.
//!
//! Run with: `cargo run --release --features candle --example bench_training`
#![allow(missing_docs)]
use std::time::Instant;
use tranz::dataset::{self, InternedDatasetExt};
use tranz::train::{train, ModelType, TrainConfig};

fn main() {
    let ds = dataset::load_dataset("data/WN18RR".as_ref()).unwrap();
    let mut interned = ds.into_interned();
    interned.add_reciprocals();

    let n_ent = interned.num_entities();
    let n_rel = interned.num_relations();
    let n_train = interned.train.len();
    eprintln!("Dataset: {n_ent} entities, {n_rel} relations, {n_train} triples");

    let device = candle_core::Device::Cpu;

    // 1-N ComplEx: 3 epochs for timing
    let config = TrainConfig {
        model_type: ModelType::ComplEx,
        dim: 100,
        one_to_n: true,
        label_smoothing: 0.1,
        lr: 0.001,
        batch_size: 512,
        epochs: 3,
        log_interval: 1,
        ..TrainConfig::default()
    };

    let start = Instant::now();
    let result = train(&interned.train, n_ent, n_rel, &config, &device).unwrap();
    let total = start.elapsed();
    let per_epoch = total.as_secs_f64() / 3.0;

    eprintln!("\nComplEx 1-N dim=100 bs=512:");
    eprintln!("  total (3ep): {:.1}s", total.as_secs_f64());
    eprintln!("  per epoch: {per_epoch:.1}s");
    eprintln!(
        "  per batch: {:.1}ms",
        per_epoch * 1000.0 / (n_train as f64 / 512.0)
    );
    eprintln!("  losses: {:?}", result.losses);

    // Neg sampling for comparison
    let config_neg = TrainConfig {
        model_type: ModelType::ComplEx,
        dim: 100,
        num_negatives: 256,
        gamma: 12.0,
        adversarial_temperature: 1.0,
        lr: 0.001,
        batch_size: 512,
        epochs: 3,
        log_interval: 1,
        ..TrainConfig::default()
    };

    let start = Instant::now();
    let result = train(&interned.train, n_ent, n_rel, &config_neg, &device).unwrap();
    let total = start.elapsed();
    let per_epoch = total.as_secs_f64() / 3.0;

    eprintln!("\nComplEx neg-sampling dim=100 bs=512 k=256:");
    eprintln!("  total (3ep): {:.1}s", total.as_secs_f64());
    eprintln!("  per epoch: {per_epoch:.1}s");
    eprintln!("  losses: {:?}", result.losses);
}
