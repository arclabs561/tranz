//! Benchmark burn WGPU (Metal) vs CPU training.
//!
//! Run with: `cargo run --release --features burn-wgpu --example bench_wgpu`
#![allow(missing_docs)]
use std::time::Instant;
use tranz::burn_train::{train_complex, BurnTrainConfig};
use tranz::dataset::{self, InternedDatasetExt};

type B = burn::backend::Autodiff<burn_wgpu::Wgpu>;

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

    let device = burn_wgpu::WgpuDevice::default();
    eprintln!("WGPU device: {device:?}");

    for &dim in &[50, 100] {
        let config = BurnTrainConfig {
            dim,
            lr: 0.001,
            label_smoothing: 0.1,
            batch_size: 512,
            epochs: 3,
            log_interval: 1,
            ..BurnTrainConfig::default()
        };
        let start = Instant::now();
        let result = train_complex::<B>(&interned.train, n_ent, n_rel, &config, &device);
        let elapsed = start.elapsed();
        eprintln!("\nBurn WGPU ComplEx 1-N dim={dim} (3 epochs):");
        eprintln!(
            "  total: {:.1}s ({:.1}s/epoch)",
            elapsed.as_secs_f64(),
            elapsed.as_secs_f64() / 3.0,
        );
        eprintln!("  losses: {:?}", result.losses);
    }
}
