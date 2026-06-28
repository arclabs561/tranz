//! End-to-end Burn KGE training on real WN18RR, for all four models.
//!
//! Loads WN18RR, trains each model with the generic burn `train_kge`, and
//! reports filtered MRR / Hits on the test split. This is the real-data check
//! that the burn trainers actually learn (toy unit tests only prove the loss
//! decreases on synthetic triples).
//!
//! Data-gated: exits 0 with a message if `data/WN18RR` is absent, so it is safe
//! in CI that compiles or runs examples.
//!
//! Cost: trains all four models on full WN18RR (40943 entities, 173670
//! triples). On Metal (`--features burn-cpu,burn-gpu`) ~30-60s per model
//! (~3 min total); on CPU ndarray it is minutes per model. Shrink for a quick
//! smoke run with `TRAIN_CAP=20000 DIM=32 EPOCHS=5`.
//!
//! Run on Metal: `cargo run --release --features "burn-cpu,burn-gpu" --example wn18rr_kge_burn`
//! (drop `burn-gpu` for CPU ndarray).
//!
//! Sample output (Metal, dim 50, 3 epochs, sampled eval over 200 candidates):
//! ```text
//! model      train_s   loss[0]->loss[-1]      test_MRR   H@10
//! DistMult     39.1    10.386 ->   3.979        0.5706   0.6459
//! TransE       32.5     9.954 ->   5.013        0.5488   0.6407
//! ComplEx      41.8    10.317 ->   4.077        0.5560   0.6390
//! RotatE       57.1     9.596 ->   4.774        0.5532   0.6307
//! ```
//! Sampled-eval MRR (200 candidates) reads higher than full-entity-ranking
//! MRR; treat it as a relative signal, not a published-benchmark number.

#![allow(missing_docs)]

use std::path::Path;
use std::time::Instant;

use tranz::burn_train::{train_kge, BurnModelType, BurnTrainConfig};
use tranz::dataset::{self, FilterIndex, InternedDatasetExt};

// Backend: Metal/Vulkan via wgpu when `burn-gpu` is enabled, else CPU ndarray.
// Run on Metal with: `--features "burn-cpu,burn-gpu"`.
#[cfg(feature = "burn-gpu")]
type B = burn::backend::Autodiff<burn_wgpu::Wgpu>;
#[cfg(not(feature = "burn-gpu"))]
type B = burn::backend::Autodiff<burn_ndarray::NdArray>;

fn main() {
    let data = Path::new("data/WN18RR");
    if !data.join("train.txt").exists() {
        eprintln!("data/WN18RR not found; skipping (run scripts to fetch WN18RR). no-op.");
        return;
    }

    let ds = dataset::load_dataset(data).unwrap();
    let mut interned = ds.into_interned();
    // Optional triple cap for a fast smoke run (real entities, subset of edges).
    // Unset = full WN18RR. Entity/relation vocab is unaffected by the cap.
    if let Some(cap) = std::env::var("TRAIN_CAP")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        interned.train.truncate(cap);
    }
    interned.add_reciprocals();
    let n_ent = interned.num_entities();
    let n_rel = interned.num_relations();
    eprintln!(
        "WN18RR: {n_ent} entities, {n_rel} relations, {} train triples, {} test",
        interned.train.len(),
        interned.test.len()
    );
    let filter = FilterIndex::from_dataset(&interned);
    #[cfg(feature = "burn-gpu")]
    let device = burn_wgpu::WgpuDevice::default();
    #[cfg(not(feature = "burn-gpu"))]
    let device = burn_ndarray::NdArrayDevice::Cpu;
    #[cfg(feature = "burn-gpu")]
    eprintln!("backend: wgpu (Metal/Vulkan)");
    #[cfg(not(feature = "burn-gpu"))]
    eprintln!("backend: ndarray (CPU)");

    // Fast-but-real config: enough to show learning on real data, not SOTA
    // convergence. Raise dim/epochs for a serious run.
    let dim: usize = std::env::var("DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let epochs: usize = std::env::var("EPOCHS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let config = BurnTrainConfig {
        dim,
        lr: 0.005,
        label_smoothing: 0.1,
        batch_size: 1000,
        epochs,
        log_interval: 0,
        ..BurnTrainConfig::default()
    };

    println!("\nmodel      train_s   loss[0]->loss[-1]      test_MRR   H@10  (dim={dim}, {epochs} ep, sampled eval)");
    for mt in [
        BurnModelType::DistMult,
        BurnModelType::TransE,
        BurnModelType::ComplEx,
        BurnModelType::RotatE,
    ] {
        eprintln!("training {mt:?} ...");
        let start = Instant::now();
        let result = train_kge::<B>(&interned.train, n_ent, n_rel, mt, &config, &device);
        let train_s = start.elapsed().as_secs_f64();
        let (l0, l1) = (result.losses[0], *result.losses.last().unwrap());
        let scorer = result.to_scorer();
        // Sampled eval (200 candidates per test triple) keeps eval fast on 40k entities.
        let m = tranz::eval::evaluate_link_prediction_sampled(
            scorer.as_ref(),
            &interned.test,
            &filter,
            n_ent,
            200,
        );
        println!(
            "{:<9} {train_s:>7.1}   {l0:>7.3} -> {l1:>7.3}      {:>8.4}   {:>5.4}",
            format!("{mt:?}"),
            m.mrr,
            m.hits_at_10
        );
    }
    println!("\nAll four burn KGE models train end-to-end on real WN18RR.");
}
