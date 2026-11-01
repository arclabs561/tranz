//! Deterministic CPU training metric, for before/after backend comparison.
//!
//! Trains all four KGE models on a fixed synthetic knowledge graph with a fixed
//! seed and prints final loss plus filtered MRR / Hits@10. Everything the run
//! depends on is a literal in this file (data, config, seed), so two backends
//! running the same binary revision produce directly comparable numbers: this
//! is the harness used to show CPU parity when the CPU backend moved from
//! `burn-ndarray` to `burn-flex`.
//!
//! Run: `cargo run --release --features burn-ndarray --example cpu_parity_metric`

#![allow(missing_docs)]

use tranz::burn_train::{train_kge_seeded, BurnModelType, BurnTrainConfig};
use tranz::dataset::{Dataset, FilterIndex, InternedDataset, Triple};

/// Seed for the parameter initialization and the per-epoch shuffle.
const SEED: u64 = 20261031;
/// Entities in the synthetic graph.
const N_ENT: usize = 24;
/// Relations in the synthetic graph.
const N_REL: usize = 2;
/// Embedding dimension.
const DIM: usize = 32;
/// Training epochs.
const EPOCHS: usize = 300;
/// Every fifth head row is held out for evaluation.
const TEST_EVERY: usize = 5;

/// CPU backend, selected by the crate's CPU feature.
type B = burn::backend::Autodiff<burn_ndarray::NdArray>;

fn main() {
    let device = burn_ndarray::NdArrayDevice::Cpu;
    let ds = synthetic_dataset();
    let filter = FilterIndex::from_dataset(&ds);
    let (n_ent, n_rel) = (ds.num_entities(), ds.num_relations());

    eprintln!(
        "synthetic KG: {n_ent} entities, {n_rel} relations, {} train / {} test triples",
        ds.train.len(),
        ds.test.len()
    );
    eprintln!(
        "config: dim={DIM} epochs={EPOCHS} batch={} lr=0.01 eps=0.1 seed={SEED}",
        batch_size(&ds)
    );

    println!("\nmodel      loss[0]   loss[-1]       MRR     H@10");
    for mt in [
        BurnModelType::TransE,
        BurnModelType::RotatE,
        BurnModelType::ComplEx,
        BurnModelType::DistMult,
    ] {
        let result =
            train_kge_seeded::<B>(&ds.train, n_ent, n_rel, mt, &config(&ds), &device, SEED);
        let first = result.losses[0];
        let last = *result.losses.last().expect("epochs > 0");
        let metrics = tranz::eval::evaluate_link_prediction(
            result.to_scorer().as_ref(),
            &ds.test,
            &filter,
            n_ent,
        );
        println!(
            "{:<9} {first:>8.4}  {last:>9.4}  {:>7.4}  {:>7.4}",
            format!("{mt:?}"),
            metrics.mrr,
            metrics.hits_at_10
        );
    }
}

/// Batch size: the smallest full-batch divisor of the training set at or above 16.
fn batch_size(ds: &InternedDataset) -> usize {
    let n = ds.train.len();
    (16..=n).find(|b| n % b == 0).unwrap_or(n)
}

fn env(name: &str, default: f64) -> f64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn config(ds: &InternedDataset) -> BurnTrainConfig {
    BurnTrainConfig {
        dim: std::env::var("DIM").ok().and_then(|v| v.parse().ok()).unwrap_or(DIM),
        init_scale: env("INIT", 1e-3),
        lr: env("LR", 0.01),
        label_smoothing: 0.1,
        n3_reg: 0.0,
        batch_size: std::env::var("BATCH").ok().and_then(|v| v.parse().ok()).unwrap_or_else(|| batch_size(ds)),
        epochs: std::env::var("EPOCHS").ok().and_then(|v| v.parse().ok()).unwrap_or(EPOCHS),
        log_interval: 0,
    }
}

/// Fixed synthetic graph: `h --r--> h + 1 + r` for every head and relation that
/// stays inside the entity range, with every `TEST_EVERY`-th head row held out.
/// A forward chain is exactly representable by a translation, so all four models
/// can fit it and the ranking metric measures fit rather than the model family's
/// structural ceiling — the point is comparability, not realism.
fn synthetic_dataset() -> InternedDataset {
    let mut train = Vec::new();
    let mut test = Vec::new();
    for h in 0..N_ENT {
        for r in 0..N_REL {
            // Forward chain only: a wrapped cycle would force the relation
            // translations to zero, the one structure TransE cannot fit.
            if h + 1 + r >= N_ENT {
                continue;
            }
            let triple = Triple::new(
                format!("e{h:03}"),
                format!("r{r}"),
                format!("e{:03}", h + 1 + r),
            );
            if h % TEST_EVERY == TEST_EVERY - 1 {
                test.push(triple);
            } else {
                train.push(triple);
            }
        }
    }
    Dataset::new(train, Vec::new(), test).into_interned()
}
