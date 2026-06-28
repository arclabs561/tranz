//! End-to-end: train tranz embeddings with Burn, then index + query them in
//! vicinity (HNSW nearest-neighbour search).
//!
//! tranz produces *point* embeddings, so a vector ANN index (vicinity) is the
//! natural serving layer (subsume's *region* embeddings go to precinct instead).
//! This trains DistMult on WN18RR, builds an HNSW index over the entity vectors,
//! and runs a few nearest-neighbour queries by cosine similarity.
//!
//! Data-gated: exits 0 if `data/WN18RR` is absent.
//!
//! Run on Metal: `cargo run --release --features "burn-cpu,burn-gpu" --example wn18rr_vicinity`
//! (drop `burn-gpu` for CPU ndarray).
//!
//! Sample output:
//! ```text
//! indexed 40943 entity vectors in vicinity HNSW
//! nearest entities by cosine similarity (id: [neighbours]):
//!       0: [13635(0.683), 24364(0.662), 14327(0.657), 5495(0.653), 29144(0.645)]
//!     100: [27053(0.585), 28238(0.555), 28369(0.548), 31452(0.541), 20207(0.530)]
//!    1000: [222(0.695), 28193(0.671), 16838(0.609), 17370(0.590), 8282(0.575)]
//!    5000: [33555(0.874), 33955(0.866), 39382(0.857), 3316(0.851), 37502(0.828)]
//! ```

#![allow(missing_docs)]

use std::path::Path;

use tranz::burn_train::{train_kge, BurnModelType, BurnTrainConfig};
use tranz::dataset::{self, InternedDatasetExt};
use vicinity::hnsw::HNSWIndex;

// Metal/Vulkan via wgpu when `burn-gpu` is enabled, else CPU ndarray.
#[cfg(feature = "burn-gpu")]
type B = burn::backend::Autodiff<burn_wgpu::Wgpu>;
#[cfg(not(feature = "burn-gpu"))]
type B = burn::backend::Autodiff<burn_ndarray::NdArray>;

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter().map(|x| x / norm).collect()
}

fn main() -> vicinity::Result<()> {
    let data = Path::new("data/WN18RR");
    if !data.join("train.txt").exists() {
        eprintln!("data/WN18RR not found; skipping. no-op.");
        return Ok(());
    }

    let ds = dataset::load_dataset(data).unwrap();
    let mut interned = ds.into_interned();
    interned.add_reciprocals();
    let n_ent = interned.num_entities();
    let n_rel = interned.num_relations();
    eprintln!(
        "WN18RR: {n_ent} entities, {n_rel} relations, {} triples",
        interned.train.len()
    );

    // Train point embeddings with Burn.
    let dim = 50;
    let config = BurnTrainConfig {
        dim,
        lr: 0.005,
        label_smoothing: 0.1,
        batch_size: 1000,
        epochs: 3,
        log_interval: 0,
        ..BurnTrainConfig::default()
    };
    eprintln!("training DistMult (dim={dim}) ...");
    let result = train_kge::<B>(
        &interned.train,
        n_ent,
        n_rel,
        BurnModelType::DistMult,
        &config,
        &device_default(),
    );
    eprintln!(
        "trained: loss {:.3} -> {:.3}",
        result.losses[0],
        result.losses.last().unwrap()
    );

    // Index the entity embeddings in vicinity (HNSW, cosine).
    let mut index = HNSWIndex::new(dim, 16, 64)?;
    for (id, vec) in result.entity_vecs.iter().enumerate() {
        index.add(id as u32, normalize(vec))?;
    }
    index.build()?;
    eprintln!(
        "indexed {} entity vectors in vicinity HNSW",
        result.entity_vecs.len()
    );

    // Nearest-neighbour queries for a few entities.
    println!("\nnearest entities by cosine similarity (id: [neighbours]):");
    for &q in &[0u32, 100, 1000, 5000] {
        let query = normalize(&result.entity_vecs[q as usize]);
        let hits = index.search(&query, 6, 64)?;
        let neighbours: Vec<String> = hits
            .iter()
            .filter(|(id, _)| *id != q)
            .take(5)
            .map(|(id, dist)| format!("{id}({:.3})", 1.0 - dist))
            .collect();
        println!("  {q:>5}: [{}]", neighbours.join(", "));
        // Self-retrieval sanity: an entity's own vector is its nearest neighbour.
        assert_eq!(hits[0].0, q, "self-retrieval failed for entity {q}");
    }
    println!(
        "\ntranz Burn embeddings index and query end-to-end in vicinity (self-retrieval holds)."
    );
    Ok(())
}

#[cfg(feature = "burn-gpu")]
fn device_default() -> <B as burn::tensor::backend::Backend>::Device {
    burn_wgpu::WgpuDevice::default()
}
#[cfg(not(feature = "burn-gpu"))]
fn device_default() -> <B as burn::tensor::backend::Backend>::Device {
    burn_ndarray::NdArrayDevice::Cpu
}
