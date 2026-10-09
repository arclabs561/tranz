//! `train_kge_seeded` is reproducible. Kept in its own test binary because
//! the ndarray backend RNG is process-wide and other tests would draw from
//! it concurrently.

use tranz::burn_train::{train_kge_seeded, BurnModelType, BurnTrainConfig};
use tranz::dataset::TripleIds;

type B = burn::backend::Autodiff<burn_ndarray::NdArray>;

#[test]
fn same_seed_gives_same_embeddings() {
    let triples: Vec<TripleIds> = (0..12)
        .map(|i| TripleIds::new(i % 6, i % 2, (i + 1) % 6))
        .collect();
    let config = BurnTrainConfig {
        dim: 8,
        epochs: 3,
        batch_size: 4,
        log_interval: 0,
        ..BurnTrainConfig::default()
    };
    let device = burn_ndarray::NdArrayDevice::Cpu;
    let run = |seed| {
        train_kge_seeded::<B>(
            &triples,
            6,
            2,
            BurnModelType::DistMult,
            &config,
            &device,
            seed,
        )
    };

    let a = run(7);
    let b = run(7);
    assert_eq!(a.entity_vecs, b.entity_vecs);
    assert_eq!(a.relation_vecs, b.relation_vecs);
    assert_eq!(a.losses, b.losses);

    let c = run(8);
    assert_ne!(a.entity_vecs, c.entity_vecs);
}
