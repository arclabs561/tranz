//! `train_kge_seeded` is reproducible. Kept in its own test binary because
//! the `burn-flex` backend RNG is process-wide and other tests would draw from
//! it concurrently.

use tranz::burn_temporal::train_tcomplex_seeded;
use tranz::burn_train::{train_complex_seeded, train_kge_seeded, BurnModelType, BurnTrainConfig};
use tranz::dataset::TripleIds;
use tranz::temporal::QuadIds;

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
    let device = burn::tensor::Device::flex();
    let run = |seed| {
        train_kge_seeded(
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

    // Keep every seeded call in this one test: flex's RNG is process-wide.
    singleton_batches_preserve_the_batch_axis();
}

fn assert_training_output(
    entities: &[Vec<f32>],
    relations: &[Vec<f32>],
    losses: &[f32],
    entity_width: usize,
    relation_width: usize,
) {
    assert_eq!(losses.len(), 1);
    assert!(losses[0].is_finite());
    assert_eq!(entities.len(), 3);
    assert_eq!(relations.len(), 1);
    for (rows, width) in [(entities, entity_width), (relations, relation_width)] {
        for row in rows {
            assert_eq!(row.len(), width);
            assert!(row.iter().all(|value| value.is_finite()));
        }
    }
}

fn singleton_batches_preserve_the_batch_axis() {
    let triples = [
        TripleIds::new(0, 0, 1),
        TripleIds::new(1, 0, 2),
        TripleIds::new(2, 0, 0),
    ];
    let quads = [
        QuadIds::new(0, 0, 1, 0),
        QuadIds::new(1, 0, 2, 1),
        QuadIds::new(2, 0, 0, 0),
    ];
    let device = burn::tensor::Device::flex();
    // Also cover a singleton remainder after a full batch, so reshaping must
    // use the actual batch length rather than the configured batch size.
    for batch_size in [1, 2] {
        let config = BurnTrainConfig {
            dim: 2,
            epochs: 1,
            batch_size,
            log_interval: 0,
            ..BurnTrainConfig::default()
        };
        let legacy = train_complex_seeded(&triples, 3, 1, &config, &device, 17);
        assert_training_output(
            &legacy.entity_vecs,
            &legacy.relation_vecs,
            &legacy.losses,
            4,
            4,
        );
        for (model, entity_width, relation_width) in [
            (BurnModelType::TransE, 2, 2),
            (BurnModelType::RotatE, 4, 2),
            (BurnModelType::ComplEx, 4, 4),
            (BurnModelType::DistMult, 2, 2),
        ] {
            let result = train_kge_seeded(&triples, 3, 1, model, &config, &device, 17);
            assert_training_output(
                &result.entity_vecs,
                &result.relation_vecs,
                &result.losses,
                entity_width,
                relation_width,
            );
        }
        let temporal = train_tcomplex_seeded(&quads, 3, 1, 2, &config, 0.01, &device, 17);
        assert_training_output(
            &temporal.entity_vecs,
            &temporal.relation_vecs,
            &temporal.losses,
            4,
            4,
        );
        assert_eq!(temporal.time_vecs.len(), 2);
        for row in &temporal.time_vecs {
            assert_eq!(row.len(), 4);
            assert!(row.iter().all(|value| value.is_finite()));
        }
    }
}
