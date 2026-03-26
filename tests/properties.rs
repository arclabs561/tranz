#![allow(missing_docs)]

use proptest::prelude::*;
use tranz::{ComplEx, DistMult, RotatE, Scorer, TransE};

const DIM: usize = 8;
const N_ENT: usize = 5;
const N_REL: usize = 3;

// ---------------------------------------------------------------------------
// TransE properties
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn transe_score_non_negative(
        h in 0..N_ENT,
        r in 0..N_REL,
        t in 0..N_ENT,
    ) {
        let model = TransE::new(N_ENT, N_REL, DIM);
        let s = model.score(h, r, t);
        prop_assert!(s >= 0.0, "TransE score must be >= 0, got {s}");
        prop_assert!(s.is_finite(), "TransE score must be finite, got {s}");
    }

    #[test]
    fn transe_self_score_zero_with_zero_relation(
        e in 0..N_ENT,
    ) {
        let entities: Vec<Vec<f32>> = (0..N_ENT)
            .map(|_| vec![1.0; DIM])
            .collect();
        let relations = vec![vec![0.0; DIM]; N_REL];
        let model = TransE::from_vecs(entities, relations, DIM);
        let s = model.score_triple(e, 0, e);
        prop_assert!((s - 0.0).abs() < 1e-6, "h + 0 - h should be 0, got {s}");
    }

    #[test]
    fn transe_score_all_tails_matches_individual(
        h in 0..N_ENT,
        r in 0..N_REL,
    ) {
        let model = TransE::new(N_ENT, N_REL, DIM);
        let all = model.score_all_tails(h, r);
        for (t, &score) in all.iter().enumerate() {
            let individual = model.score(h, r, t);
            prop_assert!(
                (score - individual).abs() < 1e-5,
                "score_all_tails[{t}]={score} vs score()={individual}"
            );
        }
    }

    #[test]
    fn transe_score_all_heads_matches_individual(
        r in 0..N_REL,
        t in 0..N_ENT,
    ) {
        let model = TransE::new(N_ENT, N_REL, DIM);
        let all = model.score_all_heads(r, t);
        for (h, &score) in all.iter().enumerate() {
            let individual = model.score(h, r, t);
            prop_assert!(
                (score - individual).abs() < 1e-5,
                "score_all_heads[{h}]={score} vs score()={individual}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// RotatE properties
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn rotate_score_non_negative(
        h in 0..N_ENT,
        r in 0..N_REL,
        t in 0..N_ENT,
    ) {
        let model = RotatE::new(N_ENT, N_REL, DIM, 12.0);
        let s = model.score(h, r, t);
        prop_assert!(s >= 0.0, "RotatE score must be >= 0, got {s}");
        prop_assert!(s.is_finite(), "RotatE score must be finite, got {s}");
    }

    #[test]
    fn rotate_identity_rotation_gives_zero(e in 0..N_ENT) {
        let entities: Vec<Vec<f32>> = (0..N_ENT)
            .map(|_| vec![1.0; DIM * 2])
            .collect();
        let relation_angles = vec![vec![0.0; DIM]; N_REL];
        let model = RotatE::from_vecs(entities, relation_angles, DIM, 12.0);
        let s = model.score_triple(e, 0, e);
        prop_assert!(s.abs() < 1e-5, "Identity rotation: score(e, 0, e) should be ~0, got {s}");
    }

    #[test]
    fn rotate_score_all_tails_matches_individual(
        h in 0..N_ENT,
        r in 0..N_REL,
    ) {
        let model = RotatE::new(N_ENT, N_REL, DIM, 12.0);
        let all = model.score_all_tails(h, r);
        for (t, &score) in all.iter().enumerate() {
            let individual = model.score(h, r, t);
            prop_assert!(
                (score - individual).abs() < 1e-4,
                "RotatE score_all_tails[{t}]={score} vs score()={individual}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// ComplEx properties
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn complex_score_is_finite(
        h in 0..N_ENT,
        r in 0..N_REL,
        t in 0..N_ENT,
    ) {
        let model = ComplEx::new(N_ENT, N_REL, DIM);
        let s = model.score(h, r, t);
        prop_assert!(s.is_finite(), "ComplEx score must be finite, got {s}");
    }

    #[test]
    fn complex_score_all_tails_matches_individual(
        h in 0..N_ENT,
        r in 0..N_REL,
    ) {
        let model = ComplEx::new(N_ENT, N_REL, DIM);
        let all = model.score_all_tails(h, r);
        for (t, &score) in all.iter().enumerate() {
            let individual = model.score(h, r, t);
            prop_assert!(
                (score - individual).abs() < 1e-4,
                "ComplEx score_all_tails[{t}]={score} vs score()={individual}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// DistMult properties
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn distmult_is_symmetric(
        h in 0..N_ENT,
        r in 0..N_REL,
        t in 0..N_ENT,
    ) {
        let model = DistMult::new(N_ENT, N_REL, DIM);
        let s_ht = model.score_triple(h, r, t);
        let s_th = model.score_triple(t, r, h);
        prop_assert!(
            (s_ht - s_th).abs() < 1e-4,
            "DistMult must be symmetric: score({h},r,{t})={s_ht} vs score({t},r,{h})={s_th}"
        );
    }

    #[test]
    fn distmult_score_is_finite(
        h in 0..N_ENT,
        r in 0..N_REL,
        t in 0..N_ENT,
    ) {
        let model = DistMult::new(N_ENT, N_REL, DIM);
        let s = model.score(h, r, t);
        prop_assert!(s.is_finite(), "DistMult score must be finite, got {s}");
    }

    #[test]
    fn distmult_score_all_tails_matches_individual(
        h in 0..N_ENT,
        r in 0..N_REL,
    ) {
        let model = DistMult::new(N_ENT, N_REL, DIM);
        let all = model.score_all_tails(h, r);
        for (t, &score) in all.iter().enumerate() {
            let individual = model.score(h, r, t);
            prop_assert!(
                (score - individual).abs() < 1e-4,
                "DistMult score_all_tails[{t}]={score} vs score()={individual}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Top-k properties (all models)
// ---------------------------------------------------------------------------

#[test]
fn top_k_returns_sorted_results() {
    let model = TransE::new(20, 3, 16);
    let top = model.top_k_tails(0, 0, 10);
    assert_eq!(top.len(), 10);
    for w in top.windows(2) {
        assert!(
            w[0].1 <= w[1].1,
            "top_k must be sorted ascending: {} > {}",
            w[0].1,
            w[1].1
        );
    }
}

#[test]
fn top_k_heads_returns_sorted_results() {
    let model = ComplEx::new(20, 3, 16);
    let top = model.top_k_heads(0, 0, 10);
    assert_eq!(top.len(), 10);
    for w in top.windows(2) {
        assert!(w[0].1 <= w[1].1);
    }
}

#[test]
fn top_k_best_is_actually_best() {
    // Verify the top-1 entity from top_k_tails matches a full scan.
    let model = DistMult::new(30, 3, 16);
    let top1 = model.top_k_tails(0, 0, 1);
    let all = model.score_all_tails(0, 0);
    let best_idx = all
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert_eq!(
        top1[0].0, best_idx,
        "top_k_tails[0] should match argmin of score_all_tails"
    );
}
