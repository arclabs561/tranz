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
        // With a zero relation vector, score(e, r, e) should be 0.
        let entities: Vec<Vec<f32>> = (0..N_ENT)
            .map(|_| vec![1.0; DIM])
            .collect();
        let relations = vec![vec![0.0; DIM]; N_REL];
        let model = TransE::from_vecs(entities, relations, DIM);
        let s = model.score_triple(e, 0, e);
        prop_assert!((s - 0.0).abs() < 1e-6, "h + 0 - h should be 0, got {s}");
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
        let relation_angles = vec![vec![0.0; DIM]; N_REL]; // angle=0 => identity
        let model = RotatE::from_vecs(entities, relation_angles, DIM, 12.0);
        let s = model.score_triple(e, 0, e);
        prop_assert!(s.abs() < 1e-5, "Identity rotation: score(e, 0, e) should be ~0, got {s}");
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
}
