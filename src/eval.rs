//! Link prediction evaluation with filtered ranking.
//!
//! Computes MRR and Hits@k in the filtered setting: for each test triple
//! `(h, r, t)`, rank all entities as head replacements and tail replacements,
//! excluding other known-true triples from the ranking.

use std::collections::HashSet;

use crate::Scorer;

/// Link prediction metrics.
#[derive(Debug, Clone, Copy)]
pub struct Metrics {
    /// Mean Reciprocal Rank (filtered).
    pub mrr: f32,
    /// Hits@1 (filtered).
    pub hits_at_1: f32,
    /// Hits@3 (filtered).
    pub hits_at_3: f32,
    /// Hits@10 (filtered).
    pub hits_at_10: f32,
}

/// Evaluate link prediction on test triples in the filtered setting.
///
/// `all_triples` should include train + valid + test triples (used to filter
/// known-true triples from the ranking). Each triple is `(head, relation, tail)`.
///
/// For each test triple `(h, r, t)`:
/// - **Tail prediction**: rank all entities `t'` by `score(h, r, t')`, filter known `(h, r, *)`.
/// - **Head prediction**: rank all entities `h'` by `score(h', r, t)`, filter known `(*, r, t)`.
///
/// Lower scores are better (distance-based models).
pub fn evaluate_link_prediction(
    model: &dyn Scorer,
    test_triples: &[(usize, usize, usize)],
    all_triples: &[(usize, usize, usize)],
    num_entities: usize,
) -> Metrics {
    let known: HashSet<(usize, usize, usize)> = all_triples.iter().copied().collect();

    let mut reciprocal_ranks = Vec::with_capacity(test_triples.len() * 2);

    for &(h, r, t) in test_triples {
        // Tail prediction: fix (h, r), rank all t'
        let target_score = model.score(h, r, t);
        let mut rank = 1u32;
        for t_prime in 0..num_entities {
            if t_prime == t {
                continue;
            }
            if known.contains(&(h, r, t_prime)) {
                continue; // filtered
            }
            if model.score(h, r, t_prime) < target_score {
                rank += 1;
            }
        }
        reciprocal_ranks.push(rank);

        // Head prediction: fix (r, t), rank all h'
        let mut rank = 1u32;
        for h_prime in 0..num_entities {
            if h_prime == h {
                continue;
            }
            if known.contains(&(h_prime, r, t)) {
                continue; // filtered
            }
            if model.score(h_prime, r, t) < target_score {
                rank += 1;
            }
        }
        reciprocal_ranks.push(rank);
    }

    let n = reciprocal_ranks.len() as f32;
    let mrr = reciprocal_ranks.iter().map(|&r| 1.0 / r as f32).sum::<f32>() / n;
    let hits_at_1 = reciprocal_ranks.iter().filter(|&&r| r <= 1).count() as f32 / n;
    let hits_at_3 = reciprocal_ranks.iter().filter(|&&r| r <= 3).count() as f32 / n;
    let hits_at_10 = reciprocal_ranks.iter().filter(|&&r| r <= 10).count() as f32 / n;

    Metrics {
        mrr,
        hits_at_1,
        hits_at_3,
        hits_at_10,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trivial model: entity 0 is always closest to itself.
    struct PerfectModel;

    impl Scorer for PerfectModel {
        fn score(&self, h: usize, _r: usize, t: usize) -> f32 {
            if h == t { 0.0 } else { 10.0 }
        }

        fn num_entities(&self) -> usize {
            5
        }
    }

    #[test]
    fn perfect_model_gets_mrr_1() {
        // Test triple: (0, 0, 0) -- head == tail, so score = 0, all others = 10.
        let test = vec![(0, 0, 0)];
        let all = vec![(0, 0, 0)];
        let metrics = evaluate_link_prediction(&PerfectModel, &test, &all, 5);
        assert!((metrics.mrr - 1.0).abs() < 1e-6, "MRR = {}", metrics.mrr);
        assert!((metrics.hits_at_1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn filtering_excludes_known_triples() {
        // Model scores everything equally -- without filtering, rank would be
        // num_entities. With filtering of known triples, rank shrinks.
        struct ConstantModel;
        impl Scorer for ConstantModel {
            fn score(&self, _h: usize, _r: usize, _t: usize) -> f32 {
                5.0
            }
            fn num_entities(&self) -> usize {
                5
            }
        }

        // All entities except 0 are known tails for (0, 0, *).
        let all = vec![(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4)];
        let test = vec![(0, 0, 0)];
        let metrics = evaluate_link_prediction(&ConstantModel, &test, &all, 5);
        // All other tails are filtered out, so tail rank = 1.
        // Head prediction: (0,0,0) is known, others are not. All score equal => rank 1.
        assert!(
            (metrics.mrr - 1.0).abs() < 1e-6,
            "Filtered MRR should be 1.0, got {}",
            metrics.mrr
        );
    }
}
