//! Link prediction evaluation with filtered ranking.
//!
//! Computes MRR and Hits@k in the **filtered** setting: for each test triple
//! `(h, r, t)`, rank all entities as head replacements and tail replacements,
//! excluding other known-true triples from the ranking.
//!
//! "Filtered" means that when computing the rank of a correct entity, we
//! remove all other entities that also form known-true triples from the
//! candidate set. This prevents penalizing a model for ranking another
//! correct answer above the target. See Bordes et al. (2013) for the
//! original protocol.
//!
//! ## Tie-breaking
//!
//! This implementation uses **pessimistic** tie-breaking: an entity with the
//! same score as the target does *not* increase the rank. Only entities with
//! strictly better (lower) scores count. This is equivalent to "best-case
//! rank among tied entities."
//!
//! ## Parallelism
//!
//! Evaluation is parallelized across test triples via rayon. Each test triple
//! is scored independently against all entities.

use std::collections::{HashMap, HashSet};

use rayon::prelude::*;

use crate::Scorer;

/// Link prediction metrics.
#[derive(Debug, Clone, Copy, Default)]
pub struct Metrics {
    /// Mean Reciprocal Rank (filtered).
    pub mrr: f32,
    /// Mean Rank (filtered). Lower = better.
    pub mean_rank: f32,
    /// Hits@1 (filtered).
    pub hits_at_1: f32,
    /// Hits@3 (filtered).
    pub hits_at_3: f32,
    /// Hits@10 (filtered).
    pub hits_at_10: f32,
}

/// Evaluation results with optional per-relation breakdown.
#[derive(Debug, Clone)]
pub struct EvalResult {
    /// Aggregate metrics over all test triples.
    pub metrics: Metrics,
    /// Per-relation metrics, keyed by relation ID.
    pub per_relation: HashMap<usize, Metrics>,
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
/// Returns zeroed [`Metrics`] if `test_triples` is empty.
///
/// Evaluation is parallelized across test triples via rayon.
pub fn evaluate_link_prediction(
    model: &(dyn Scorer + Sync),
    test_triples: &[(usize, usize, usize)],
    all_triples: &[(usize, usize, usize)],
    num_entities: usize,
) -> Metrics {
    evaluate_link_prediction_detailed(model, test_triples, all_triples, num_entities).metrics
}

/// Evaluate link prediction with per-relation breakdown.
///
/// Same protocol as [`evaluate_link_prediction`], but also returns
/// per-relation metrics keyed by relation ID.
pub fn evaluate_link_prediction_detailed(
    model: &(dyn Scorer + Sync),
    test_triples: &[(usize, usize, usize)],
    all_triples: &[(usize, usize, usize)],
    _num_entities: usize,
) -> EvalResult {
    if test_triples.is_empty() {
        return EvalResult {
            metrics: Metrics::default(),
            per_relation: HashMap::new(),
        };
    }

    let known: HashSet<(usize, usize, usize)> = all_triples.iter().copied().collect();

    // Parallel: each test triple produces two (relation, rank) pairs.
    let rel_ranks: Vec<(usize, u32)> = test_triples
        .par_iter()
        .flat_map_iter(|&(h, r, t)| {
            let target_score = model.score(h, r, t);

            // Tail prediction.
            let mut tail_rank = 1u32;
            for t_prime in 0..model.num_entities() {
                if t_prime == t {
                    continue;
                }
                if known.contains(&(h, r, t_prime)) {
                    continue;
                }
                if model.score(h, r, t_prime) < target_score {
                    tail_rank += 1;
                }
            }

            // Head prediction.
            let mut head_rank = 1u32;
            for h_prime in 0..model.num_entities() {
                if h_prime == h {
                    continue;
                }
                if known.contains(&(h_prime, r, t)) {
                    continue;
                }
                if model.score(h_prime, r, t) < target_score {
                    head_rank += 1;
                }
            }

            [(r, tail_rank), (r, head_rank)]
        })
        .collect();

    // Aggregate metrics.
    let metrics = compute_metrics(&rel_ranks.iter().map(|&(_, rank)| rank).collect::<Vec<_>>());

    // Per-relation metrics.
    let mut per_rel: HashMap<usize, Vec<u32>> = HashMap::new();
    for &(r, rank) in &rel_ranks {
        per_rel.entry(r).or_default().push(rank);
    }
    let per_relation: HashMap<usize, Metrics> = per_rel
        .into_iter()
        .map(|(r, ranks)| (r, compute_metrics(&ranks)))
        .collect();

    EvalResult {
        metrics,
        per_relation,
    }
}

fn compute_metrics(ranks: &[u32]) -> Metrics {
    if ranks.is_empty() {
        return Metrics::default();
    }
    let usize_ranks: Vec<usize> = ranks.iter().map(|&r| r as usize).collect();
    Metrics {
        mrr: lattix::metrics::mean_reciprocal_rank(&usize_ranks) as f32,
        mean_rank: lattix::metrics::mean_rank(&usize_ranks) as f32,
        hits_at_1: lattix::metrics::hits_at_k(&usize_ranks, 1) as f32,
        hits_at_3: lattix::metrics::hits_at_k(&usize_ranks, 3) as f32,
        hits_at_10: lattix::metrics::hits_at_k(&usize_ranks, 10) as f32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct PerfectModel;

    impl Scorer for PerfectModel {
        fn score(&self, h: usize, _r: usize, t: usize) -> f32 {
            if h == t {
                0.0
            } else {
                10.0
            }
        }

        fn num_entities(&self) -> usize {
            5
        }
    }

    #[test]
    fn perfect_model_gets_mrr_1() {
        let test = vec![(0, 0, 0)];
        let all = vec![(0, 0, 0)];
        let metrics = evaluate_link_prediction(&PerfectModel, &test, &all, 5);
        assert!((metrics.mrr - 1.0).abs() < 1e-6, "MRR = {}", metrics.mrr);
        assert!((metrics.hits_at_1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn filtering_excludes_known_triples() {
        struct ConstantModel;
        impl Scorer for ConstantModel {
            fn score(&self, _h: usize, _r: usize, _t: usize) -> f32 {
                5.0
            }
            fn num_entities(&self) -> usize {
                5
            }
        }

        let all = vec![(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4)];
        let test = vec![(0, 0, 0)];
        let metrics = evaluate_link_prediction(&ConstantModel, &test, &all, 5);
        assert!(
            (metrics.mrr - 1.0).abs() < 1e-6,
            "Filtered MRR should be 1.0, got {}",
            metrics.mrr
        );
    }

    #[test]
    fn empty_test_returns_zeroed_metrics() {
        let metrics = evaluate_link_prediction(&PerfectModel, &[], &[], 5);
        assert_eq!(metrics.mrr, 0.0);
    }

    #[test]
    fn tie_breaking_is_pessimistic() {
        struct TiedModel;
        impl Scorer for TiedModel {
            fn score(&self, _h: usize, _r: usize, _t: usize) -> f32 {
                5.0
            }
            fn num_entities(&self) -> usize {
                3
            }
        }

        let test = vec![(0, 0, 1)];
        let all = vec![(0, 0, 1)];
        let metrics = evaluate_link_prediction(&TiedModel, &test, &all, 3);
        assert!(
            (metrics.hits_at_1 - 1.0).abs() < 1e-6,
            "Pessimistic tie-breaking: rank should be 1 when all scores tie"
        );
    }

    #[test]
    fn per_relation_breakdown() {
        struct SplitModel;
        impl Scorer for SplitModel {
            fn score(&self, _h: usize, r: usize, t: usize) -> f32 {
                if r == 0 {
                    if t == 1 {
                        0.0
                    } else {
                        10.0
                    }
                } else {
                    5.0
                }
            }
            fn num_entities(&self) -> usize {
                3
            }
        }

        let test = vec![(0, 0, 1), (0, 1, 1)];
        let all = vec![(0, 0, 1), (0, 1, 1)];
        let result = evaluate_link_prediction_detailed(&SplitModel, &test, &all, 3);

        let r0 = result.per_relation[&0];
        let r1 = result.per_relation[&1];
        assert!(
            r0.mrr >= r1.mrr,
            "Relation 0 MRR ({}) should be >= Relation 1 MRR ({})",
            r0.mrr,
            r1.mrr
        );
    }
}
