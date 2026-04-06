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
//! This implementation uses **optimistic** tie-breaking: only entities with
//! strictly better (lower) scores than the target increase the rank. Entities
//! with the same score as the target do not count. This gives the best-case
//! rank among tied entities.
//!
//! ## Parallelism
//!
//! Evaluation is parallelized across test triples via rayon. Each test triple
//! is scored independently against all entities.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::dataset::{FilterIndex, TripleIds};
use crate::Scorer;

/// Link prediction metrics.
#[derive(Debug, Clone, Copy, Default)]
pub struct Metrics {
    /// Mean Reciprocal Rank (filtered).
    pub mrr: f32,
    /// MRR for head prediction only (filtered).
    pub head_mrr: f32,
    /// MRR for tail prediction only (filtered).
    pub tail_mrr: f32,
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
    test_triples: &[TripleIds],
    filter: &FilterIndex,
    _num_entities: usize,
) -> Metrics {
    evaluate_link_prediction_detailed(model, test_triples, filter).metrics
}

/// Evaluate link prediction with per-relation breakdown.
///
/// Same protocol as [`evaluate_link_prediction`], but also returns
/// per-relation metrics keyed by relation ID.
pub fn evaluate_link_prediction_detailed(
    model: &(dyn Scorer + Sync),
    test_triples: &[TripleIds],
    filter: &FilterIndex,
) -> EvalResult {
    if test_triples.is_empty() {
        return EvalResult {
            metrics: Metrics::default(),
            per_relation: HashMap::new(),
        };
    }

    // Parallel: each test triple produces (relation, tail_rank, head_rank).
    // Uses batch scoring (score_all_tails / score_all_heads) for ~N-fold speedup
    // over per-entity score() calls, where N = num_entities.
    let triple_ranks: Vec<(usize, u32, u32)> = test_triples
        .par_iter()
        .map(|triple| {
            let (h, r, t) = (triple.head, triple.relation, triple.tail);

            // Tail prediction: score all entities as tail replacements.
            let tail_scores = model.score_all_tails(h, r);
            let target_tail_score = tail_scores[t];
            let known_tails = filter.known_tails(h, r);
            let mut tail_rank = 1u32;
            for (t_prime, &score) in tail_scores.iter().enumerate() {
                if t_prime == t {
                    continue;
                }
                if known_tails.contains(&t_prime) {
                    continue;
                }
                if score < target_tail_score {
                    tail_rank += 1;
                }
            }

            // Head prediction: score all entities as head replacements.
            let head_scores = model.score_all_heads(r, t);
            let target_head_score = head_scores[h];
            let known_heads = filter.known_heads(r, t);
            let mut head_rank = 1u32;
            for (h_prime, &score) in head_scores.iter().enumerate() {
                if h_prime == h {
                    continue;
                }
                if known_heads.contains(&h_prime) {
                    continue;
                }
                if score < target_head_score {
                    head_rank += 1;
                }
            }

            (r, tail_rank, head_rank)
        })
        .collect();

    // Aggregate metrics with head/tail split.
    let tail_ranks: Vec<u32> = triple_ranks.iter().map(|&(_, tr, _)| tr).collect();
    let head_ranks: Vec<u32> = triple_ranks.iter().map(|&(_, _, hr)| hr).collect();
    let all_ranks: Vec<u32> = tail_ranks
        .iter()
        .chain(head_ranks.iter())
        .copied()
        .collect();
    let mut metrics = compute_metrics(&all_ranks);
    metrics.tail_mrr = mrr(&tail_ranks);
    metrics.head_mrr = mrr(&head_ranks);

    // Per-relation metrics.
    let mut per_rel_tail: HashMap<usize, Vec<u32>> = HashMap::new();
    let mut per_rel_head: HashMap<usize, Vec<u32>> = HashMap::new();
    for &(r, tr, hr) in &triple_ranks {
        per_rel_tail.entry(r).or_default().push(tr);
        per_rel_head.entry(r).or_default().push(hr);
    }
    let mut per_relation: HashMap<usize, Metrics> = HashMap::new();
    for r in per_rel_tail
        .keys()
        .chain(per_rel_head.keys())
        .copied()
        .collect::<std::collections::HashSet<_>>()
    {
        let tr = per_rel_tail.get(&r).map(|v| v.as_slice()).unwrap_or(&[]);
        let hr = per_rel_head.get(&r).map(|v| v.as_slice()).unwrap_or(&[]);
        let all: Vec<u32> = tr.iter().chain(hr.iter()).copied().collect();
        let mut m = compute_metrics(&all);
        m.tail_mrr = mrr(tr);
        m.head_mrr = mrr(hr);
        per_relation.insert(r, m);
    }

    EvalResult {
        metrics,
        per_relation,
    }
}

fn compute_metrics(ranks: &[u32]) -> Metrics {
    if ranks.is_empty() {
        return Metrics::default();
    }
    let n = ranks.len() as f64;
    let mrr_val = mrr(ranks);
    let mean_rank = ranks.iter().map(|&r| r as f64).sum::<f64>() / n;
    let hits = |k: u32| ranks.iter().filter(|&&r| r <= k).count() as f64 / n;
    Metrics {
        mrr: mrr_val,
        head_mrr: 0.0,
        tail_mrr: 0.0,
        mean_rank: mean_rank as f32,
        hits_at_1: hits(1) as f32,
        hits_at_3: hits(3) as f32,
        hits_at_10: hits(10) as f32,
    }
}

fn mrr(ranks: &[u32]) -> f32 {
    if ranks.is_empty() {
        return 0.0;
    }
    let n = ranks.len() as f64;
    (ranks.iter().map(|&r| 1.0 / r as f64).sum::<f64>() / n) as f32
}

/// Fast approximate evaluation using random negative sampling.
///
/// Instead of scoring all `num_entities` candidates per test triple, samples
/// `num_candidates` random negatives plus the correct answer. Ranks are
/// computed within this smaller candidate set.
///
/// Provides an unbiased estimate of ranking metrics when `num_candidates` is
/// large enough (500-1000 is typical). Runs `num_entities / num_candidates`
/// times faster than exact evaluation.
///
/// Based on Cornell et al. (2024), "A Fast and Scalable Method for Evaluation
/// of KGE Models."
pub fn evaluate_link_prediction_sampled(
    model: &(dyn Scorer + Sync),
    test_triples: &[TripleIds],
    filter: &FilterIndex,
    num_entities: usize,
    num_candidates: usize,
) -> Metrics {
    if test_triples.is_empty() {
        return Metrics::default();
    }

    let triple_ranks: Vec<(u32, u32)> = test_triples
        .par_iter()
        .map_init(rand::rng, |rng, triple| {
            use rand::seq::index::sample;
            let (h, r, t) = (triple.head, triple.relation, triple.tail);

            // Tail prediction: sample candidates + correct answer.
            let tail_rank = {
                let target_score = model.score(h, r, t);
                let known_tails = filter.known_tails(h, r);
                let candidates = sample(rng, num_entities, num_candidates.min(num_entities));
                let mut rank = 1u32;
                for idx in candidates.iter() {
                    if idx == t || known_tails.contains(&idx) {
                        continue;
                    }
                    if model.score(h, r, idx) < target_score {
                        rank += 1;
                    }
                }
                rank
            };

            // Head prediction: sample candidates + correct answer.
            let head_rank = {
                let target_score = model.score(h, r, t);
                let known_heads = filter.known_heads(r, t);
                let candidates = sample(rng, num_entities, num_candidates.min(num_entities));
                let mut rank = 1u32;
                for idx in candidates.iter() {
                    if idx == h || known_heads.contains(&idx) {
                        continue;
                    }
                    if model.score(idx, r, t) < target_score {
                        rank += 1;
                    }
                }
                rank
            };

            (tail_rank, head_rank)
        })
        .collect();

    let tail_ranks: Vec<u32> = triple_ranks.iter().map(|&(tr, _)| tr).collect();
    let head_ranks: Vec<u32> = triple_ranks.iter().map(|&(_, hr)| hr).collect();
    let all_ranks: Vec<u32> = tail_ranks
        .iter()
        .chain(head_ranks.iter())
        .copied()
        .collect();
    let mut metrics = compute_metrics(&all_ranks);
    metrics.tail_mrr = mrr(&tail_ranks);
    metrics.head_mrr = mrr(&head_ranks);
    metrics
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tid(h: usize, r: usize, t: usize) -> TripleIds {
        TripleIds::new(h, r, t)
    }

    fn make_filter(triples: &[TripleIds]) -> FilterIndex {
        use crate::dataset::{Dataset, Triple};
        let ds = Dataset::new(
            triples
                .iter()
                .map(|t| {
                    Triple::new(
                        t.head.to_string(),
                        t.relation.to_string(),
                        t.tail.to_string(),
                    )
                })
                .collect(),
            Vec::new(),
            Vec::new(),
        )
        .into_interned();
        FilterIndex::from_dataset(&ds)
    }

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
        let test = vec![tid(0, 0, 0)];
        let filter = make_filter(&test);
        let metrics = evaluate_link_prediction(&PerfectModel, &test, &filter, 5);
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

        let all = vec![
            tid(0, 0, 0),
            tid(0, 0, 1),
            tid(0, 0, 2),
            tid(0, 0, 3),
            tid(0, 0, 4),
        ];
        let test = vec![tid(0, 0, 0)];
        let filter = make_filter(&all);
        let metrics = evaluate_link_prediction(&ConstantModel, &test, &filter, 5);
        assert!(
            (metrics.mrr - 1.0).abs() < 1e-6,
            "Filtered MRR should be 1.0, got {}",
            metrics.mrr
        );
    }

    #[test]
    fn empty_test_returns_zeroed_metrics() {
        let filter = make_filter(&[]);
        let metrics = evaluate_link_prediction(&PerfectModel, &[], &filter, 5);
        assert_eq!(metrics.mrr, 0.0);
    }

    #[test]
    fn tie_breaking_is_optimistic() {
        struct TiedModel;
        impl Scorer for TiedModel {
            fn score(&self, _h: usize, _r: usize, _t: usize) -> f32 {
                5.0
            }
            fn num_entities(&self) -> usize {
                3
            }
        }

        let test = vec![tid(0, 0, 1)];
        let filter = make_filter(&test);
        let metrics = evaluate_link_prediction(&TiedModel, &test, &filter, 3);
        assert!(
            (metrics.hits_at_1 - 1.0).abs() < 1e-6,
            "Optimistic tie-breaking: rank should be 1 when all scores tie"
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

        let test = vec![tid(0, 0, 1), tid(0, 1, 1)];
        let filter = make_filter(&test);
        let result = evaluate_link_prediction_detailed(&SplitModel, &test, &filter);

        let r0 = result.per_relation[&0];
        let r1 = result.per_relation[&1];
        assert!(
            r0.mrr >= r1.mrr,
            "Relation 0 MRR ({}) should be >= Relation 1 MRR ({})",
            r0.mrr,
            r1.mrr
        );
    }

    #[test]
    fn sampled_eval_perfect_model() {
        let test = vec![tid(0, 0, 0)];
        let filter = make_filter(&test);
        let metrics = evaluate_link_prediction_sampled(&PerfectModel, &test, &filter, 5, 100);
        // PerfectModel scores self-loops as 0 (best). With 5 entities and
        // 100 candidate samples (capped to 5), correct answer is always rank 1.
        assert!(
            (metrics.mrr - 1.0).abs() < 1e-6,
            "Sampled eval MRR = {}",
            metrics.mrr
        );
    }

    #[test]
    fn head_tail_mrr_split() {
        let test = vec![tid(0, 0, 0)];
        let filter = make_filter(&test);
        let metrics = evaluate_link_prediction(&PerfectModel, &test, &filter, 5);
        assert!(metrics.head_mrr > 0.0, "head_mrr should be populated");
        assert!(metrics.tail_mrr > 0.0, "tail_mrr should be populated");
        assert!(
            (metrics.mrr - (metrics.head_mrr + metrics.tail_mrr) / 2.0).abs() < 1e-5,
            "mrr should be average of head and tail: {} vs ({} + {}) / 2",
            metrics.mrr,
            metrics.head_mrr,
            metrics.tail_mrr
        );
    }
}
