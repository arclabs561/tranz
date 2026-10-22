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
//! Ranks are **realistic** (PyKEEN's default): each entity with a strictly
//! better (lower) score adds 1 to the rank and each entity tied with the
//! target adds 1/2, which is the expected rank under a uniformly random
//! order among ties. A model that scores everything the same therefore gets
//! MRR about 2/N, not 1.0. Versions up to 0.8 used optimistic ranks (ties
//! did not count).
//!
//! ## Reciprocal relations
//!
//! For a dataset augmented with
//! [`add_reciprocals`](crate::dataset::InternedDatasetExt::add_reciprocals),
//! use [`evaluate_link_prediction_reciprocal`], which ranks tails only, once
//! for `(h, r, ?)` and once for `(t, r_inv, ?)` (Lacroix et al. 2018).
//! [`evaluate_link_prediction`] ranks heads and tails of every test triple,
//! so on an augmented test split it would score each fact four times.
//!
//! ## Parallelism
//!
//! Evaluation is parallelized across test triples via rayon. Each test triple
//! is scored independently against all entities.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::dataset::{FilterIndex, TripleIds};
use crate::Scorer;

/// True when `competitor` strictly outranks `target` (lower score = better).
///
/// Non-finite scores are handled defensively so a broken/`NaN` model cannot
/// silently inflate MRR: a `NaN` competitor never beats anyone, and a `NaN`
/// target ranks worst (every finite competitor beats it) instead of slipping
/// to rank 1 because `NaN < x` is always false.
pub(crate) fn beats(competitor: f32, target: f32) -> bool {
    competitor.is_finite() && (target.is_nan() || competitor < target)
}

/// Running count for one realistic filtered rank.
#[derive(Default)]
struct RankCount {
    better: u32,
    tied: u32,
}

impl RankCount {
    fn add(&mut self, competitor: f32, target: f32) {
        if beats(competitor, target) {
            self.better += 1;
        } else if competitor.is_finite() && competitor == target {
            self.tied += 1;
        }
    }

    /// `1 + better + tied / 2`: the expected rank under a random order among ties.
    fn rank(&self) -> f64 {
        1.0 + self.better as f64 + 0.5 * self.tied as f64
    }
}

/// Realistic filtered rank of `target` within `scores`, skipping the target
/// itself and every id in `known`.
pub(crate) fn filtered_rank(
    scores: &[f32],
    target: usize,
    known: &std::collections::HashSet<usize>,
) -> f64 {
    let target_score = scores[target];
    let mut count = RankCount::default();
    for (id, &score) in scores.iter().enumerate() {
        if id != target && !known.contains(&id) {
            count.add(score, target_score);
        }
    }
    count.rank()
}

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
    let triple_ranks: Vec<(usize, f64, f64)> = test_triples
        .par_iter()
        .map(|triple| {
            let (h, r, t) = (triple.head, triple.relation, triple.tail);
            let tail_rank =
                filtered_rank(&model.score_all_tails(h, r), t, filter.known_tails(h, r));
            let head_rank =
                filtered_rank(&model.score_all_heads(r, t), h, filter.known_heads(r, t));
            (r, tail_rank, head_rank)
        })
        .collect();

    // Aggregate metrics with head/tail split.
    let tail_ranks: Vec<f64> = triple_ranks.iter().map(|&(_, tr, _)| tr).collect();
    let head_ranks: Vec<f64> = triple_ranks.iter().map(|&(_, _, hr)| hr).collect();
    let all_ranks: Vec<f64> = tail_ranks
        .iter()
        .chain(head_ranks.iter())
        .copied()
        .collect();
    let mut metrics = compute_metrics(&all_ranks);
    metrics.tail_mrr = mrr(&tail_ranks);
    metrics.head_mrr = mrr(&head_ranks);

    // Per-relation metrics.
    let mut per_rel_tail: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut per_rel_head: HashMap<usize, Vec<f64>> = HashMap::new();
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
        let all: Vec<f64> = tr.iter().chain(hr.iter()).copied().collect();
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

/// Evaluate a model trained with reciprocal relations (Lacroix et al. 2018).
///
/// `test_triples` is a test split augmented by
/// [`add_reciprocals`](crate::dataset::InternedDatasetExt::add_reciprocals),
/// and `num_relations` is the number of original relations (before
/// augmentation). Every triple is ranked by tail prediction only: a triple
/// with relation `r < num_relations` is the fact's tail query `(h, r, ?)`,
/// and its reciprocal `(t, r + num_relations, ?)` is the fact's head query.
/// `tail_mrr` and `head_mrr` report those two groups; `mrr` and the other
/// metrics average over both. Per-relation metrics are keyed by the original
/// relation id. Filtering and tie handling are as in
/// [`evaluate_link_prediction_detailed`].
pub fn evaluate_link_prediction_reciprocal(
    model: &(dyn Scorer + Sync),
    test_triples: &[TripleIds],
    filter: &FilterIndex,
    num_relations: usize,
) -> EvalResult {
    if test_triples.is_empty() {
        return EvalResult {
            metrics: Metrics::default(),
            per_relation: HashMap::new(),
        };
    }

    // (original relation, is_inverse, rank)
    let ranks: Vec<(usize, bool, f64)> = test_triples
        .par_iter()
        .map(|triple| {
            let (h, r, t) = (triple.head, triple.relation, triple.tail);
            let rank = filtered_rank(&model.score_all_tails(h, r), t, filter.known_tails(h, r));
            let inverse = r >= num_relations;
            let original = if inverse { r - num_relations } else { r };
            (original, inverse, rank)
        })
        .collect();

    let split = |rows: &mut dyn Iterator<Item = &(usize, bool, f64)>| {
        let (mut tail, mut head) = (Vec::new(), Vec::new());
        for &(_, inverse, rank) in rows {
            if inverse {
                head.push(rank);
            } else {
                tail.push(rank);
            }
        }
        let all: Vec<f64> = tail.iter().chain(head.iter()).copied().collect();
        let mut m = compute_metrics(&all);
        m.tail_mrr = mrr(&tail);
        m.head_mrr = mrr(&head);
        m
    };

    let metrics = split(&mut ranks.iter());
    let mut relations: Vec<usize> = ranks.iter().map(|&(r, _, _)| r).collect();
    relations.sort_unstable();
    relations.dedup();
    let per_relation = relations
        .into_iter()
        .map(|r| (r, split(&mut ranks.iter().filter(|row| row.0 == r))))
        .collect();

    EvalResult {
        metrics,
        per_relation,
    }
}

pub(crate) fn compute_metrics(ranks: &[f64]) -> Metrics {
    if ranks.is_empty() {
        return Metrics::default();
    }
    let n = ranks.len() as f64;
    let mrr_val = mrr(ranks);
    let mean_rank = ranks.iter().sum::<f64>() / n;
    let hits = |k: f64| ranks.iter().filter(|&&r| r <= k).count() as f64 / n;
    Metrics {
        mrr: mrr_val,
        head_mrr: 0.0,
        tail_mrr: 0.0,
        mean_rank: mean_rank as f32,
        hits_at_1: hits(1.0) as f32,
        hits_at_3: hits(3.0) as f32,
        hits_at_10: hits(10.0) as f32,
    }
}

pub(crate) fn mrr(ranks: &[f64]) -> f32 {
    if ranks.is_empty() {
        return 0.0;
    }
    let n = ranks.len() as f64;
    (ranks.iter().map(|&r| 1.0 / r).sum::<f64>() / n) as f32
}

/// Fast approximate evaluation using random negative sampling.
///
/// For each head and tail query, draws `min(num_candidates, num_entities)`
/// distinct entity IDs uniformly without replacement from all entity IDs.
/// The target and other known-true completions are then removed from the draw;
/// the target is scored separately, so fewer than `num_candidates` eligible
/// negatives may remain. Uses realistic midranks within that candidate subset.
///
/// These are **subset-ranking metrics**, not unbiased estimates of full-entity
/// MRR or Hits@k. A smaller candidate set can inflate metrics and change model
/// comparisons. Sampling all entities recovers the full filtered protocol;
/// otherwise report the candidate count and do not compare directly with
/// full-ranking benchmarks. Sampling is random on each call.
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

    let triple_ranks: Vec<(f64, f64)> = test_triples
        .par_iter()
        .map_init(rand::rng, |rng, triple| {
            use rand::seq::index::sample;
            let (h, r, t) = (triple.head, triple.relation, triple.tail);

            // Tail prediction: sample candidates + correct answer.
            let tail_rank = {
                let target_score = model.score(h, r, t);
                let known_tails = filter.known_tails(h, r);
                let candidates = sample(rng, num_entities, num_candidates.min(num_entities));
                let mut count = RankCount::default();
                for idx in candidates.iter() {
                    if idx == t || known_tails.contains(&idx) {
                        continue;
                    }
                    count.add(model.score(h, r, idx), target_score);
                }
                count.rank()
            };

            // Head prediction: sample candidates + correct answer.
            let head_rank = {
                let target_score = model.score(h, r, t);
                let known_heads = filter.known_heads(r, t);
                let candidates = sample(rng, num_entities, num_candidates.min(num_entities));
                let mut count = RankCount::default();
                for idx in candidates.iter() {
                    if idx == h || known_heads.contains(&idx) {
                        continue;
                    }
                    count.add(model.score(idx, r, t), target_score);
                }
                count.rank()
            };

            (tail_rank, head_rank)
        })
        .collect();

    let tail_ranks: Vec<f64> = triple_ranks.iter().map(|&(tr, _)| tr).collect();
    let head_ranks: Vec<f64> = triple_ranks.iter().map(|&(_, hr)| hr).collect();
    let all_ranks: Vec<f64> = tail_ranks
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

        // Every other tail of (0, 0, ?) and every other head of (?, 0, 0)
        // is a known triple, so both rankings filter all competitors. (The
        // head side used to pass only because optimistic ranks ignored ties.)
        let all = vec![
            tid(0, 0, 0),
            tid(0, 0, 1),
            tid(0, 0, 2),
            tid(0, 0, 3),
            tid(0, 0, 4),
            tid(1, 0, 0),
            tid(2, 0, 0),
            tid(3, 0, 0),
            tid(4, 0, 0),
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
    fn tie_breaking_is_realistic() {
        struct TiedModel;
        impl Scorer for TiedModel {
            fn score(&self, _h: usize, _r: usize, _t: usize) -> f32 {
                5.0
            }
            fn num_entities(&self) -> usize {
                3
            }
        }

        // Two unfiltered competitors tie with the target on each side, so
        // the realistic rank is 1 + 2/2 = 2 (optimistic would say 1).
        let test = vec![tid(0, 0, 1)];
        let filter = make_filter(&test);
        let metrics = evaluate_link_prediction(&TiedModel, &test, &filter, 3);
        assert!((metrics.mrr - 0.5).abs() < 1e-6, "MRR = {}", metrics.mrr);
        assert_eq!(metrics.hits_at_1, 0.0);
        assert!((metrics.mean_rank - 2.0).abs() < 1e-6);
    }

    /// A model that scores every triple the same carries no information,
    /// so its expected rank is (N + 1) / 2 and MRR is about 2/N, not 1.0.
    #[test]
    fn constant_score_model_has_mrr_about_two_over_n() {
        struct ConstantModel;
        const N: usize = 1000;
        impl Scorer for ConstantModel {
            fn score(&self, _h: usize, _r: usize, _t: usize) -> f32 {
                0.0
            }
            fn num_entities(&self) -> usize {
                N
            }
        }

        let test: Vec<_> = (0..20).map(|i| tid(i, 0, i + 1)).collect();
        let filter = make_filter(&test);
        let metrics = evaluate_link_prediction(&ConstantModel, &test, &filter, N);
        let expected = 2.0 / (N as f32 + 1.0);
        assert!(
            (metrics.mrr - expected).abs() < 1e-5,
            "MRR = {}, expected about 2/N = {expected}",
            metrics.mrr
        );
        assert_eq!(metrics.hits_at_10, 0.0);
        let sampled = evaluate_link_prediction_sampled(&ConstantModel, &test, &filter, N, N);
        assert!(
            (sampled.mrr - expected).abs() < 1e-5,
            "sampled MRR = {}",
            sampled.mrr
        );
    }

    /// Reciprocal protocol (Lacroix et al. 2018): each fact is ranked once
    /// as (h, r, ?) and once as (t, r_inv, ?), tails only. Head queries on
    /// the augmented split would rank each fact four times.
    #[test]
    fn reciprocal_eval_ranks_tails_only_once_per_direction() {
        // 1 original relation, so r_inv = 1. Fact (0, 0, 1).
        struct DirModel;
        impl Scorer for DirModel {
            fn score(&self, h: usize, r: usize, t: usize) -> f32 {
                match (h, r, t) {
                    (0, 0, 1) => 0.0, // forward query: target is best
                    (1, 1, 0) => 9.0, // inverse query: target is worst
                    _ => 1.0,
                }
            }
            fn num_entities(&self) -> usize {
                3
            }
        }

        let test = vec![tid(0, 0, 1), tid(1, 1, 0)];
        let filter = make_filter(&test);
        let result = evaluate_link_prediction_reciprocal(&DirModel, &test, &filter, 1);
        let m = result.metrics;
        assert!((m.tail_mrr - 1.0).abs() < 1e-6, "tail {}", m.tail_mrr);
        assert!((m.head_mrr - 1.0 / 3.0).abs() < 1e-6, "head {}", m.head_mrr);
        assert!(
            (m.mrr - (1.0 + 1.0 / 3.0) / 2.0).abs() < 1e-6,
            "mrr {}",
            m.mrr
        );
        assert_eq!(result.per_relation.len(), 1);
        assert!((result.per_relation[&0].mrr - m.mrr).abs() < 1e-6);
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
    fn sampled_subset_mrr_is_not_full_ranking_mrr() {
        let model = crate::DistMult::from_vecs(vec![vec![0.0]; 1000], vec![vec![0.0]], 1);
        let test = [tid(0, 0, 1)];
        let filter = make_filter(&test);
        let full = evaluate_link_prediction(&model, &test, &filter, 1000);
        let sampled = evaluate_link_prediction_sampled(&model, &test, &filter, 1000, 500);
        assert!((full.mrr - 2.0 / 1001.0).abs() < 1e-7);
        // For every RNG outcome, at most 500 ties enter either ranking:
        // rank <= 251. Thus these subset metrics cannot be unbiased full MRR.
        assert!(sampled.mrr >= 1.0 / 251.0);
        assert!(sampled.mrr > full.mrr);
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

    /// A NaN target score must not be treated as a perfect top-1 prediction.
    /// `score < NaN` is always `false`, so without NaN-safety every competitor
    /// "loses" to the target and MRR inflates to 1.0. With `beats`, a NaN
    /// target ranks worst (MRR minimal, not inflated).
    #[test]
    fn nan_target_ranks_worst_not_first() {
        struct NanTarget;
        impl Scorer for NanTarget {
            fn score(&self, h: usize, r: usize, t: usize) -> f32 {
                // Only the exact target triple (0, 0, 1) is NaN; everything
                // else is a finite 1.0. This isolates the tail-target and
                // head-target as NaN while competitors stay finite.
                if (h, r, t) == (0, 0, 1) {
                    f32::NAN
                } else {
                    1.0
                }
            }
            fn num_entities(&self) -> usize {
                3
            }
        }
        // Without the fix, `score < NaN` is false for every competitor, so the
        // NaN target stayed at rank 1 => MRR 1.0 / H@1 1.0 (inflated). With the
        // fix the NaN target ranks worst: both competitors beat it, so both the
        // head and tail predictions give rank 3 => MRR 1/3 and H@1 0.
        let test = vec![tid(0, 0, 1)];
        let filter = make_filter(&test);
        let metrics = evaluate_link_prediction(&NanTarget, &test, &filter, 3);
        assert!(
            (metrics.mrr - 1.0 / 3.0).abs() < 1e-5,
            "NaN target should rank worst: got MRR={} want ~0.333",
            metrics.mrr
        );
        assert!(
            (metrics.hits_at_1 - 0.0).abs() < 1e-6,
            "H@1 must be 0 for a NaN target"
        );
    }
}
