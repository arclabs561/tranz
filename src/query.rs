//! Fuzzy logic query answering (CQD-Beam).
//!
//! Answers multi-hop conjunctive/disjunctive queries by decomposing them
//! into atomic link prediction calls and aggregating with t-norms.
//!
//! No complex-query training needed: uses a pretrained [`Scorer`] directly.
//!
//! Based on Arakelyan et al. (2021), "Complex Query Answering with Neural
//! Link Predictors" (ICLR, Outstanding Paper).
//!
//! ## Supported query types
//!
//! | Pattern | Structure | Example |
//! |---|---|---|
//! | 1p | `a -r-> ?` | one-hop link prediction |
//! | 2p | `a -r1-> V -r2-> ?` | two-hop chain |
//! | 3p | `a -r1-> V1 -r2-> V2 -r3-> ?` | three-hop chain |
//! | 2i | `(a1 -r1-> ?) AND (a2 -r2-> ?)` | intersection |
//! | 3i | `(a1 -r1-> ?) AND ... AND (a3 -r3-> ?)` | 3-way intersection |
//! | pi | `(a1 -r1-> V AND a2 -r2-> V) -r3-> ?` | intersect then project |
//! | ip | `(a1 -r1-> V -r2-> ?) AND (a2 -r3-> ?)` | project then intersect |
//! | 2u | `(a1 -r1-> ?) OR (a2 -r2-> ?)` | union |
//! | up | `(a1 -r1-> ? OR a2 -r2-> ?) -r3-> ?` | union then project |
//!
//! Negation queries (2in, 3in, inp, pin, pni) are also supported via
//! the fuzzy complement `1 - score`.
//!
//! ## Example
//!
//! ```no_run
//! use tranz::{DistMult, Scorer};
//! use tranz::query::{Query, QueryConfig, answer_query_topk};
//!
//! let model = DistMult::new(100, 10, 200);
//!
//! // 2-hop chain: entity 0 -rel 0-> V -rel 1-> ?
//! let query = Query::anchor(0, 0).then(1);
//! let top10 = answer_query_topk(&model, &query, &QueryConfig::default(), 10);
//!
//! // Intersection: (entity 0 -rel 0-> ?) AND (entity 1 -rel 1-> ?)
//! let query = Query::intersection(vec![
//!     Query::anchor(0, 0),
//!     Query::anchor(1, 1),
//! ]);
//! let top10 = answer_query_topk(&model, &query, &QueryConfig::default(), 10);
//! ```

use crate::Scorer;

/// T-norm for combining conjunctive (AND) scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TNorm {
    /// Gödel: `min(x, y)`. Best for intersection queries.
    Min,
    /// Product: `x * y`. Best for chain queries.
    Product,
}

/// Configuration for query answering.
#[derive(Debug, Clone)]
pub struct QueryConfig {
    /// T-norm for chain projection (hop) operations. Default: [`TNorm::Product`].
    ///
    /// Product tends to work better for chains (2p, 3p) because it
    /// propagates score magnitude through hops.
    pub t_norm_projection: TNorm,
    /// T-norm for intersection (AND) operations. Default: [`TNorm::Min`].
    ///
    /// Min tends to work better for intersections (2i, 3i) because it
    /// selects the weakest evidence.
    pub t_norm_intersection: TNorm,
    /// Beam width for intermediate variable search. Default: 128.
    ///
    /// Higher values improve recall at the cost of `O(k * |E|)` per hop.
    pub beam_k: usize,
}

impl Default for QueryConfig {
    fn default() -> Self {
        Self {
            t_norm_projection: TNorm::Product,
            t_norm_intersection: TNorm::Min,
            beam_k: 128,
        }
    }
}

/// A compositional query over a knowledge graph.
///
/// Queries are built recursively from atomic link predictions.
/// Use [`answer_query`] or [`answer_query_topk`] to evaluate.
#[derive(Debug, Clone)]
pub enum Query {
    /// Atomic: score all entities as tails for `(entity, relation, ?)`.
    Anchor {
        /// Head entity ID.
        entity: usize,
        /// Relation ID.
        relation: usize,
    },
    /// Chain: evaluate `inner` to get intermediate entities, then
    /// project each through `relation` via beam search.
    Project {
        /// Inner query producing intermediate entity scores.
        inner: Box<Query>,
        /// Relation to project through.
        relation: usize,
    },
    /// Conjunction: evaluate all branches, combine with t-norm.
    Intersection {
        /// Branches to intersect (2 or more).
        branches: Vec<Query>,
    },
    /// Disjunction: evaluate all branches, combine with t-conorm.
    Union {
        /// Branches to union (2 or more).
        branches: Vec<Query>,
    },
    /// Fuzzy complement: `1 - score(inner)`.
    Negation {
        /// Query to negate.
        inner: Box<Query>,
    },
}

impl Query {
    /// Create a one-hop query: `(entity, relation, ?)`.
    pub fn anchor(entity: usize, relation: usize) -> Self {
        Query::Anchor { entity, relation }
    }

    /// Chain a relation onto this query: `self -> relation -> ?`.
    pub fn then(self, relation: usize) -> Self {
        Query::Project {
            inner: Box::new(self),
            relation,
        }
    }

    /// Intersect multiple queries (conjunction).
    pub fn intersection(branches: Vec<Query>) -> Self {
        Query::Intersection { branches }
    }

    /// Union multiple queries (disjunction).
    pub fn union(branches: Vec<Query>) -> Self {
        Query::Union { branches }
    }

    /// Negate this query (fuzzy complement).
    pub fn negate(self) -> Self {
        Query::Negation {
            inner: Box::new(self),
        }
    }
}

/// Answer a query, returning scores for all entities.
///
/// Scores are in `[0, 1]` where higher = more likely answer.
/// The length of the returned vec equals `model.num_entities()`.
pub fn answer_query(model: &dyn Scorer, query: &Query, config: &QueryConfig) -> Vec<f32> {
    let n = model.num_entities();
    eval_query(model, query, config, n)
}

/// Answer a query and return the top-k `(entity_id, score)` pairs.
///
/// Results are sorted by score descending (best first).
pub fn answer_query_topk(
    model: &dyn Scorer,
    query: &Query,
    config: &QueryConfig,
    k: usize,
) -> Vec<(usize, f32)> {
    let scores = answer_query(model, query, config);
    top_k_descending(&scores, k)
}

fn eval_query(model: &dyn Scorer, query: &Query, config: &QueryConfig, n: usize) -> Vec<f32> {
    match query {
        Query::Anchor { entity, relation } => {
            let raw = model.score_all_tails(*entity, *relation);
            normalize_scores(&raw)
        }
        Query::Project { inner, relation } => {
            let inner_scores = eval_query(model, inner, config, n);
            beam_project(model, &inner_scores, *relation, config, n)
        }
        Query::Intersection { branches } => {
            let branch_scores: Vec<Vec<f32>> = branches
                .iter()
                .map(|b| eval_query(model, b, config, n))
                .collect();
            combine_conjunction(&branch_scores, config.t_norm_intersection, n)
        }
        Query::Union { branches } => {
            let branch_scores: Vec<Vec<f32>> = branches
                .iter()
                .map(|b| eval_query(model, b, config, n))
                .collect();
            combine_disjunction(&branch_scores, config.t_norm_intersection, n)
        }
        Query::Negation { inner } => {
            let scores = eval_query(model, inner, config, n);
            scores.iter().map(|&s| 1.0 - s).collect()
        }
    }
}

/// Beam search for chain projection.
///
/// Takes the top-k intermediate entities from `inner_scores`, scores all
/// tails through `relation` for each, combines inner and tail scores with
/// the t-norm, and returns the max over all beam candidates per target.
///
/// For `TNorm::Min`, sigmoid is deferred: raw scores are compared directly
/// (negated, since lower raw = better) and sigmoid is applied once per
/// entity at the end. This reduces sigmoid calls from `beam_k * N` to `N`.
fn beam_project(
    model: &dyn Scorer,
    inner_scores: &[f32],
    relation: usize,
    config: &QueryConfig,
    n: usize,
) -> Vec<f32> {
    let candidates = top_k_descending(inner_scores, config.beam_k);
    let norm = config.t_norm_projection;

    match norm {
        TNorm::Min => {
            // Deferred sigmoid: min(sigmoid(a), sigmoid(b)) = sigmoid(min(a, b))
            // since sigmoid is monotone. Work in raw-score space (negated: higher = better).
            let mut best_raw = vec![f32::NEG_INFINITY; n];
            for &(entity, inner_score) in &candidates {
                // Convert inner_score back to raw: sigmoid(-raw) = inner_score → raw = -logit(inner_score)
                // But we can just compare min(inner_raw, tail_raw) where raw = -logit(prob).
                // Since sigmoid is monotone, min(prob_a, prob_b) = sigmoid(min(raw_a, raw_b))
                // where raw = -score (negated scorer output). So inner_raw = logit(inner_score).
                let inner_raw = logit(inner_score);
                let raw_tail_scores = model.score_all_tails(entity, relation);
                for (t, &raw) in raw_tail_scores.iter().enumerate() {
                    let tail_raw = -raw; // negate: higher = better
                    let combined_raw = inner_raw.min(tail_raw);
                    if combined_raw > best_raw[t] {
                        best_raw[t] = combined_raw;
                    }
                }
            }
            // Apply sigmoid once per entity.
            best_raw.iter().map(|&r| sigmoid(r)).collect()
        }
        TNorm::Product => {
            let mut result = vec![0.0_f32; n];
            for &(entity, inner_score) in &candidates {
                let raw_tail_scores = model.score_all_tails(entity, relation);
                for (t, &raw) in raw_tail_scores.iter().enumerate() {
                    let tail_prob = sigmoid(-raw);
                    let combined = inner_score * tail_prob;
                    if combined > result[t] {
                        result[t] = combined;
                    }
                }
            }
            result
        }
    }
}

/// Inverse of sigmoid: `ln(p / (1 - p))`.
fn logit(p: f32) -> f32 {
    (p / (1.0 - p)).ln()
}

/// Convert raw Scorer outputs (lower=better) to `[0, 1]` (higher=better).
fn normalize_scores(raw: &[f32]) -> Vec<f32> {
    raw.iter().map(|&s| sigmoid(-s)).collect()
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn apply_t_norm(a: f32, b: f32, norm: TNorm) -> f32 {
    match norm {
        TNorm::Min => a.min(b),
        TNorm::Product => a * b,
    }
}

fn apply_t_conorm(a: f32, b: f32, norm: TNorm) -> f32 {
    match norm {
        TNorm::Min => a.max(b),
        TNorm::Product => a + b - a * b,
    }
}

fn combine_conjunction(branch_scores: &[Vec<f32>], norm: TNorm, n: usize) -> Vec<f32> {
    let mut result = vec![1.0_f32; n];
    for branch in branch_scores {
        for (i, &s) in branch.iter().enumerate() {
            result[i] = apply_t_norm(result[i], s, norm);
        }
    }
    result
}

fn combine_disjunction(branch_scores: &[Vec<f32>], norm: TNorm, n: usize) -> Vec<f32> {
    let mut result = vec![0.0_f32; n];
    for branch in branch_scores {
        for (i, &s) in branch.iter().enumerate() {
            result[i] = apply_t_conorm(result[i], s, norm);
        }
    }
    result
}

fn top_k_descending(scores: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    indexed
}

#[cfg(test)]
mod tests {
    use super::*;

    // A deterministic model where score(h, r, t) = |h + r - t| (TransE-like).
    // Entities: 0..N, relations shift by relation ID.
    // So the "correct" tail for (h, r) is h + r (mod N).
    struct ChainModel {
        n: usize,
    }

    impl Scorer for ChainModel {
        fn score(&self, h: usize, r: usize, t: usize) -> f32 {
            let expected = (h + r + 1) % self.n;
            t.abs_diff(expected) as f32
        }

        fn num_entities(&self) -> usize {
            self.n
        }
    }

    #[test]
    fn anchor_query_matches_score_all_tails() {
        let model = ChainModel { n: 10 };
        let config = QueryConfig::default();
        let query = Query::anchor(2, 3);
        let scores = answer_query(&model, &query, &config);
        assert_eq!(scores.len(), 10);

        // Entity (2+3+1)%10 = 6 should have the best score (distance 0 → sigmoid(0)=0.5... wait)
        // Actually distance 0 → sigmoid(-0) = 0.5, distance 1 → sigmoid(-1) ≈ 0.27
        // The best entity should have the highest normalized score.
        let best = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assert_eq!(best.0, 6, "Entity 6 should score best for (2, 3, ?)");
    }

    #[test]
    fn chain_2p_finds_two_hop_answer() {
        let model = ChainModel { n: 20 };
        let config = QueryConfig {
            t_norm_projection: TNorm::Product,
            t_norm_intersection: TNorm::Min,
            beam_k: 20, // full beam for small model
        };
        // 2p: entity 0 -rel 0-> V -rel 2-> ?
        // Step 1: best V for (0, 0, ?) is entity 1 (0+0+1=1)
        // Step 2: best answer for (1, 2, ?) is entity 4 (1+2+1=4)
        let query = Query::anchor(0, 0).then(2);
        let scores = answer_query(&model, &query, &config);

        let top = top_k_descending(&scores, 3);
        assert_eq!(top[0].0, 4, "Two-hop answer should be entity 4");
    }

    #[test]
    fn intersection_narrows_results() {
        let model = ChainModel { n: 20 };
        let config = QueryConfig::default();

        // Branch 1: (0, 4, ?) → best is entity 5
        // Branch 2: (2, 2, ?) → best is entity 5
        // Both agree on entity 5, so intersection should rank it first.
        let query = Query::intersection(vec![Query::anchor(0, 4), Query::anchor(2, 2)]);
        let scores = answer_query(&model, &query, &config);
        let top = top_k_descending(&scores, 1);
        assert_eq!(top[0].0, 5, "Intersection should agree on entity 5");
    }

    #[test]
    fn union_at_least_as_good_as_branches() {
        let model = ChainModel { n: 10 };
        let config = QueryConfig {
            t_norm_intersection: TNorm::Product,
            ..QueryConfig::default()
        };

        let branch1 = Query::anchor(0, 0);
        let branch2 = Query::anchor(3, 3);

        let scores1 = answer_query(&model, &branch1, &config);
        let scores2 = answer_query(&model, &branch2, &config);
        let union_scores = answer_query(&model, &Query::union(vec![branch1, branch2]), &config);

        for i in 0..10 {
            assert!(
                union_scores[i] >= scores1[i] - 1e-6,
                "Union score should be >= branch 1 for entity {i}"
            );
            assert!(
                union_scores[i] >= scores2[i] - 1e-6,
                "Union score should be >= branch 2 for entity {i}"
            );
        }
    }

    #[test]
    fn negation_inverts_scores() {
        let model = ChainModel { n: 10 };
        let config = QueryConfig::default();

        let query = Query::anchor(0, 0);
        let scores = answer_query(&model, &query, &config);
        let neg_scores = answer_query(&model, &query.clone().negate(), &config);

        for i in 0..10 {
            assert!(
                (scores[i] + neg_scores[i] - 1.0).abs() < 1e-6,
                "score + negated should equal 1.0 for entity {i}: {} + {} = {}",
                scores[i],
                neg_scores[i],
                scores[i] + neg_scores[i],
            );
        }
    }

    #[test]
    fn topk_returns_sorted_descending() {
        let model = ChainModel { n: 20 };
        let config = QueryConfig::default();
        let query = Query::anchor(0, 0);
        let top = answer_query_topk(&model, &query, &config, 5);
        assert_eq!(top.len(), 5);
        for w in top.windows(2) {
            assert!(w[0].1 >= w[1].1, "Top-k should be sorted descending");
        }
    }

    #[test]
    fn pi_query_intersect_then_project() {
        let model = ChainModel { n: 20 };
        let config = QueryConfig {
            t_norm_projection: TNorm::Min,
            t_norm_intersection: TNorm::Min,
            beam_k: 20,
        };
        // pi: (entity 0 -rel 4-> V) AND (entity 2 -rel 2-> V), then V -rel 0-> ?
        // Both branches agree on V=5, so intersection peaks at V=5.
        // Then (5, 0, ?) → best is entity 6.
        let query = Query::intersection(vec![Query::anchor(0, 4), Query::anchor(2, 2)]).then(0);
        let top = answer_query_topk(&model, &query, &config, 1);
        assert_eq!(top[0].0, 6, "pi query should find entity 6");
    }

    #[test]
    fn sigmoid_is_numerically_stable() {
        // Large positive and negative values should not NaN or Inf.
        assert!((sigmoid(100.0) - 1.0).abs() < 1e-6);
        assert!(sigmoid(-100.0).abs() < 1e-6);
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(f32::MAX).is_finite());
        assert!(sigmoid(f32::MIN).is_finite());
    }

    #[test]
    fn t_norm_properties() {
        // Identity: t_norm(x, 1) = x
        for &x in &[0.0, 0.3, 0.7, 1.0] {
            assert!((apply_t_norm(x, 1.0, TNorm::Min) - x).abs() < 1e-6);
            assert!((apply_t_norm(x, 1.0, TNorm::Product) - x).abs() < 1e-6);
        }
        // Commutativity: t_norm(a, b) = t_norm(b, a)
        let (a, b) = (0.3, 0.7);
        assert_eq!(
            apply_t_norm(a, b, TNorm::Min),
            apply_t_norm(b, a, TNorm::Min)
        );
        assert!(
            (apply_t_norm(a, b, TNorm::Product) - apply_t_norm(b, a, TNorm::Product)).abs() < 1e-6
        );
    }

    #[test]
    fn t_conorm_de_morgan_duality() {
        // t_conorm(a, b) = 1 - t_norm(1-a, 1-b)
        let (a, b) = (0.3, 0.7);
        for norm in [TNorm::Min, TNorm::Product] {
            let conorm = apply_t_conorm(a, b, norm);
            let dual = 1.0 - apply_t_norm(1.0 - a, 1.0 - b, norm);
            assert!(
                (conorm - dual).abs() < 1e-6,
                "De Morgan failed for {norm:?}: {conorm} vs {dual}"
            );
        }
    }
}
