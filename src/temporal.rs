//! Temporal knowledge graphs: quads `(head, relation, tail, timestamp)`.
//!
//! Facts in an event KG (ICEWS-style) carry a discrete timestamp. This
//! module loads quad datasets, defines the [`TemporalScorer`] trait (the
//! temporal counterpart of [`Scorer`](crate::Scorer), same lower-is-better
//! energy convention), provides the [`TComplEx`] CPU scorer (Lacroix,
//! Obozinski, Usunier, ICLR 2020: ComplEx with a complex timestamp
//! embedding multiplied into the relation), and evaluates temporal link
//! prediction in the time-aware filtered setting.
//!
//! Training lives in [`burn_temporal`](crate::burn_temporal) (feature
//! `burn-ndarray` or `burn-wgpu`); the CLI subcommand is
//! `tranz train-temporal`.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use rayon::prelude::*;

use crate::dataset::Vocab;
use crate::eval::Metrics;

/// One interned quad: `(head, relation, tail, time)` as dense ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuadIds {
    /// Head entity id.
    pub head: usize,
    /// Relation id.
    pub relation: usize,
    /// Tail entity id.
    pub tail: usize,
    /// Timestamp id (chronological order: smaller id = earlier).
    pub time: usize,
}

impl QuadIds {
    /// Construct a quad.
    pub fn new(head: usize, relation: usize, tail: usize, time: usize) -> Self {
        Self {
            head,
            relation,
            tail,
            time,
        }
    }
}

/// An interned temporal dataset: vocabularies plus train/valid/test quads.
///
/// Timestamp ids are assigned in sorted order of the timestamp strings, so
/// for ISO dates (`YYYY-MM-DD`) id order is chronological order; the time
/// axis is meaningful for smoothing regularizers and windowed queries.
#[derive(Debug, Default)]
pub struct TemporalDataset {
    /// Entity vocabulary (first-appearance order).
    pub entities: Vocab,
    /// Relation vocabulary (first-appearance order).
    pub relations: Vocab,
    /// Timestamp strings, sorted; index = timestamp id.
    pub times: Vec<String>,
    /// Training quads.
    pub train: Vec<QuadIds>,
    /// Validation quads.
    pub valid: Vec<QuadIds>,
    /// Test quads.
    pub test: Vec<QuadIds>,
}

impl TemporalDataset {
    /// Number of entities.
    pub fn num_entities(&self) -> usize {
        self.entities.len()
    }
    /// Number of relations.
    pub fn num_relations(&self) -> usize {
        self.relations.len()
    }
    /// Number of distinct timestamps.
    pub fn num_timestamps(&self) -> usize {
        self.times.len()
    }

    /// All quads across the three splits.
    pub fn all_quads(&self) -> Vec<QuadIds> {
        let mut all = self.train.clone();
        all.extend_from_slice(&self.valid);
        all.extend_from_slice(&self.test);
        all
    }

    /// Add reciprocal relations: for each relation `r` with id `i`, a new
    /// `r_inv` with id `num_relations + i`, and for each quad
    /// `(h, r, t, τ)` the quad `(t, r_inv, h, τ)` in the same split.
    pub fn add_reciprocals(&mut self) {
        let n_rel = self.relations.len();
        for i in 0..n_rel {
            let name = format!("{}_inv", self.relations.get(i).unwrap());
            self.relations.intern(name);
        }
        fn augment(quads: &mut Vec<QuadIds>, n_rel: usize) {
            let originals: Vec<_> = quads.clone();
            quads.reserve(originals.len());
            for q in &originals {
                quads.push(QuadIds::new(q.tail, q.relation + n_rel, q.head, q.time));
            }
        }
        augment(&mut self.train, n_rel);
        augment(&mut self.valid, n_rel);
        augment(&mut self.test, n_rel);
    }
}

/// Load a temporal dataset from a directory with `train.txt`, `valid.txt`,
/// `test.txt`, each line `head \t relation \t tail \t timestamp`.
///
/// Timestamps are interned in sorted (chronological, for ISO dates) order
/// over the union of all splits. Malformed lines are skipped with a count
/// on stderr.
pub fn load_temporal_dataset(dir: &Path) -> Result<TemporalDataset, crate::Error> {
    let read = |name: &str| -> Result<Vec<(String, String, String, String)>, crate::Error> {
        let content = std::fs::read_to_string(dir.join(name))?;
        let mut rows = Vec::new();
        let mut dropped = 0usize;
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = trimmed.split('\t').map(str::trim).collect();
            if parts.len() >= 4 {
                rows.push((
                    parts[0].to_string(),
                    parts[1].to_string(),
                    parts[2].to_string(),
                    parts[3].to_string(),
                ));
            } else {
                dropped += 1;
            }
        }
        if dropped > 0 {
            eprintln!("warning: {name}: skipped {dropped} lines (expected 4 fields)");
        }
        Ok(rows)
    };
    let raw_train = read("train.txt")?;
    let raw_valid = read("valid.txt")?;
    let raw_test = read("test.txt")?;

    // Timestamp axis first: sorted unique strings over all splits.
    let mut stamps: Vec<&str> = raw_train
        .iter()
        .chain(&raw_valid)
        .chain(&raw_test)
        .map(|r| r.3.as_str())
        .collect();
    stamps.sort_unstable();
    stamps.dedup();
    let time_id: HashMap<&str, usize> = stamps.iter().enumerate().map(|(i, s)| (*s, i)).collect();

    let mut ds = TemporalDataset {
        times: stamps.iter().map(|s| (*s).to_string()).collect(),
        ..TemporalDataset::default()
    };
    let intern =
        |rows: &[(String, String, String, String)], entities: &mut Vocab, relations: &mut Vocab| {
            rows.iter()
                .map(|(h, r, t, s)| {
                    QuadIds::new(
                        entities.intern(h.clone()),
                        relations.intern(r.clone()),
                        entities.intern(t.clone()),
                        time_id[s.as_str()],
                    )
                })
                .collect::<Vec<_>>()
        };
    ds.train = intern(&raw_train, &mut ds.entities, &mut ds.relations);
    ds.valid = intern(&raw_valid, &mut ds.entities, &mut ds.relations);
    ds.test = intern(&raw_test, &mut ds.entities, &mut ds.relations);
    Ok(ds)
}

/// Trait for scoring temporal quads. Same convention as
/// [`Scorer`](crate::Scorer): **lower values indicate more likely quads**.
pub trait TemporalScorer: Sync {
    /// Score `(head, relation, tail, time)`. Lower = more likely.
    fn score(&self, head: usize, relation: usize, tail: usize, time: usize) -> f32;

    /// Number of entities in the model.
    fn num_entities(&self) -> usize;

    /// Number of relations in the model.
    fn num_relations(&self) -> usize;

    /// Number of timestamps in the model.
    fn num_timestamps(&self) -> usize;

    /// Score all entities as tails for `(head, relation, ?, time)`.
    fn score_all_tails(&self, head: usize, relation: usize, time: usize) -> Vec<f32> {
        (0..self.num_entities())
            .map(|t| self.score(head, relation, t, time))
            .collect()
    }

    /// Score all entities as heads for `(?, relation, tail, time)`.
    fn score_all_heads(&self, relation: usize, tail: usize, time: usize) -> Vec<f32> {
        (0..self.num_entities())
            .map(|h| self.score(h, relation, tail, time))
            .collect()
    }
}

/// TComplEx (Lacroix et al., ICLR 2020): `-Re(<h, r ∘ w_τ, conj(t)>)`.
///
/// Entity, relation, and timestamp embeddings are complex vectors stored as
/// `[re..; im..]` rows of width `2 * dim`. The timestamp embedding rotates
/// and scales the relation componentwise, so one relation can mean
/// different things at different times.
#[derive(Debug, Clone)]
pub struct TComplEx {
    entity: Vec<Vec<f32>>,
    relation: Vec<Vec<f32>>,
    time: Vec<Vec<f32>>,
    dim: usize,
}

impl TComplEx {
    /// Build from embedding rows (each of width `2 * dim`).
    ///
    /// # Panics
    /// Panics if any row's width is not `2 * dim`.
    pub fn from_vecs(
        entity: Vec<Vec<f32>>,
        relation: Vec<Vec<f32>>,
        time: Vec<Vec<f32>>,
        dim: usize,
    ) -> Self {
        for rows in [&entity, &relation, &time] {
            assert!(
                rows.iter().all(|r| r.len() == 2 * dim),
                "embedding rows must have width 2*dim = {}",
                2 * dim
            );
        }
        Self {
            entity,
            relation,
            time,
            dim,
        }
    }

    /// The complex dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl TemporalScorer for TComplEx {
    fn score(&self, head: usize, relation: usize, tail: usize, time: usize) -> f32 {
        let d = self.dim;
        let (h, r, w, t) = (
            &self.entity[head],
            &self.relation[relation],
            &self.time[time],
            &self.entity[tail],
        );
        let mut s = 0.0_f32;
        for i in 0..d {
            // q = r ∘ w (complex componentwise product).
            let q_re = r[i] * w[i] - r[i + d] * w[i + d];
            let q_im = r[i] * w[i + d] + r[i + d] * w[i];
            // Re(h_i q_i conj(t_i)).
            s +=
                (h[i] * q_re - h[i + d] * q_im) * t[i] + (h[i] * q_im + h[i + d] * q_re) * t[i + d];
        }
        -s
    }

    fn num_entities(&self) -> usize {
        self.entity.len()
    }
    fn num_relations(&self) -> usize {
        self.relation.len()
    }
    fn num_timestamps(&self) -> usize {
        self.time.len()
    }
}

/// Time-aware filter index: known tails of `(h, r, τ)` and known heads of
/// `(r, t, τ)`, over all splits.
#[derive(Debug, Default)]
pub struct TemporalFilterIndex {
    tails: HashMap<(usize, usize, usize), HashSet<usize>>,
    heads: HashMap<(usize, usize, usize), HashSet<usize>>,
}

impl TemporalFilterIndex {
    /// Build from every quad the ranking must treat as known-true.
    pub fn from_quads(quads: &[QuadIds]) -> Self {
        let mut idx = Self::default();
        for q in quads {
            idx.tails
                .entry((q.head, q.relation, q.time))
                .or_default()
                .insert(q.tail);
            idx.heads
                .entry((q.relation, q.tail, q.time))
                .or_default()
                .insert(q.head);
        }
        idx
    }
}

static EMPTY: std::sync::LazyLock<HashSet<usize>> = std::sync::LazyLock::new(HashSet::new);

/// Evaluate temporal link prediction in the time-aware filtered setting.
///
/// For each test quad, ranks the true tail among all entities for
/// `(h, r, ?, τ)` and the true head for `(?, r, t, τ)`, filtering other
/// known-true completions **at the same timestamp**. Returns zeroed
/// [`Metrics`] on an empty test set. Parallelized via rayon.
pub fn evaluate_temporal_link_prediction(
    model: &dyn TemporalScorer,
    test: &[QuadIds],
    filter: &TemporalFilterIndex,
) -> Metrics {
    if test.is_empty() {
        return Metrics::default();
    }
    let ranks: Vec<(u32, u32)> = test
        .par_iter()
        .map(|q| {
            let tail_scores = model.score_all_tails(q.head, q.relation, q.time);
            let known_tails = filter
                .tails
                .get(&(q.head, q.relation, q.time))
                .unwrap_or(&EMPTY);
            let target = tail_scores[q.tail];
            let mut tail_rank = 1u32;
            for (t, &s) in tail_scores.iter().enumerate() {
                if t != q.tail && !known_tails.contains(&t) && s < target {
                    tail_rank += 1;
                }
            }

            let head_scores = model.score_all_heads(q.relation, q.tail, q.time);
            let known_heads = filter
                .heads
                .get(&(q.relation, q.tail, q.time))
                .unwrap_or(&EMPTY);
            let target = head_scores[q.head];
            let mut head_rank = 1u32;
            for (h, &s) in head_scores.iter().enumerate() {
                if h != q.head && !known_heads.contains(&h) && s < target {
                    head_rank += 1;
                }
            }
            (tail_rank, head_rank)
        })
        .collect();

    let n = (ranks.len() * 2) as f32;
    let all: Vec<u32> = ranks.iter().flat_map(|&(t, h)| [t, h]).collect();
    let mrr = all.iter().map(|&r| 1.0 / r as f32).sum::<f32>() / n;
    let tail_mrr = ranks.iter().map(|&(t, _)| 1.0 / t as f32).sum::<f32>() / ranks.len() as f32;
    let head_mrr = ranks.iter().map(|&(_, h)| 1.0 / h as f32).sum::<f32>() / ranks.len() as f32;
    let hits = |k: u32| all.iter().filter(|&&r| r <= k).count() as f32 / n;
    Metrics {
        mrr,
        head_mrr,
        tail_mrr,
        mean_rank: all.iter().map(|&r| r as f32).sum::<f32>() / n,
        hits_at_1: hits(1),
        hits_at_3: hits(3),
        hits_at_10: hits(10),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn tcomplex_score_matches_hand_computation() {
        // dim=1: h = 1+2i, r = 0.5, w = 1+1i, t = 2-1i.
        // q = r∘w = 0.5+0.5i; h∘q = -0.5+1.5i; times conj(t) = (−0.5+1.5i)(2+1i)
        // has real part −1 − 1.5 = −2.5. Energy = 2.5.
        let m = TComplEx::from_vecs(
            vec![vec![1.0, 2.0], vec![2.0, -1.0]],
            vec![vec![0.5, 0.0]],
            vec![vec![1.0, 1.0]],
            1,
        );
        let e = m.score(0, 0, 1, 0);
        assert!((e - 2.5).abs() < 1e-6, "energy {e}");
    }

    #[test]
    fn loader_interns_sorted_timestamps() {
        let dir = tempfile::tempdir().unwrap();
        let write = |name: &str, body: &str| {
            let mut f = std::fs::File::create(dir.path().join(name)).unwrap();
            f.write_all(body.as_bytes()).unwrap();
        };
        write("train.txt", "A\tr\tB\t2014-05-13\nB\tr\tC\t2014-01-02\n");
        write("valid.txt", "A\tr\tC\t2014-03-01\n");
        write("test.txt", "C\tr\tA\t2014-01-02\nbad line\n");

        let ds = load_temporal_dataset(dir.path()).unwrap();
        assert_eq!(ds.num_entities(), 3);
        assert_eq!(ds.num_relations(), 1);
        // Sorted: 2014-01-02 < 2014-03-01 < 2014-05-13.
        assert_eq!(ds.times, vec!["2014-01-02", "2014-03-01", "2014-05-13"]);
        assert_eq!(ds.train[0].time, 2);
        assert_eq!(ds.train[1].time, 0);
        assert_eq!(ds.valid[0].time, 1);
        assert_eq!(ds.test.len(), 1, "malformed line dropped");
    }

    #[test]
    fn reciprocals_swap_and_offset() {
        let mut ds = TemporalDataset::default();
        ds.entities.intern("A".to_string());
        ds.entities.intern("B".to_string());
        ds.relations.intern("r".to_string());
        ds.times = vec!["2014-01-01".to_string()];
        ds.train = vec![QuadIds::new(0, 0, 1, 0)];
        ds.add_reciprocals();
        assert_eq!(ds.num_relations(), 2);
        assert_eq!(ds.relations.get(1), Some("r_inv"));
        assert_eq!(ds.train[1], QuadIds::new(1, 1, 0, 0));
    }

    /// A scorer whose planted quad is best exactly at its own timestamp:
    /// filtered eval must rank it 1 there.
    struct Planted;
    impl TemporalScorer for Planted {
        fn score(&self, h: usize, _r: usize, t: usize, time: usize) -> f32 {
            if (h, t, time) == (0, 1, 0) {
                -1.0
            } else {
                0.0
            }
        }
        fn num_entities(&self) -> usize {
            3
        }
        fn num_relations(&self) -> usize {
            1
        }
        fn num_timestamps(&self) -> usize {
            2
        }
    }

    #[test]
    fn filtered_eval_ranks_planted_quad_first() {
        let test = vec![QuadIds::new(0, 0, 1, 0)];
        let filter = TemporalFilterIndex::from_quads(&test);
        let m = evaluate_temporal_link_prediction(&Planted, &test, &filter);
        assert!((m.tail_mrr - 1.0).abs() < 1e-6, "tail mrr {}", m.tail_mrr);
        assert!((m.head_mrr - 1.0).abs() < 1e-6, "head mrr {}", m.head_mrr);
        assert_eq!(m.hits_at_1, 1.0);
    }

    /// Same-timestamp filtering: a second known tail at τ=0 is filtered, so
    /// the target still ranks 1 even when the known tail scores better.
    struct TwoTrue;
    impl TemporalScorer for TwoTrue {
        fn score(&self, h: usize, _r: usize, t: usize, time: usize) -> f32 {
            match (h, t, time) {
                (0, 2, 0) => -2.0, // known-true competitor, better score
                (0, 1, 0) => -1.0, // target
                _ => 0.0,
            }
        }
        fn num_entities(&self) -> usize {
            3
        }
        fn num_relations(&self) -> usize {
            1
        }
        fn num_timestamps(&self) -> usize {
            2
        }
    }

    #[test]
    fn filter_is_time_aware() {
        let test = vec![QuadIds::new(0, 0, 1, 0)];
        let all = vec![QuadIds::new(0, 0, 1, 0), QuadIds::new(0, 0, 2, 0)];
        let m = evaluate_temporal_link_prediction(
            &TwoTrue,
            &test,
            &TemporalFilterIndex::from_quads(&all),
        );
        assert!((m.tail_mrr - 1.0).abs() < 1e-6, "filtered: {}", m.tail_mrr);

        // Without the competitor in the filter it outranks the target.
        let m = evaluate_temporal_link_prediction(
            &TwoTrue,
            &test,
            &TemporalFilterIndex::from_quads(&test),
        );
        assert!(
            (m.tail_mrr - 0.5).abs() < 1e-6,
            "unfiltered: {}",
            m.tail_mrr
        );
    }
}
