//! # tranz
//!
//! Point-embedding knowledge graph completion models.
//!
//! Entities are points in vector space. Relations are transformations
//! (translation, rotation, diagonal scaling). Lower distance between
//! `transform(head, relation)` and `tail` indicates a more likely triple.
//!
//! ## Models
//!
//! - [`TransE`]: `head + relation ~ tail` (Bordes et al., 2013)
//! - [`RotatE`]: `head * relation ~ tail` in complex space (Sun et al., 2019)
//! - [`ComplEx`]: Hermitian dot product in complex space (Trouillon et al., 2016)
//! - [`DistMult`]: diagonal bilinear in real space (Yang et al., 2015)
//!
//! ## Feature flags
//!
//! - **`rand`** (default): enables random initialization via `Model::new()`.
//! - **`candle`**: enables GPU training via the [`train`] module.
//! - **`cuda`**: implies `candle`, enables CUDA acceleration.

#![warn(missing_docs)]

pub mod dataset;
pub mod eval;
pub mod io;
#[cfg(feature = "candle")]
pub mod train;

/// Errors from tranz operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Dimension mismatch.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },
    /// IO error.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Trait for scoring knowledge graph triples.
///
/// Scores are distances or negative similarities: **lower values indicate
/// more likely triples**.
pub trait Scorer {
    /// Score a triple `(head, relation, tail)`. Lower = more likely.
    fn score(&self, head: usize, relation: usize, tail: usize) -> f32;

    /// Number of entities in the model.
    fn num_entities(&self) -> usize;

    /// Score all entities as tail replacements for `(head, relation, ?)`.
    ///
    /// Returns a vec of length `num_entities()` where index `t` holds
    /// `score(head, relation, t)`.
    fn score_all_tails(&self, head: usize, relation: usize) -> Vec<f32> {
        (0..self.num_entities())
            .map(|t| self.score(head, relation, t))
            .collect()
    }

    /// Score all entities as head replacements for `(?, relation, tail)`.
    fn score_all_heads(&self, relation: usize, tail: usize) -> Vec<f32> {
        (0..self.num_entities())
            .map(|h| self.score(h, relation, tail))
            .collect()
    }

    /// Return the top-k entities by score for `(head, relation, ?)`.
    ///
    /// Returns `(entity_id, score)` pairs sorted by score ascending
    /// (best first, since lower = more likely).
    fn top_k_tails(&self, head: usize, relation: usize, k: usize) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = self
            .score_all_tails(head, relation)
            .into_iter()
            .enumerate()
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Return the top-k entities by score for `(?, relation, tail)`.
    fn top_k_heads(&self, relation: usize, tail: usize, k: usize) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = self
            .score_all_heads(relation, tail)
            .into_iter()
            .enumerate()
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Score all relations for `(head, ?, tail)`.
    ///
    /// Returns a vec where index `r` holds `score(head, r, tail)`.
    /// Requires knowing the number of relations (passed as parameter
    /// since the Scorer trait doesn't expose it).
    fn score_all_relations(&self, head: usize, tail: usize, num_relations: usize) -> Vec<f32> {
        (0..num_relations)
            .map(|r| self.score(head, r, tail))
            .collect()
    }

    /// Return the top-k relations by score for `(head, ?, tail)`.
    fn top_k_relations(
        &self,
        head: usize,
        tail: usize,
        num_relations: usize,
        k: usize,
    ) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = self
            .score_all_relations(head, tail, num_relations)
            .into_iter()
            .enumerate()
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}

// ---------------------------------------------------------------------------
// TransE
// ---------------------------------------------------------------------------

/// TransE: translational distance model.
///
/// Scores triples by `||head + relation - tail||_2`. Lower = more likely.
///
/// Initialization uses uniform `[-6/sqrt(dim), 6/sqrt(dim)]` (Xavier-like).
///
/// Reference: Bordes et al. (2013), "Translating Embeddings for Modeling
/// Multi-relational Data."
pub struct TransE {
    entities: Vec<Vec<f32>>,
    relations: Vec<Vec<f32>>,
    dim: usize,
}

impl TransE {
    /// Create a new TransE model with random initialization.
    ///
    /// Entity and relation embeddings are drawn from
    /// `Uniform(-6/sqrt(dim), 6/sqrt(dim))`.
    #[cfg(feature = "rand")]
    pub fn new(num_entities: usize, num_relations: usize, dim: usize) -> Self {
        let mut rng = rand::rng();
        let scale = 6.0_f32 / (dim as f32).sqrt();
        Self::from_vecs(
            init_vecs(&mut rng, num_entities, dim, scale),
            init_vecs(&mut rng, num_relations, dim, scale),
            dim,
        )
    }

    /// Create from pre-built embedding vectors.
    ///
    /// # Panics
    ///
    /// Panics if any inner vector length differs from `dim`.
    pub fn from_vecs(entities: Vec<Vec<f32>>, relations: Vec<Vec<f32>>, dim: usize) -> Self {
        assert_dims(&entities, dim, "entity");
        assert_dims(&relations, dim, "relation");
        Self {
            entities,
            relations,
            dim,
        }
    }

    /// Entity embeddings: `[num_entities][dim]`.
    pub fn entities(&self) -> &[Vec<f32>] {
        &self.entities
    }

    /// Relation embeddings: `[num_relations][dim]`.
    pub fn relations(&self) -> &[Vec<f32>] {
        &self.relations
    }

    /// Embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Score a triple (h, r, t) via L2 distance of `h + r - t`.
    pub fn score_triple(&self, head: usize, relation: usize, tail: usize) -> f32 {
        let h = &self.entities[head];
        let r = &self.relations[relation];
        let t = &self.entities[tail];

        let mut dist_sq = 0.0_f64;
        for i in 0..self.dim {
            let d = h[i] as f64 + r[i] as f64 - t[i] as f64;
            dist_sq += d * d;
        }
        dist_sq.sqrt() as f32
    }
}

impl Scorer for TransE {
    fn score(&self, head: usize, relation: usize, tail: usize) -> f32 {
        self.score_triple(head, relation, tail)
    }

    fn num_entities(&self) -> usize {
        self.entities.len()
    }

    fn score_all_tails(&self, head: usize, relation: usize) -> Vec<f32> {
        let h = &self.entities[head];
        let r = &self.relations[relation];
        let dim = self.dim;
        // Precompute h + r once.
        let mut hr = vec![0.0_f64; dim];
        for i in 0..dim {
            hr[i] = h[i] as f64 + r[i] as f64;
        }
        self.entities
            .iter()
            .map(|t| {
                let mut dist_sq = 0.0_f64;
                for i in 0..dim {
                    let d = hr[i] - t[i] as f64;
                    dist_sq += d * d;
                }
                dist_sq.sqrt() as f32
            })
            .collect()
    }

    fn score_all_heads(&self, relation: usize, tail: usize) -> Vec<f32> {
        let r = &self.relations[relation];
        let t = &self.entities[tail];
        let dim = self.dim;
        // Precompute r - t once (since h + r - t = h - (t - r)).
        let mut neg_rt = vec![0.0_f64; dim];
        for i in 0..dim {
            neg_rt[i] = r[i] as f64 - t[i] as f64;
        }
        self.entities
            .iter()
            .map(|h| {
                let mut dist_sq = 0.0_f64;
                for i in 0..dim {
                    let d = h[i] as f64 + neg_rt[i];
                    dist_sq += d * d;
                }
                dist_sq.sqrt() as f32
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// RotatE
// ---------------------------------------------------------------------------

/// RotatE: rotation in complex space.
///
/// Entities are complex vectors. Relations are element-wise rotations
/// (unit-modulus complex numbers parameterized by angles).
///
/// Score: `||head * relation - tail||_2` where `*` is element-wise complex
/// multiplication and `|relation_i| = 1`.
///
/// Initialization: entities from `Uniform(-gamma/sqrt(dim), gamma/sqrt(dim))`,
/// relation angles from `Uniform(-pi, pi)`.
///
/// Reference: Sun et al. (2019), "RotatE: Knowledge Graph Embedding by
/// Relational Rotation in Complex Space."
pub struct RotatE {
    /// Entity embeddings: `[re_0..re_{dim-1}, im_0..im_{dim-1}]`, length = dim * 2.
    entities: Vec<Vec<f32>>,
    /// Relation angles, length = dim. The complex rotation is `(cos(angle), sin(angle))`.
    relation_angles: Vec<Vec<f32>>,
    /// Complex dimension (half the real storage per entity).
    dim: usize,
    /// Margin parameter used for initialization scaling.
    gamma: f32,
}

impl RotatE {
    /// Create a new RotatE model with random initialization.
    ///
    /// `dim` is the complex dimension (each entity stores `2 * dim` floats).
    /// `gamma` is the margin, used to scale entity initialization to
    /// `Uniform(-gamma/sqrt(dim), gamma/sqrt(dim))`.
    #[cfg(feature = "rand")]
    pub fn new(num_entities: usize, num_relations: usize, dim: usize, gamma: f32) -> Self {
        use rand::Rng;
        let mut rng = rand::rng();
        let entity_scale = gamma / (dim as f32).sqrt();
        let entities = init_vecs(&mut rng, num_entities, dim * 2, entity_scale);
        let relation_angles: Vec<Vec<f32>> = (0..num_relations)
            .map(|_| {
                (0..dim)
                    .map(|_| rng.random_range(-std::f32::consts::PI..std::f32::consts::PI))
                    .collect()
            })
            .collect();
        Self {
            entities,
            relation_angles,
            dim,
            gamma,
        }
    }

    /// Create from pre-built embedding vectors.
    ///
    /// `entities` must have inner length `dim * 2` (interleaved re/im).
    /// `relation_angles` must have inner length `dim`.
    ///
    /// # Panics
    ///
    /// Panics if any dimension is wrong.
    pub fn from_vecs(
        entities: Vec<Vec<f32>>,
        relation_angles: Vec<Vec<f32>>,
        dim: usize,
        gamma: f32,
    ) -> Self {
        assert_dims(&entities, dim * 2, "entity (re+im)");
        assert_dims(&relation_angles, dim, "relation angle");
        Self {
            entities,
            relation_angles,
            dim,
            gamma,
        }
    }

    /// Entity embeddings: `[re..., im...]` contiguous layout.
    pub fn entities(&self) -> &[Vec<f32>] {
        &self.entities
    }

    /// Relation angles.
    pub fn relation_angles(&self) -> &[Vec<f32>] {
        &self.relation_angles
    }

    /// Complex dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Margin parameter.
    pub fn gamma(&self) -> f32 {
        self.gamma
    }

    /// Score a triple by `||h * r - t||_2` in complex space.
    ///
    /// Entity layout: first `dim` floats are real parts, next `dim` are imaginary.
    pub fn score_triple(&self, head: usize, relation: usize, tail: usize) -> f32 {
        let h = &self.entities[head];
        let r = &self.relation_angles[relation];
        let t = &self.entities[tail];
        let dim = self.dim;

        let mut dist_sq = 0.0_f64;
        for i in 0..dim {
            let h_re = h[i] as f64;
            let h_im = h[dim + i] as f64;
            let (r_sin, r_cos) = (r[i] as f64).sin_cos();
            let t_re = t[i] as f64;
            let t_im = t[dim + i] as f64;

            let hr_re = h_re * r_cos - h_im * r_sin;
            let hr_im = h_re * r_sin + h_im * r_cos;

            let d_re = hr_re - t_re;
            let d_im = hr_im - t_im;
            dist_sq += d_re * d_re + d_im * d_im;
        }
        dist_sq.sqrt() as f32
    }
}

impl Scorer for RotatE {
    fn score(&self, head: usize, relation: usize, tail: usize) -> f32 {
        self.score_triple(head, relation, tail)
    }

    fn num_entities(&self) -> usize {
        self.entities.len()
    }

    fn score_all_tails(&self, head: usize, relation: usize) -> Vec<f32> {
        let h = &self.entities[head];
        let r = &self.relation_angles[relation];
        let dim = self.dim;
        // Precompute h * r (complex rotation) once.
        let mut hr_re = vec![0.0_f64; dim];
        let mut hr_im = vec![0.0_f64; dim];
        for i in 0..dim {
            let h_re = h[i] as f64;
            let h_im = h[dim + i] as f64;
            let (r_sin, r_cos) = (r[i] as f64).sin_cos();
            hr_re[i] = h_re * r_cos - h_im * r_sin;
            hr_im[i] = h_re * r_sin + h_im * r_cos;
        }
        self.entities
            .iter()
            .map(|t| {
                let mut dist_sq = 0.0_f64;
                for i in 0..dim {
                    let d_re = hr_re[i] - t[i] as f64;
                    let d_im = hr_im[i] - t[dim + i] as f64;
                    dist_sq += d_re * d_re + d_im * d_im;
                }
                dist_sq.sqrt() as f32
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// ComplEx
// ---------------------------------------------------------------------------

/// ComplEx: complex bilinear model.
///
/// Entities and relations are complex vectors. Score is the real part of the
/// Hermitian dot product: `Re(sum_i h_i * r_i * conj(t_i)))`.
///
/// **Higher scores = more likely**, so the `Scorer` implementation returns
/// the negated score (lower = more likely) for compatibility with
/// distance-based evaluation.
///
/// Initialization: Xavier uniform `Uniform(-sqrt(6/dim), sqrt(6/dim))`.
///
/// Reference: Trouillon et al. (2016), "Complex Embeddings for Simple Link
/// Prediction."
pub struct ComplEx {
    /// Entity embeddings: `[re_0..re_{dim-1}, im_0..im_{dim-1}]`, length = dim * 2.
    entities: Vec<Vec<f32>>,
    /// Relation embeddings: `[re..., im...]`, length = dim * 2.
    relations: Vec<Vec<f32>>,
    /// Complex dimension.
    dim: usize,
}

impl ComplEx {
    /// Create a new ComplEx model with random initialization.
    ///
    /// `dim` is the complex dimension (each embedding stores `2 * dim` floats).
    /// Xavier uniform initialization: `Uniform(-sqrt(6/dim), sqrt(6/dim))`.
    #[cfg(feature = "rand")]
    pub fn new(num_entities: usize, num_relations: usize, dim: usize) -> Self {
        let mut rng = rand::rng();
        let scale = (6.0_f32 / dim as f32).sqrt();
        Self::from_vecs(
            init_vecs(&mut rng, num_entities, dim * 2, scale),
            init_vecs(&mut rng, num_relations, dim * 2, scale),
            dim,
        )
    }

    /// Create from pre-built embedding vectors.
    ///
    /// Both `entities` and `relations` must have inner length `dim * 2`.
    ///
    /// # Panics
    ///
    /// Panics if any dimension is wrong.
    pub fn from_vecs(entities: Vec<Vec<f32>>, relations: Vec<Vec<f32>>, dim: usize) -> Self {
        assert_dims(&entities, dim * 2, "entity (re+im)");
        assert_dims(&relations, dim * 2, "relation (re+im)");
        Self {
            entities,
            relations,
            dim,
        }
    }

    /// Entity embeddings: `[re..., im...]` contiguous layout.
    pub fn entities(&self) -> &[Vec<f32>] {
        &self.entities
    }

    /// Relation embeddings: `[re..., im...]` contiguous layout.
    pub fn relations(&self) -> &[Vec<f32>] {
        &self.relations
    }

    /// Complex dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Raw score: `Re(sum_i h_i * r_i * conj(t_i))`.
    ///
    /// Higher = more likely. Use [`Scorer::score`] for the negated
    /// distance-compatible version.
    ///
    /// Entity/relation layout: first `dim` floats are real, next `dim` are imaginary.
    pub fn score_triple(&self, head: usize, relation: usize, tail: usize) -> f32 {
        let h = &self.entities[head];
        let r = &self.relations[relation];
        let t = &self.entities[tail];
        let dim = self.dim;

        let mut dot = 0.0_f64;
        for i in 0..dim {
            let h_re = h[i] as f64;
            let h_im = h[dim + i] as f64;
            let r_re = r[i] as f64;
            let r_im = r[dim + i] as f64;
            let t_re = t[i] as f64;
            let t_im = t[dim + i] as f64;

            let hr_re = h_re * r_re - h_im * r_im;
            let hr_im = h_re * r_im + h_im * r_re;

            dot += hr_re * t_re + hr_im * t_im;
        }
        dot as f32
    }
}

impl Scorer for ComplEx {
    /// Returns `-score_triple()` so that lower = more likely.
    fn score(&self, head: usize, relation: usize, tail: usize) -> f32 {
        -self.score_triple(head, relation, tail)
    }

    fn num_entities(&self) -> usize {
        self.entities.len()
    }

    fn score_all_tails(&self, head: usize, relation: usize) -> Vec<f32> {
        let h = &self.entities[head];
        let r = &self.relations[relation];
        let dim = self.dim;
        // Precompute h * r (complex) once.
        let mut hr_re = vec![0.0_f64; dim];
        let mut hr_im = vec![0.0_f64; dim];
        for i in 0..dim {
            let h_re = h[i] as f64;
            let h_im = h[dim + i] as f64;
            let r_re = r[i] as f64;
            let r_im = r[dim + i] as f64;
            hr_re[i] = h_re * r_re - h_im * r_im;
            hr_im[i] = h_re * r_im + h_im * r_re;
        }
        self.entities
            .iter()
            .map(|t| {
                let mut dot = 0.0_f64;
                for i in 0..dim {
                    // Re(hr * conj(t)) = hr_re * t_re + hr_im * t_im
                    dot += hr_re[i] * t[i] as f64 + hr_im[i] * t[dim + i] as f64;
                }
                -(dot as f32)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// DistMult
// ---------------------------------------------------------------------------

/// DistMult: diagonal bilinear model in real space.
///
/// Score: `sum_i h_i * r_i * t_i`. Higher = more likely, so the `Scorer`
/// implementation returns the negated score.
///
/// DistMult can only model symmetric relations (`score(h,r,t) == score(t,r,h)`).
///
/// Initialization: Xavier uniform `Uniform(-sqrt(6/dim), sqrt(6/dim))`.
///
/// Reference: Yang et al. (2015), "Embedding Entities and Relations for
/// Learning and Inference in Knowledge Bases."
pub struct DistMult {
    entities: Vec<Vec<f32>>,
    relations: Vec<Vec<f32>>,
    dim: usize,
}

impl DistMult {
    /// Create a new DistMult model with random initialization.
    #[cfg(feature = "rand")]
    pub fn new(num_entities: usize, num_relations: usize, dim: usize) -> Self {
        let mut rng = rand::rng();
        let scale = (6.0_f32 / dim as f32).sqrt();
        Self::from_vecs(
            init_vecs(&mut rng, num_entities, dim, scale),
            init_vecs(&mut rng, num_relations, dim, scale),
            dim,
        )
    }

    /// Create from pre-built embedding vectors.
    ///
    /// # Panics
    ///
    /// Panics if any inner vector length differs from `dim`.
    pub fn from_vecs(entities: Vec<Vec<f32>>, relations: Vec<Vec<f32>>, dim: usize) -> Self {
        assert_dims(&entities, dim, "entity");
        assert_dims(&relations, dim, "relation");
        Self {
            entities,
            relations,
            dim,
        }
    }

    /// Entity embeddings.
    pub fn entities(&self) -> &[Vec<f32>] {
        &self.entities
    }

    /// Relation embeddings.
    pub fn relations(&self) -> &[Vec<f32>] {
        &self.relations
    }

    /// Embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Raw score: `sum_i h_i * r_i * t_i`. Higher = more likely.
    pub fn score_triple(&self, head: usize, relation: usize, tail: usize) -> f32 {
        let h = &self.entities[head];
        let r = &self.relations[relation];
        let t = &self.entities[tail];

        let mut dot = 0.0_f64;
        for i in 0..self.dim {
            dot += h[i] as f64 * r[i] as f64 * t[i] as f64;
        }
        dot as f32
    }
}

impl Scorer for DistMult {
    /// Returns `-score_triple()` so that lower = more likely.
    fn score(&self, head: usize, relation: usize, tail: usize) -> f32 {
        -self.score_triple(head, relation, tail)
    }

    fn num_entities(&self) -> usize {
        self.entities.len()
    }

    fn score_all_tails(&self, head: usize, relation: usize) -> Vec<f32> {
        let h = &self.entities[head];
        let r = &self.relations[relation];
        let dim = self.dim;
        // Precompute h * r once.
        let mut hr = vec![0.0_f64; dim];
        for i in 0..dim {
            hr[i] = h[i] as f64 * r[i] as f64;
        }
        self.entities
            .iter()
            .map(|t| {
                let mut dot = 0.0_f64;
                for i in 0..dim {
                    dot += hr[i] * t[i] as f64;
                }
                -(dot as f32) // negate for distance convention
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn assert_dims(vecs: &[Vec<f32>], expected: usize, label: &str) {
    for (i, v) in vecs.iter().enumerate() {
        assert_eq!(
            v.len(),
            expected,
            "{label} embedding {i} has length {}, expected {expected}",
            v.len()
        );
    }
}

#[cfg(feature = "rand")]
fn init_vecs(rng: &mut impl rand::Rng, count: usize, len: usize, scale: f32) -> Vec<Vec<f32>> {
    (0..count)
        .map(|_| (0..len).map(|_| rng.random_range(-scale..scale)).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- TransE ---------------------------------------------------------------

    #[test]
    fn transe_manual_score() {
        // h = [1, 0], r = [0, 1], t = [1, 1]
        // h + r - t = [0, 0], ||.||_2 = 0
        let model = TransE::from_vecs(
            vec![vec![1.0, 0.0], vec![1.0, 1.0]],
            vec![vec![0.0, 1.0]],
            2,
        );
        let score = model.score_triple(0, 0, 1);
        assert!((score - 0.0).abs() < 1e-6, "expected 0, got {score}");
    }

    #[test]
    fn transe_manual_nonzero() {
        // h = [3, 0], r = [0, 0], t = [0, 4]
        // h + r - t = [3, -4], ||.||_2 = 5
        let model = TransE::from_vecs(
            vec![vec![3.0, 0.0], vec![0.0, 4.0]],
            vec![vec![0.0, 0.0]],
            2,
        );
        let score = model.score_triple(0, 0, 1);
        assert!((score - 5.0).abs() < 1e-5, "expected 5, got {score}");
    }

    #[test]
    fn transe_scorer_trait() {
        let model = TransE::new(10, 3, 8);
        let scorer: &dyn Scorer = &model;
        assert_eq!(scorer.num_entities(), 10);
        let s = scorer.score(0, 0, 1);
        assert!(s.is_finite());
        assert!(s >= 0.0);
    }

    #[test]
    #[should_panic(expected = "entity embedding 0 has length 3, expected 2")]
    fn transe_rejects_bad_dims() {
        TransE::from_vecs(vec![vec![1.0, 2.0, 3.0]], vec![vec![1.0, 2.0]], 2);
    }

    // -- RotatE ---------------------------------------------------------------

    #[test]
    fn rotate_identity_rotation() {
        // Angle = 0 means rotation is identity (cos=1, sin=0).
        // h * r = h, so score = ||h - t||.
        // h = [1,0] (re=1, im=0), t = [1,0], score should be 0.
        let model = RotatE::from_vecs(
            vec![vec![1.0, 0.0], vec![1.0, 0.0]],
            vec![vec![0.0]], // angle = 0
            1,
            12.0,
        );
        let score = model.score_triple(0, 0, 1);
        assert!((score - 0.0).abs() < 1e-6, "expected 0, got {score}");
    }

    #[test]
    fn rotate_90_degrees() {
        use std::f32::consts::FRAC_PI_2;
        // h = (1, 0), rotate by pi/2 => (0, 1)
        // t = (0, 1), score should be 0.
        let model = RotatE::from_vecs(
            vec![
                vec![1.0, 0.0], // entity 0: re=1, im=0
                vec![0.0, 1.0], // entity 1: re=0, im=1
            ],
            vec![vec![FRAC_PI_2]], // 90 degrees
            1,
            12.0,
        );
        let score = model.score_triple(0, 0, 1);
        assert!(score < 1e-5, "expected ~0, got {score}");
    }

    #[test]
    fn rotate_scorer_trait() {
        let model = RotatE::new(10, 3, 8, 12.0);
        let scorer: &dyn Scorer = &model;
        assert_eq!(scorer.num_entities(), 10);
        let s = scorer.score(0, 0, 1);
        assert!(s.is_finite());
        assert!(s >= 0.0);
    }

    #[test]
    fn rotate_contiguous_layout_dim2() {
        use std::f32::consts::FRAC_PI_2;
        // dim=2: entity layout is [re0, re1, im0, im1]
        // Entity 0: (1+0i, 0+0i) -> [1, 0, 0, 0]
        // Entity 1: (0+1i, 0+0i) -> [0, 0, 1, 0]
        // Rotation by [pi/2, 0]: first component rotates 90 deg, second is identity.
        // (1+0i) * (cos(pi/2)+i*sin(pi/2)) = 0+1i
        // (0+0i) * (cos(0)+i*sin(0)) = 0+0i
        // Result: (0+1i, 0+0i) -> matches entity 1.
        let model = RotatE::from_vecs(
            vec![
                vec![1.0, 0.0, 0.0, 0.0], // entity 0: [re0=1, re1=0, im0=0, im1=0]
                vec![0.0, 0.0, 1.0, 0.0], // entity 1: [re0=0, re1=0, im0=1, im1=0]
            ],
            vec![vec![FRAC_PI_2, 0.0]], // rotate first dim by 90 deg
            2,
            12.0,
        );
        let score = model.score_triple(0, 0, 1);
        assert!(score < 1e-5, "expected ~0, got {score}");
    }

    // -- ComplEx ---------------------------------------------------------------

    #[test]
    fn complex_manual_score() {
        // h = (1+0i), r = (1+0i), t = (1+0i), dim=1
        // h*r*conj(t) = 1*1*1 = 1. Re = 1.
        let model = ComplEx::from_vecs(vec![vec![1.0, 0.0]], vec![vec![1.0, 0.0]], 1);
        let score = model.score_triple(0, 0, 0);
        assert!((score - 1.0).abs() < 1e-6, "expected 1.0, got {score}");
    }

    #[test]
    fn complex_imaginary_parts() {
        // h = (0+1i), r = (0+1i), t = (1+0i) [entity 1]
        // h*r = (0*0 - 1*1) + (0*1 + 1*0)i = -1 + 0i
        // conj(t) = (1 - 0i)
        // Re((-1+0i)*(1-0i)) = -1
        let model = ComplEx::from_vecs(
            vec![
                vec![0.0, 1.0], // entity 0: 0+1i
                vec![1.0, 0.0], // entity 1: 1+0i
            ],
            vec![vec![0.0, 1.0]],
            1,
        );
        let score = model.score_triple(0, 0, 1);
        assert!((score - (-1.0)).abs() < 1e-6, "expected -1.0, got {score}");
    }

    #[test]
    fn complex_scorer_negates() {
        // Scorer returns -score_triple, so a positive raw score becomes negative.
        let model = ComplEx::from_vecs(vec![vec![1.0, 0.0]], vec![vec![1.0, 0.0]], 1);
        let raw = model.score_triple(0, 0, 0);
        let via_scorer = model.score(0, 0, 0);
        assert!((via_scorer - (-raw)).abs() < 1e-6);
    }

    // -- DistMult -------------------------------------------------------------

    #[test]
    fn distmult_manual_score() {
        // h = [2, 3], r = [1, -1], t = [4, 5]
        // sum = 2*1*4 + 3*(-1)*5 = 8 - 15 = -7
        let model = DistMult::from_vecs(
            vec![vec![2.0, 3.0], vec![4.0, 5.0]],
            vec![vec![1.0, -1.0]],
            2,
        );
        let score = model.score_triple(0, 0, 1);
        assert!((score - (-7.0)).abs() < 1e-5, "expected -7.0, got {score}");
    }

    #[test]
    fn distmult_symmetric() {
        // DistMult is symmetric: score(h, r, t) == score(t, r, h).
        let model = DistMult::new(10, 3, 16);
        let s1 = model.score_triple(0, 0, 1);
        let s2 = model.score_triple(1, 0, 0);
        assert!(
            (s1 - s2).abs() < 1e-5,
            "DistMult should be symmetric: {s1} vs {s2}"
        );
    }

    #[test]
    fn distmult_scorer_negates() {
        let model = DistMult::from_vecs(vec![vec![1.0], vec![2.0]], vec![vec![3.0]], 1);
        let raw = model.score_triple(0, 0, 1);
        let via_scorer = model.score(0, 0, 1);
        assert!((via_scorer - (-raw)).abs() < 1e-6);
    }

    // -- Batch scoring --------------------------------------------------------

    #[test]
    fn score_all_tails_length() {
        let model = TransE::new(10, 3, 8);
        let scores = model.score_all_tails(0, 0);
        assert_eq!(scores.len(), 10);
        assert!(scores.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn top_k_tails_sorted() {
        let model = TransE::new(20, 3, 8);
        let top = model.top_k_tails(0, 0, 5);
        assert_eq!(top.len(), 5);
        for w in top.windows(2) {
            assert!(w[0].1 <= w[1].1, "top_k should be sorted ascending");
        }
    }

    #[test]
    fn top_k_heads_sorted() {
        let model = TransE::new(20, 3, 8);
        let top = model.top_k_heads(0, 0, 5);
        assert_eq!(top.len(), 5);
        for w in top.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
    }
}
