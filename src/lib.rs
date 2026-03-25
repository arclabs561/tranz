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
//! - [`TransE`]: `head + relation ≈ tail` (Bordes et al., 2013)
//! - `RotatE`: `head ∘ relation ≈ tail` in complex space (Sun et al., 2019)
//! - `ComplEx`: diagonal complex bilinear (Trouillon et al., 2016)

#![warn(missing_docs)]

/// Errors from tranz operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Dimension mismatch.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },
    /// Internal error.
    #[error("{0}")]
    Internal(String),
}

/// TransE: translational distance model.
///
/// Scores triples by `||head + relation - tail||`. Lower = more likely.
///
/// Reference: Bordes et al. (2013), "Translating Embeddings for Modeling
/// Multi-relational Data."
pub struct TransE {
    /// Entity embeddings: `[num_entities, dim]`.
    pub entities: Vec<Vec<f32>>,
    /// Relation embeddings: `[num_relations, dim]`.
    pub relations: Vec<Vec<f32>>,
    /// Embedding dimension.
    pub dim: usize,
}

impl TransE {
    /// Create a new TransE model with random initialization.
    #[cfg(feature = "rand")]
    pub fn new(num_entities: usize, num_relations: usize, dim: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::rng();
        let scale = 6.0_f32 / (dim as f32).sqrt();

        let entities: Vec<Vec<f32>> = (0..num_entities)
            .map(|_| (0..dim).map(|_| rng.random_range(-scale..scale)).collect())
            .collect();
        let relations: Vec<Vec<f32>> = (0..num_relations)
            .map(|_| (0..dim).map(|_| rng.random_range(-scale..scale)).collect())
            .collect();

        Self {
            entities,
            relations,
            dim,
        }
    }

    /// Score a triple (h, r, t). Lower = more likely.
    pub fn score(&self, head: usize, relation: usize, tail: usize) -> f32 {
        let h = &self.entities[head];
        let r = &self.relations[relation];
        let t = &self.entities[tail];

        let mut dist_sq = 0.0f32;
        for i in 0..self.dim {
            let d = h[i] + r[i] - t[i];
            dist_sq += d * d;
        }
        dist_sq.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transe_creates_and_scores() {
        let model = TransE::new(100, 10, 50);
        assert_eq!(model.entities.len(), 100);
        assert_eq!(model.relations.len(), 10);
        assert_eq!(model.dim, 50);

        let score = model.score(0, 0, 1);
        assert!(score.is_finite());
        assert!(score >= 0.0);
    }
}
