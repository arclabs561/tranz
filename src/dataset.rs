//! Dataset loading for KGE benchmark triple files.
//!
//! Core types and loading functions are provided by [`lattix::kge`].
//! This module adds tranz-specific extensions (reciprocal relations, splitting).

// Re-export core types from lattix::kge.
pub use lattix::kge::{
    load_dataset, load_triples, Dataset, FilterIndex, InternedDataset, Triple, TripleIds, Vocab,
};

/// Extension methods for [`InternedDataset`] specific to tranz.
pub trait InternedDatasetExt {
    /// Add reciprocal (inverse) relations.
    ///
    /// For each original relation `r` with ID `i`, creates a new relation
    /// `r_inv` with ID `num_relations + i`. For each triple `(h, r, t)`,
    /// adds `(t, r_inv, h)` to the same split.
    ///
    /// This consistently improves all models (Ali et al., 2022 / PyKEEN).
    fn add_reciprocals(&mut self);
}

impl InternedDatasetExt for InternedDataset {
    fn add_reciprocals(&mut self) {
        let n_rel = self.relations.len();

        // Add inverse relation names.
        for i in 0..n_rel {
            let name = format!("{}_inv", self.relations.get(i).unwrap());
            self.relations.intern(name);
        }

        // Add reciprocal triples to each split.
        fn augment(triples: &mut Vec<TripleIds>, n_rel: usize) {
            let originals: Vec<_> = triples.clone();
            triples.reserve(originals.len());
            for t in &originals {
                triples.push(TripleIds::new(t.tail, t.relation + n_rel, t.head));
            }
        }
        augment(&mut self.train, n_rel);
        augment(&mut self.valid, n_rel);
        augment(&mut self.test, n_rel);
    }
}

/// Extension methods for [`Dataset`] specific to tranz.
pub trait DatasetExt {
    /// Split the training set into train/valid/test by ratio.
    ///
    /// `valid_frac` and `test_frac` are fractions of the total triples.
    /// Remaining triples stay in train. Splits are deterministic
    /// (takes from the end of the list).
    fn split(self, valid_frac: f32, test_frac: f32) -> Dataset;

    /// Load triples from a single CSV/TSV file (flexible separator).
    ///
    /// All triples go into the `train` split. Use [`DatasetExt::split`]
    /// to create validation and test splits.
    fn load_flexible(path: &std::path::Path) -> Result<Dataset, lattix::Error>;
}

impl DatasetExt for Dataset {
    fn split(self, valid_frac: f32, test_frac: f32) -> Dataset {
        let total = self.train.len() + self.valid.len() + self.test.len();
        let mut all = self.train;
        let mut v = self.valid;
        let mut t = self.test;
        all.append(&mut v);
        all.append(&mut t);

        let n_test = (total as f32 * test_frac).round() as usize;
        let n_valid = (total as f32 * valid_frac).round() as usize;
        let test = all.split_off(all.len().saturating_sub(n_test));
        let valid = all.split_off(all.len().saturating_sub(n_valid));

        Dataset::new(all, valid, test)
    }

    fn load_flexible(path: &std::path::Path) -> Result<Dataset, lattix::Error> {
        let content = std::fs::read_to_string(path)?;
        let triples = parse_flexible(&content);
        Ok(Dataset::new(triples, Vec::new(), Vec::new()))
    }
}

fn parse_flexible(content: &str) -> Vec<Triple> {
    content
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .filter_map(|line| {
            let sep = if line.contains('\t') { '\t' } else { ',' };
            let parts: Vec<&str> = line.split(sep).map(str::trim).collect();
            if parts.len() >= 3 {
                Some(Triple::new(parts[0], parts[1], parts[2]))
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_triples(dir: &std::path::Path, name: &str, triples: &[(&str, &str, &str)]) {
        let path = dir.join(name);
        let mut f = std::fs::File::create(path).unwrap();
        for (h, r, t) in triples {
            writeln!(f, "{h}\t{r}\t{t}").unwrap();
        }
    }

    #[test]
    fn load_and_intern() {
        let dir = tempfile::tempdir().unwrap();
        write_triples(
            dir.path(),
            "train.txt",
            &[("A", "r1", "B"), ("B", "r2", "C")],
        );
        write_triples(dir.path(), "valid.txt", &[("A", "r1", "C")]);
        write_triples(dir.path(), "test.txt", &[("C", "r2", "A")]);

        let ds = load_dataset(dir.path()).unwrap();
        assert_eq!(ds.train.len(), 2);
        assert_eq!(ds.valid.len(), 1);
        assert_eq!(ds.test.len(), 1);

        let interned = ds.into_interned();
        assert_eq!(interned.num_entities(), 3);
        assert_eq!(interned.num_relations(), 2);
        assert_eq!(interned.all_triples().len(), 4);

        // First-appearance order: A=0, B=1, C=2
        assert_eq!(interned.entities.id("A"), Some(0));
        assert_eq!(interned.entities.id("B"), Some(1));
        assert_eq!(interned.entities.id("C"), Some(2));
    }

    #[test]
    fn reciprocal_relations() {
        let dir = tempfile::tempdir().unwrap();
        write_triples(dir.path(), "train.txt", &[("A", "r1", "B")]);
        write_triples(dir.path(), "valid.txt", &[("B", "r1", "C")]);
        write_triples(dir.path(), "test.txt", &[("C", "r1", "A")]);

        let ds = load_dataset(dir.path()).unwrap();
        let mut interned = ds.into_interned();
        assert_eq!(interned.num_relations(), 1);

        interned.add_reciprocals();
        assert_eq!(interned.num_relations(), 2);
        assert_eq!(interned.relations.get(1), Some("r1_inv"));

        // Train: (A,r1,B) + (B,r1_inv,A)
        assert_eq!(interned.train.len(), 2);
        let t = interned.train[1];
        assert_eq!(t.head, interned.entities.id("B").unwrap());
        assert_eq!(t.relation, 1); // r1_inv
        assert_eq!(t.tail, interned.entities.id("A").unwrap());
    }

    #[test]
    fn load_flexible_csv() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("triples.csv");
        std::fs::write(&path, "# comment\nAlice,knows,Bob\nBob,works_at,Acme\n").unwrap();

        let ds = Dataset::load_flexible(&path).unwrap();
        assert_eq!(ds.train.len(), 2);
        assert_eq!(ds.train[0].head, "Alice");
        assert!(ds.valid.is_empty());
    }

    #[test]
    fn dataset_split() {
        let ds = Dataset::new(
            (0..100)
                .map(|i| Triple::new(format!("e{i}"), "r", format!("e{}", i + 1)))
                .collect(),
            Vec::new(),
            Vec::new(),
        );
        let ds = ds.split(0.1, 0.1);
        assert_eq!(ds.test.len(), 10);
        assert_eq!(ds.valid.len(), 10);
        assert_eq!(ds.train.len(), 80);
    }

    #[test]
    fn reciprocal_with_multiple_relations() {
        let dir = tempfile::tempdir().unwrap();
        write_triples(
            dir.path(),
            "train.txt",
            &[("A", "r1", "B"), ("C", "r2", "D")],
        );
        write_triples(dir.path(), "valid.txt", &[]);
        write_triples(dir.path(), "test.txt", &[]);

        let ds = load_dataset(dir.path()).unwrap();
        let mut interned = ds.into_interned();
        assert_eq!(interned.num_relations(), 2);

        interned.add_reciprocals();
        assert_eq!(interned.num_relations(), 4);
        assert_eq!(interned.relations.get(2), Some("r1_inv"));
        assert_eq!(interned.relations.get(3), Some("r2_inv"));
        assert_eq!(interned.train.len(), 4);
    }
}
