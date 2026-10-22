//! Dataset loading for KGE benchmark triple files.
//!
//! Core types and loading functions are provided by [`lattix::kge`].
//! This module adds tranz-specific extensions (reciprocal relations, splitting).

// Re-export core types from lattix::kge.
pub use lattix::kge::{
    load_dataset, load_triples, Dataset, FilterIndex, InternedDataset, Triple, TripleIds, Vocab,
};

/// Intern one distinct inverse display name per existing relation.
///
/// The returned count is the number of original relations. New IDs are
/// therefore exactly `count + original_id`, even when an input relation
/// already uses the conventional `{name}_inv` spelling.
pub(crate) fn intern_inverse_relation_names(relations: &mut Vocab) -> usize {
    let n_relations = relations.len();
    let original_names: Vec<String> = (0..n_relations)
        .map(|id| {
            relations
                .get(id)
                .expect("relation ID must exist in its vocabulary")
                .to_owned()
        })
        .collect();

    for (original_id, name) in original_names.into_iter().enumerate() {
        let base = format!("{name}_inv");
        let inverse_name = if relations.id(&base).is_none() {
            base
        } else {
            let mut suffix = 2usize;
            loop {
                let candidate = format!("{base}_{suffix}");
                if relations.id(&candidate).is_none() {
                    break candidate;
                }
                suffix = suffix
                    .checked_add(1)
                    .expect("inverse relation name suffix overflowed usize");
            }
        };
        let inverse_id = relations.intern(inverse_name);
        assert_eq!(
            inverse_id,
            n_relations + original_id,
            "unused inverse relation name must receive its fixed-offset ID"
        );
    }

    n_relations
}

/// Extension methods for [`InternedDataset`] specific to tranz.
pub trait InternedDatasetExt {
    /// Add reciprocal (inverse) relations.
    ///
    /// For each original relation `r` with ID `i`, creates a new relation
    /// `r_inv` with ID `num_relations + i`. For each triple `(h, r, t)`,
    /// adds `(t, r_inv, h)` to the same split.
    /// If that display name is already in use, the first unused suffix
    /// (`r_inv_2`, `r_inv_3`, and so on) is used instead.
    ///
    /// This consistently improves all models (Ali et al., 2022 / PyKEEN).
    fn add_reciprocals(&mut self);
}

impl InternedDatasetExt for InternedDataset {
    fn add_reciprocals(&mut self) {
        let n_rel = intern_inverse_relation_names(&mut self.relations);

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
    let mut triples = Vec::new();
    let mut dropped = 0usize;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let sep = if trimmed.contains('\t') { '\t' } else { ',' };
        let parts: Vec<&str> = trimmed.split(sep).map(str::trim).collect();
        if parts.len() >= 3 {
            triples.push(Triple::new(parts[0], parts[1], parts[2]));
        } else {
            dropped += 1;
        }
    }
    if dropped > 0 {
        eprintln!("warning: skipped {dropped} malformed lines (expected 3 fields per line)");
    }
    triples
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
    fn parse_flexible_drops_malformed_lines() {
        let content = "Alice\tknows\tBob\nbad line\nCarol\tknows\tDave\n";
        let triples = super::parse_flexible(content);
        assert_eq!(triples.len(), 2);
        assert_eq!(triples[0].head, "Alice");
        assert_eq!(triples[1].head, "Carol");
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

    fn assert_reciprocal_names(original_names: &[&str], inverse_names: &[&str]) {
        assert_eq!(original_names.len(), inverse_names.len());
        let train: Vec<_> = original_names
            .iter()
            .enumerate()
            .map(|(i, &relation)| Triple::new(format!("head-{i}"), relation, format!("tail-{i}")))
            .collect();
        let mut interned = Dataset::new(train, Vec::new(), Vec::new()).into_interned();
        let n_rel = original_names.len();

        for (id, &name) in original_names.iter().enumerate() {
            assert_eq!(interned.relations.get(id), Some(name));
            assert_eq!(interned.relations.id(name), Some(id));
        }

        interned.add_reciprocals();

        assert_eq!(interned.num_relations(), n_rel * 2);
        assert_eq!(interned.train.len(), n_rel * 2);
        for (i, &inverse_name) in inverse_names.iter().enumerate() {
            assert_eq!(interned.relations.get(n_rel + i), Some(inverse_name));
            assert_eq!(interned.relations.id(inverse_name), Some(n_rel + i));
            let original = interned.train[i];
            let inverse = interned.train[n_rel + i];
            assert_eq!(inverse.head, original.tail);
            assert_eq!(inverse.relation, n_rel + i);
            assert_eq!(inverse.tail, original.head);
            assert!(inverse.relation < interned.num_relations());
        }
    }

    #[test]
    fn reciprocal_names_skip_direct_collision() {
        assert_reciprocal_names(&["r", "r_inv"], &["r_inv_2", "r_inv_inv"]);
    }

    #[test]
    fn reciprocal_names_skip_chained_collisions() {
        assert_reciprocal_names(
            &["r", "r_inv", "r_inv_inv"],
            &["r_inv_2", "r_inv_inv_2", "r_inv_inv_inv"],
        );
    }

    #[test]
    fn reciprocal_names_skip_numeric_suffix_collisions() {
        assert_reciprocal_names(
            &["r", "r_inv", "r_inv_2", "r_inv_3"],
            &["r_inv_4", "r_inv_inv", "r_inv_2_inv", "r_inv_3_inv"],
        );
    }
}
