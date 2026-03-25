//! Dataset loading for WN18RR-format triple files.
//!
//! Expects a directory containing `train.txt`, `valid.txt`, `test.txt`,
//! each with tab-separated triples: `head\trelation\ttail`.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// A raw triple with string identifiers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Triple {
    /// Head entity name.
    pub head: String,
    /// Relation name.
    pub relation: String,
    /// Tail entity name.
    pub tail: String,
}

/// Raw dataset with train/valid/test splits.
#[derive(Debug, Clone)]
pub struct Dataset {
    /// Training triples.
    pub train: Vec<Triple>,
    /// Validation triples.
    pub valid: Vec<Triple>,
    /// Test triples.
    pub test: Vec<Triple>,
}

/// Dataset with integer IDs and vocabulary mappings.
#[derive(Debug, Clone)]
pub struct InternedDataset {
    /// Training triples as `(head, relation, tail)` IDs.
    pub train: Vec<(usize, usize, usize)>,
    /// Validation triples as `(head, relation, tail)` IDs.
    pub valid: Vec<(usize, usize, usize)>,
    /// Test triples as `(head, relation, tail)` IDs.
    pub test: Vec<(usize, usize, usize)>,
    /// Entity name to ID.
    pub entity_to_id: HashMap<String, usize>,
    /// Relation name to ID.
    pub relation_to_id: HashMap<String, usize>,
    /// ID to entity name.
    pub id_to_entity: Vec<String>,
    /// ID to relation name.
    pub id_to_relation: Vec<String>,
}

impl InternedDataset {
    /// Number of distinct entities.
    pub fn num_entities(&self) -> usize {
        self.id_to_entity.len()
    }

    /// Number of distinct relations.
    pub fn num_relations(&self) -> usize {
        self.id_to_relation.len()
    }

    /// All triples across all splits.
    pub fn all_triples(&self) -> Vec<(usize, usize, usize)> {
        let mut all = Vec::with_capacity(self.train.len() + self.valid.len() + self.test.len());
        all.extend_from_slice(&self.train);
        all.extend_from_slice(&self.valid);
        all.extend_from_slice(&self.test);
        all
    }
}

fn parse_triples(content: &str) -> Vec<Triple> {
    content
        .lines()
        .filter(|line| !line.is_empty())
        .filter_map(|line| {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 3 {
                Some(Triple {
                    head: parts[0].to_string(),
                    relation: parts[1].to_string(),
                    tail: parts[2].to_string(),
                })
            } else {
                None
            }
        })
        .collect()
}

/// Load a dataset from a directory containing `train.txt`, `valid.txt`, `test.txt`.
///
/// Each file has tab-separated triples: `head\trelation\ttail`, one per line.
pub fn load_dataset(path: &Path) -> Result<Dataset, crate::Error> {
    let read = |name: &str| -> Result<Vec<Triple>, crate::Error> {
        let file_path = path.join(name);
        let content = fs::read_to_string(&file_path).map_err(|e| {
            crate::Error::Io(std::io::Error::new(
                e.kind(),
                format!("{}: {e}", file_path.display()),
            ))
        })?;
        Ok(parse_triples(&content))
    };

    Ok(Dataset {
        train: read("train.txt")?,
        valid: read("valid.txt")?,
        test: read("test.txt")?,
    })
}

impl Dataset {
    /// Convert to an interned dataset with integer IDs.
    ///
    /// Entities and relations are assigned IDs in order of first appearance
    /// (train, then valid, then test).
    pub fn into_interned(self) -> InternedDataset {
        let mut entity_to_id: HashMap<String, usize> = HashMap::new();
        let mut relation_to_id: HashMap<String, usize> = HashMap::new();
        let mut id_to_entity: Vec<String> = Vec::new();
        let mut id_to_relation: Vec<String> = Vec::new();

        let mut intern_entity = |name: &str| -> usize {
            let len = entity_to_id.len();
            *entity_to_id.entry(name.to_string()).or_insert_with(|| {
                id_to_entity.push(name.to_string());
                len
            })
        };

        let mut intern_relation = |name: &str| -> usize {
            let len = relation_to_id.len();
            *relation_to_id.entry(name.to_string()).or_insert_with(|| {
                id_to_relation.push(name.to_string());
                len
            })
        };

        let mut intern_triples = |triples: &[Triple]| -> Vec<(usize, usize, usize)> {
            triples
                .iter()
                .map(|t| {
                    let h = intern_entity(&t.head);
                    let r = intern_relation(&t.relation);
                    let tl = intern_entity(&t.tail);
                    (h, r, tl)
                })
                .collect()
        };

        let train = intern_triples(&self.train);
        let valid = intern_triples(&self.valid);
        let test = intern_triples(&self.test);

        InternedDataset {
            train,
            valid,
            test,
            entity_to_id,
            relation_to_id,
            id_to_entity,
            id_to_relation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_triples(dir: &Path, name: &str, triples: &[(&str, &str, &str)]) {
        let path = dir.join(name);
        let mut f = fs::File::create(path).unwrap();
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
        assert_eq!(interned.entity_to_id["A"], 0);
        assert_eq!(interned.entity_to_id["B"], 1);
        assert_eq!(interned.entity_to_id["C"], 2);
    }

    #[test]
    fn parse_skips_blank_lines() {
        let content = "A\tr1\tB\n\nC\tr2\tD\n";
        let triples = parse_triples(content);
        assert_eq!(triples.len(), 2);
    }
}
