//! Embedding import/export.
//!
//! Supports two formats:
//!
//! - **w2v TSV**: header line `count dim\n`, then one line per entity:
//!   `entity_name\tval0\tval1\t...\n`. Compatible with GloVe/word2vec tools.
//! - **Binary + JSON sidecar**: flat `f32` matrix in little-endian binary,
//!   plus a JSON file mapping entity names to row indices.

use std::io::{self, BufRead, BufWriter, Write};
use std::path::Path;

/// Write embeddings in w2v TSV format.
///
/// Format: first line is `count dim`, subsequent lines are
/// `name<TAB>val0<TAB>val1<TAB>...`.
///
/// `names` and `vecs` must have the same length.
pub fn write_w2v_tsv(
    writer: &mut impl Write,
    names: &[String],
    vecs: &[Vec<f32>],
) -> io::Result<()> {
    assert_eq!(names.len(), vecs.len(), "names and vecs must match");
    if vecs.is_empty() {
        return Ok(());
    }
    let dim = vecs[0].len();
    let mut w = BufWriter::new(writer);
    writeln!(w, "{} {dim}", names.len())?;
    for (name, vec) in names.iter().zip(vecs.iter()) {
        write!(w, "{name}")?;
        for v in vec {
            write!(w, "\t{v}")?;
        }
        writeln!(w)?;
    }
    w.flush()
}

/// Read embeddings from w2v TSV format.
///
/// Returns `(names, vecs)`.
pub fn read_w2v_tsv(reader: impl io::Read) -> io::Result<(Vec<String>, Vec<Vec<f32>>)> {
    let buf = io::BufReader::new(reader);
    let mut lines = buf.lines();

    let header = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "empty file"))??;
    let parts: Vec<&str> = header.split_whitespace().collect();
    if parts.len() != 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("expected 'count dim' header, got: {header}"),
        ));
    }
    let count: usize = parts[0]
        .parse()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("bad count: {e}")))?;
    let dim: usize = parts[1]
        .parse()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("bad dim: {e}")))?;

    let mut names = Vec::with_capacity(count);
    let mut vecs = Vec::with_capacity(count);

    for line in lines {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split('\t');
        let name = parts
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "empty line"))?
            .to_string();
        let vec: Vec<f32> = parts
            .map(|s| {
                s.parse::<f32>().map_err(|e| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("bad float: {e}"))
                })
            })
            .collect::<io::Result<_>>()?;
        if vec.len() != dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("expected {dim} values for '{name}', got {}", vec.len()),
            ));
        }
        names.push(name);
        vecs.push(vec);
    }

    Ok((names, vecs))
}

/// Write embeddings as flat little-endian f32 binary.
///
/// Layout: `count * dim` f32 values, row-major. No header.
/// Write the vocabulary separately (e.g., as JSON or TSV).
pub fn write_binary(writer: &mut impl Write, vecs: &[Vec<f32>]) -> io::Result<()> {
    let mut w = BufWriter::new(writer);
    for vec in vecs {
        for &v in vec {
            w.write_all(&v.to_le_bytes())?;
        }
    }
    w.flush()
}

/// Write entity-to-ID mapping as TSV.
///
/// Format: `id<TAB>name` per line.
pub fn write_vocab_tsv(writer: &mut impl Write, names: &[String]) -> io::Result<()> {
    let mut w = BufWriter::new(writer);
    for (id, name) in names.iter().enumerate() {
        writeln!(w, "{id}\t{name}")?;
    }
    w.flush()
}

/// Export entity and relation embeddings from a trained model to a directory.
///
/// Creates:
/// - `entities.tsv` (w2v format)
/// - `relations.tsv` (w2v format)
pub fn export_embeddings(
    dir: &Path,
    entity_names: &[String],
    entity_vecs: &[Vec<f32>],
    relation_names: &[String],
    relation_vecs: &[Vec<f32>],
) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;

    let mut ent_file = std::fs::File::create(dir.join("entities.tsv"))?;
    write_w2v_tsv(&mut ent_file, entity_names, entity_vecs)?;

    let mut rel_file = std::fs::File::create(dir.join("relations.tsv"))?;
    write_w2v_tsv(&mut rel_file, relation_names, relation_vecs)?;

    Ok(())
}

/// Import entity embeddings from a w2v TSV file.
///
/// Returns `(names, vecs)`.
pub fn import_embeddings(path: &Path) -> io::Result<(Vec<String>, Vec<Vec<f32>>)> {
    let file = std::fs::File::open(path)?;
    read_w2v_tsv(file)
}

/// Loaded entity and relation embeddings.
pub struct LoadedEmbeddings {
    /// Entity names in row order.
    pub entity_names: Vec<String>,
    /// Entity embedding vectors.
    pub entity_vecs: Vec<Vec<f32>>,
    /// Relation names in row order.
    pub relation_names: Vec<String>,
    /// Relation embedding vectors.
    pub relation_vecs: Vec<Vec<f32>>,
}

/// Load entity and relation embeddings from a directory.
///
/// Expects `entities.tsv` and `relations.tsv` in w2v format (as written
/// by [`export_embeddings`]).
pub fn load_embeddings(dir: &Path) -> io::Result<LoadedEmbeddings> {
    let (entity_names, entity_vecs) = import_embeddings(&dir.join("entities.tsv"))?;
    let (relation_names, relation_vecs) = import_embeddings(&dir.join("relations.tsv"))?;
    Ok(LoadedEmbeddings {
        entity_names,
        entity_vecs,
        relation_names,
        relation_vecs,
    })
}

/// Flatten `Vec<Vec<f32>>` into a contiguous row-major `Vec<f32>`.
///
/// Useful for handing off to FAISS, Qdrant, or any system expecting
/// a flat `[f32]` matrix of shape `[num_rows, dim]`.
pub fn flatten_matrix(vecs: &[Vec<f32>]) -> Vec<f32> {
    let total: usize = vecs.iter().map(|v| v.len()).sum();
    let mut flat = Vec::with_capacity(total);
    for v in vecs {
        flat.extend_from_slice(v);
    }
    flat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w2v_roundtrip() {
        let names = vec!["alice".to_string(), "bob".to_string()];
        let vecs = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

        let mut buf = Vec::new();
        write_w2v_tsv(&mut buf, &names, &vecs).unwrap();

        let (read_names, read_vecs) = read_w2v_tsv(buf.as_slice()).unwrap();
        assert_eq!(read_names, names);
        assert_eq!(read_vecs.len(), 2);
        for (a, b) in vecs.iter().zip(read_vecs.iter()) {
            for (x, y) in a.iter().zip(b.iter()) {
                assert!((x - y).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn w2v_empty() {
        let mut buf = Vec::new();
        write_w2v_tsv(&mut buf, &[], &[]).unwrap();
        assert!(buf.is_empty());
    }

    #[test]
    fn binary_write() {
        let vecs = vec![vec![1.0_f32, 2.0], vec![3.0, 4.0]];
        let mut buf = Vec::new();
        write_binary(&mut buf, &vecs).unwrap();
        assert_eq!(buf.len(), 4 * 4); // 4 floats * 4 bytes
        let first = f32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        assert!((first - 1.0).abs() < 1e-6);
    }

    #[test]
    fn export_import_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let ent_names = vec!["a".to_string(), "b".to_string()];
        let ent_vecs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let rel_names = vec!["r1".to_string()];
        let rel_vecs = vec![vec![0.5, 0.5]];

        export_embeddings(dir.path(), &ent_names, &ent_vecs, &rel_names, &rel_vecs).unwrap();

        let loaded = load_embeddings(dir.path()).unwrap();
        assert_eq!(loaded.entity_names, ent_names);
        assert_eq!(loaded.relation_names, rel_names);
        assert_eq!(loaded.entity_vecs.len(), 2);
        assert_eq!(loaded.relation_vecs.len(), 1);
    }

    #[test]
    fn flatten_matrix_works() {
        let vecs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let flat = flatten_matrix(&vecs);
        assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
