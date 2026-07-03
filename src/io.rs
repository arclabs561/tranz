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

#[cfg(feature = "artifact-manifest")]
use sha2::{Digest, Sha256};

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

/// Training-output manifest for exported embeddings.
///
/// The manifest describes the files written by [`export_embeddings`] plus the
/// training context needed to compare or reproduce the output directory.
#[cfg(feature = "artifact-manifest")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EmbeddingManifest {
    /// Manifest schema identifier.
    pub schema: String,
    /// Model family, e.g. `transe`, `rotate`, `complex`, or `distmult`.
    pub model: String,
    /// Score ordering used by the exported model API.
    pub score_order: String,
    /// Exported artifact descriptors.
    pub artifacts: Vec<EmbeddingArtifact>,
    /// Dataset and split information.
    pub dataset: ManifestDataset,
    /// Training configuration.
    pub training: ManifestTraining,
    /// Optional aggregate evaluation metrics.
    pub metrics: Option<ManifestMetrics>,
}

/// Descriptor for one exported embedding artifact.
#[cfg(feature = "artifact-manifest")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EmbeddingArtifact {
    /// Relative path inside the export directory.
    pub path: String,
    /// Media-type-like artifact kind.
    pub artifact_type: String,
    /// On-disk format name.
    pub format: String,
    /// SHA-256 digest as lowercase hex.
    pub sha256: String,
    /// File size in bytes.
    pub bytes: u64,
    /// Matrix rows.
    pub rows: usize,
    /// Floats per row.
    pub dim: usize,
}

/// Dataset metadata captured in an embedding export manifest.
#[cfg(feature = "artifact-manifest")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ManifestDataset {
    /// Source kind, e.g. `directory` or `triple_file`.
    pub source_kind: String,
    /// Source path as supplied to the CLI.
    pub source_path: String,
    /// Split policy used for train/valid/test.
    pub split: String,
    /// Entity vocabulary size.
    pub entities: usize,
    /// Relation vocabulary size after any augmentation.
    pub relations: usize,
    /// Training triple count.
    pub train_triples: usize,
    /// Validation triple count.
    pub valid_triples: usize,
    /// Test triple count.
    pub test_triples: usize,
}

/// Training configuration captured in an embedding export manifest.
#[cfg(feature = "artifact-manifest")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ManifestTraining {
    /// Trainer identifier.
    pub trainer: String,
    /// Burn backend feature used by the CLI.
    pub backend: String,
    /// Embedding dimension requested by the user.
    pub dim: usize,
    /// Initialization scale.
    pub init_scale: f64,
    /// Learning rate.
    pub lr: f64,
    /// Label smoothing epsilon.
    pub label_smoothing: f64,
    /// N3 regularization coefficient.
    pub n3_reg: f64,
    /// Batch size.
    pub batch_size: usize,
    /// Number of epochs.
    pub epochs: usize,
    /// Whether reciprocal relations were added before training.
    pub reciprocals: bool,
    /// Final training loss, if at least one epoch ran.
    pub final_loss: Option<f32>,
}

/// Aggregate link-prediction metrics captured in an embedding export manifest.
#[cfg(feature = "artifact-manifest")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ManifestMetrics {
    /// Mean reciprocal rank.
    pub mrr: f32,
    /// Head-prediction MRR.
    pub head_mrr: f32,
    /// Tail-prediction MRR.
    pub tail_mrr: f32,
    /// Mean rank.
    pub mean_rank: f32,
    /// Hits@1.
    pub hits_at_1: f32,
    /// Hits@3.
    pub hits_at_3: f32,
    /// Hits@10.
    pub hits_at_10: f32,
}

/// Describe an exported embedding file using its current on-disk bytes.
#[cfg(feature = "artifact-manifest")]
pub fn describe_embedding_artifact(
    dir: &Path,
    path: &str,
    artifact_type: &str,
    format: &str,
    rows: usize,
    dim: usize,
) -> io::Result<EmbeddingArtifact> {
    let full_path = dir.join(path);
    let bytes = std::fs::metadata(&full_path)?.len();
    let sha256 = sha256_file(&full_path)?;
    Ok(EmbeddingArtifact {
        path: path.to_string(),
        artifact_type: artifact_type.to_string(),
        format: format.to_string(),
        sha256,
        bytes,
        rows,
        dim,
    })
}

/// Write an embedding export manifest as pretty JSON.
#[cfg(feature = "artifact-manifest")]
pub fn write_embedding_manifest(dir: &Path, manifest: &EmbeddingManifest) -> io::Result<()> {
    let mut file = BufWriter::new(std::fs::File::create(dir.join("manifest.json"))?);
    serde_json::to_writer_pretty(&mut file, manifest).map_err(io::Error::other)?;
    writeln!(file)?;
    file.flush()
}

/// Read an embedding export manifest from `manifest.json`.
#[cfg(feature = "artifact-manifest")]
pub fn load_embedding_manifest(dir: &Path) -> io::Result<EmbeddingManifest> {
    let file = std::fs::File::open(dir.join("manifest.json"))?;
    serde_json::from_reader(file).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Verify that every artifact named by an embedding manifest still matches its
/// recorded relative path, byte length, and SHA-256 digest.
#[cfg(feature = "artifact-manifest")]
pub fn verify_embedding_manifest(dir: &Path, manifest: &EmbeddingManifest) -> io::Result<()> {
    for artifact in &manifest.artifacts {
        let path = safe_artifact_path(dir, &artifact.path)?;
        let actual_bytes = std::fs::metadata(&path)?.len();
        if actual_bytes != artifact.bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{} byte length mismatch: manifest={}, actual={actual_bytes}",
                    artifact.path, artifact.bytes
                ),
            ));
        }

        let actual_sha256 = sha256_file(&path)?;
        if actual_sha256 != artifact.sha256 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{} sha256 mismatch", artifact.path),
            ));
        }
    }
    Ok(())
}

#[cfg(feature = "artifact-manifest")]
fn safe_artifact_path(dir: &Path, path: &str) -> io::Result<std::path::PathBuf> {
    let rel = Path::new(path);
    if path.is_empty()
        || rel.is_absolute()
        || rel
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("artifact path must be relative and confined: {path}"),
        ));
    }
    Ok(dir.join(rel))
}

#[cfg(feature = "artifact-manifest")]
fn sha256_file(path: &Path) -> io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0_u8; 64 * 1024];
    loop {
        let n = io::Read::read(&mut file, &mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex_lower(&hasher.finalize()))
}

#[cfg(feature = "artifact-manifest")]
fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
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

    #[test]
    fn flatten_empty() {
        let flat = flatten_matrix(&[]);
        assert!(flat.is_empty());
    }

    #[test]
    fn w2v_tsv_preserves_precision() {
        // Verify we don't lose too much precision through text roundtrip.
        let names = vec!["x".to_string()];
        let vecs = vec![vec![std::f32::consts::PI, std::f32::consts::E]];

        let mut buf = Vec::new();
        write_w2v_tsv(&mut buf, &names, &vecs).unwrap();

        let (_, read_vecs) = read_w2v_tsv(buf.as_slice()).unwrap();
        assert!((read_vecs[0][0] - std::f32::consts::PI).abs() < 1e-4);
        assert!((read_vecs[0][1] - std::f32::consts::E).abs() < 1e-4);
    }

    #[test]
    fn read_w2v_bad_header() {
        let bad = b"not_a_number dim\n";
        let result = read_w2v_tsv(bad.as_slice());
        assert!(result.is_err());
    }

    #[test]
    fn read_w2v_dim_mismatch() {
        // Header says dim=3 but data line has 2 values.
        let bad = b"1 3\nalice\t1.0\t2.0\n";
        let result = read_w2v_tsv(bad.as_slice());
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("expected 3"),
            "Error should mention expected dim: {msg}"
        );
    }

    #[test]
    fn write_vocab_tsv_roundtrip() {
        let names = vec![
            "alice".to_string(),
            "bob".to_string(),
            "charlie".to_string(),
        ];
        let mut buf = Vec::new();
        write_vocab_tsv(&mut buf, &names).unwrap();
        let content = String::from_utf8(buf).unwrap();
        assert_eq!(content, "0\talice\n1\tbob\n2\tcharlie\n");
    }

    #[cfg(feature = "artifact-manifest")]
    #[test]
    fn manifest_records_artifact_hashes() {
        let dir = tempfile::tempdir().unwrap();
        let ent_names = vec!["a".to_string(), "b".to_string()];
        let ent_vecs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let rel_names = vec!["r1".to_string()];
        let rel_vecs = vec![vec![0.5, 0.5]];

        export_embeddings(dir.path(), &ent_names, &ent_vecs, &rel_names, &rel_vecs).unwrap();

        let artifacts = vec![
            describe_embedding_artifact(
                dir.path(),
                "entities.tsv",
                "application/vnd.tranz.entity-embeddings+w2v-tsv",
                "w2v-tsv",
                2,
                2,
            )
            .unwrap(),
            describe_embedding_artifact(
                dir.path(),
                "relations.tsv",
                "application/vnd.tranz.relation-embeddings+w2v-tsv",
                "w2v-tsv",
                1,
                2,
            )
            .unwrap(),
        ];

        assert_eq!(
            artifacts[0].bytes,
            std::fs::metadata(dir.path().join("entities.tsv"))
                .unwrap()
                .len()
        );
        assert_eq!(artifacts[0].sha256.len(), 64);

        let manifest = EmbeddingManifest {
            schema: "tranz.embedding-export.v1".to_string(),
            model: "distmult".to_string(),
            score_order: "lower_is_better".to_string(),
            artifacts,
            dataset: ManifestDataset {
                source_kind: "triple_file".to_string(),
                source_path: "toy.tsv".to_string(),
                split: "auto_80_10_10".to_string(),
                entities: 2,
                relations: 1,
                train_triples: 1,
                valid_triples: 0,
                test_triples: 0,
            },
            training: ManifestTraining {
                trainer: "burn-1n-adamw".to_string(),
                backend: "burn-ndarray".to_string(),
                dim: 2,
                init_scale: 0.001,
                lr: 0.001,
                label_smoothing: 0.0,
                n3_reg: 0.0,
                batch_size: 4,
                epochs: 1,
                reciprocals: false,
                final_loss: Some(0.5),
            },
            metrics: None,
        };

        write_embedding_manifest(dir.path(), &manifest).unwrap();
        let parsed = load_embedding_manifest(dir.path()).unwrap();
        verify_embedding_manifest(dir.path(), &parsed).unwrap();

        let json = std::fs::read_to_string(dir.path().join("manifest.json")).unwrap();
        let parsed: EmbeddingManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.artifacts.len(), 2);
        assert_eq!(parsed.artifacts[0].path, "entities.tsv");
        assert_eq!(parsed.dataset.entities, 2);
    }

    #[cfg(feature = "artifact-manifest")]
    #[test]
    fn manifest_verification_rejects_bad_hash_and_escape_path() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("entities.tsv"), b"payload").unwrap();

        let artifact = describe_embedding_artifact(
            dir.path(),
            "entities.tsv",
            "application/vnd.tranz.entity-embeddings+w2v-tsv",
            "w2v-tsv",
            1,
            1,
        )
        .unwrap();

        let manifest = EmbeddingManifest {
            schema: "tranz.embedding-export.v1".to_string(),
            model: "distmult".to_string(),
            score_order: "lower_is_better".to_string(),
            artifacts: vec![artifact.clone()],
            dataset: ManifestDataset {
                source_kind: "triple_file".to_string(),
                source_path: "toy.tsv".to_string(),
                split: "auto_80_10_10".to_string(),
                entities: 1,
                relations: 1,
                train_triples: 1,
                valid_triples: 0,
                test_triples: 0,
            },
            training: ManifestTraining {
                trainer: "burn-1n-adamw".to_string(),
                backend: "burn-ndarray".to_string(),
                dim: 1,
                init_scale: 0.001,
                lr: 0.001,
                label_smoothing: 0.0,
                n3_reg: 0.0,
                batch_size: 4,
                epochs: 1,
                reciprocals: false,
                final_loss: None,
            },
            metrics: None,
        };

        let mut bad_hash = manifest.clone();
        bad_hash.artifacts[0].sha256 = "0".repeat(64);
        let err = verify_embedding_manifest(dir.path(), &bad_hash).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);

        let mut escaped = manifest;
        escaped.artifacts[0].path = "../entities.tsv".to_string();
        let err = verify_embedding_manifest(dir.path(), &escaped).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }
}
