//! Numeric argument errors and initialized embedding exports through the real CLI.

#![cfg(feature = "burn-ndarray")]

use std::process::{Command, Output};

// Exercise the shipped CLI, including the ordering of parsing and input loading.
const OPTIONS: &[(&str, &str, bool)] = &[
    ("predict", "--k", true),
    ("train", "--dim", true),
    ("train", "--epochs", true),
    ("train", "--batch-size", true),
    ("train", "--init-scale", false),
    ("train", "--lr", false),
    ("train", "--label-smoothing", false),
    ("train", "--n3-reg", false),
    ("train-temporal", "--dim", true),
    ("train-temporal", "--epochs", true),
    ("train-temporal", "--batch-size", true),
    ("train-temporal", "--init-scale", false),
    ("train-temporal", "--lr", false),
    ("train-temporal", "--label-smoothing", false),
    ("train-temporal", "--time-smooth", false),
    ("train-temporal", "--n3-reg", false),
];

fn invoke(command: &str, option: &str, value: Option<&str>) -> Output {
    let sandbox = tempfile::tempdir().unwrap();
    let input_option = match command {
        "predict" => "--embeddings",
        "train" => "--triples",
        "train-temporal" => "--data",
        _ => unreachable!(),
    };
    let mut process = Command::new(env!("CARGO_BIN_EXE_tranz"));
    process
        .current_dir(sandbox.path())
        .args([command, input_option])
        .arg(sandbox.path().join("missing-input"))
        .arg(option);
    if let Some(value) = value {
        process.arg(value);
    }
    process.output().unwrap()
}

fn check_rejections(value: Option<&str>, integers_only: bool) {
    let mut failures = Vec::new();
    for &(command, option, integer) in OPTIONS {
        if integers_only && !integer {
            continue;
        }
        let output = invoke(command, option, value);
        let stderr = String::from_utf8_lossy(&output.stderr);
        if output.status.code() != Some(1)
            || !stderr.contains(option)
            || stderr.contains("panicked")
            || stderr.contains("Loading")
            || stderr.contains("Failed")
            || stderr.contains("Training")
            || stderr.contains("Entities:")
            || !output.stdout.is_empty()
        {
            failures.push(format!(
                "{command} {option} {value:?}: status {:?}, stderr {stderr:?}",
                output.status.code()
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn malformed_numbers_are_argument_errors_before_input_loading() {
    check_rejections(Some("invalid"), false);
}

#[test]
fn missing_numbers_are_argument_errors_before_input_loading() {
    check_rejections(None, false);
}

#[test]
fn negative_integers_are_argument_errors_before_input_loading() {
    check_rejections(Some("-1"), true);
}

#[test]
fn overflowing_integers_are_argument_errors_before_input_loading() {
    // Greater than any usize on supported 32-bit or 64-bit targets.
    check_rejections(Some("18446744073709551616"), true);
}

#[test]
fn valid_numbers_reach_the_missing_input_error_without_training() {
    for &(command, option, integer) in OPTIONS {
        // These floating-point tokens are accepted by Rust's parser. Their
        // training-domain policy is deliberately outside argument parsing.
        let values: &[&str] = if integer {
            &["1"]
        } else {
            &["0.1", "-0.1", "NaN", "inf", "-inf"]
        };
        for &value in values {
            let output = invoke(command, option, Some(value));
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert_eq!(output.status.code(), Some(1), "{stderr}");
            assert!(stderr.contains("Failed"), "{command} {option}: {stderr}");
            assert!(!stderr.contains("panicked"), "{stderr}");
            assert!(!stderr.contains("Training"), "{stderr}");
            assert!(!stderr.contains("Entities:"), "{stderr}");
            assert!(output.stdout.is_empty());
        }
    }
}

fn assert_embeddings(path: &std::path::Path, names: &[&str], width: usize) {
    let contents = std::fs::read_to_string(path).unwrap();
    let mut lines = contents.lines();
    assert_eq!(lines.next().unwrap(), format!("{} {width}", names.len()));
    let mut actual_names = Vec::new();
    for line in lines {
        let mut fields = line.split('\t');
        actual_names.push(fields.next().unwrap());
        let coordinates: Vec<f32> = fields.map(|field| field.parse().unwrap()).collect();
        assert_eq!(coordinates.len(), width);
        assert!(coordinates.iter().all(|value| value.is_finite()));
    }
    assert_eq!(actual_names, names);
}

fn check_export(temporal: bool, epochs: usize) {
    let sandbox = tempfile::tempdir().unwrap();
    let output_dir = sandbox.path().join("embeddings");
    let mut process = Command::new(env!("CARGO_BIN_EXE_tranz"));
    process.current_dir(sandbox.path());
    if temporal {
        let dataset = sandbox.path().join("quads");
        std::fs::create_dir(&dataset).unwrap();
        // Reverse chronological input checks the exported timestamp order.
        std::fs::write(
            dataset.join("train.txt"),
            "a\tr\tb\t2026-02-01\na\tr\tb\t2026-01-01\n",
        )
        .unwrap();
        std::fs::write(dataset.join("valid.txt"), "").unwrap();
        std::fs::write(dataset.join("test.txt"), "").unwrap();
        process.args(["train-temporal", "--data"]).arg(dataset);
    } else {
        let triples = sandbox.path().join("triples.tsv");
        std::fs::write(&triples, "a\tr\tb\n").unwrap();
        process
            .args(["train", "--model", "distmult", "--triples"])
            .arg(triples);
    }
    let output = process
        .args(["--dim", "2", "--batch-size", "1", "--epochs"])
        .arg(epochs.to_string())
        .arg("--output")
        .arg(&output_dir)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");
    if epochs == 0 {
        assert!(stderr.contains("Initialization complete"), "{stderr}");
        assert!(stderr.contains("zero epochs, no training loss"), "{stderr}");
        assert!(!stderr.contains("final loss:"), "{stderr}");
    } else {
        assert!(stderr.contains("Training complete"), "{stderr}");
        assert!(stderr.contains("final loss:"), "{stderr}");
    }
    // TComplEx stores both real and imaginary coordinates for each dimension.
    let width = if temporal { 4 } else { 2 };
    assert_embeddings(&output_dir.join("entities.tsv"), &["a", "b"], width);
    assert_embeddings(&output_dir.join("relations.tsv"), &["r"], width);
    if temporal {
        assert_embeddings(
            &output_dir.join("times.tsv"),
            &["2026-01-01", "2026-02-01"],
            width,
        );
    } else {
        let manifest: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(output_dir.join("manifest.json")).unwrap(),
        )
        .unwrap();
        let training = manifest.get("training").unwrap();
        assert_eq!(training["epochs"].as_u64(), Some(epochs as u64));
        let final_loss = training.get("final_loss").unwrap();
        if epochs == 0 {
            assert_eq!(final_loss, &serde_json::Value::Null);
        } else {
            assert!(final_loss.as_f64().unwrap().is_finite());
        }
    }
}

#[test]
fn zero_epochs_export_initialized_embeddings() {
    check_export(false, 0);
}

#[test]
fn zero_epochs_export_initialized_temporal_embeddings() {
    check_export(true, 0);
}

#[test]
fn positive_epochs_still_export_embeddings_and_loss() {
    check_export(false, 1);
}

#[test]
fn positive_epochs_still_export_temporal_embeddings_and_loss() {
    check_export(true, 1);
}

#[test]
fn temporal_eval_protocol_matches_exported_embeddings() {
    use tranz::temporal::{load_temporal_dataset, TComplEx, TemporalScorer};
    for reciprocals in [false, true] {
        let sandbox = tempfile::tempdir().unwrap();
        let data = sandbox.path().join("data");
        let exported = sandbox.path().join("exported");
        std::fs::create_dir(&data).unwrap();
        // Several distinct queries exercise direction-specific relation vectors;
        // reuse no training steps or assumed random initialization values.
        std::fs::write(
            data.join("train.txt"),
            "a\tr\tb\t0\nb\ts\tc\t1\nc\tr\td\t0\nd\ts\te\t1\ne\tr\tf\t0\nf\ts\ta\t1\n",
        )
        .unwrap();
        std::fs::write(data.join("valid.txt"), "").unwrap();
        std::fs::write(
            data.join("test.txt"),
            "a\tr\tc\t0\nb\ts\td\t1\nc\tr\te\t1\nd\ts\tf\t0\ne\tr\ta\t0\nf\ts\tb\t1\na\ts\td\t1\nb\tr\te\t0\n",
        )
        .unwrap();
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_tranz"));
        cmd.args(["train-temporal", "--data"])
            .arg(&data)
            .args([
                "--epochs",
                "0",
                "--dim",
                "4",
                "--init-scale",
                "0.5",
                "--eval",
                "--output",
            ])
            .arg(&exported);
        if reciprocals {
            cmd.arg("--reciprocals");
        }
        let output = cmd.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let mut ds = load_temporal_dataset(&data).unwrap();
        let original_relations = ds.num_relations();
        if reciprocals {
            ds.add_reciprocals();
        }
        let read = |name: &str| {
            tranz::io::read_w2v_tsv(std::fs::File::open(exported.join(name)).unwrap())
                .unwrap()
                .1
        };
        let model = TComplEx::from_vecs(
            read("entities.tsv"),
            read("relations.tsv"),
            read("times.tsv"),
            4,
        );
        let known = ds.all_quads();
        let mut heads = Vec::new();
        let mut tails = Vec::new();
        for q in &ds.test {
            let target = model.score(q.head, q.relation, q.tail, q.time);
            let rank = |head: bool| {
                let mut better = 0;
                let mut tied = 0;
                for entity in 0..ds.num_entities() {
                    let (h, t) = if head {
                        (entity, q.tail)
                    } else {
                        (q.head, entity)
                    };
                    if (h, t) == (q.head, q.tail)
                        || known.iter().any(|k| {
                            k.head == h
                                && k.tail == t
                                && k.relation == q.relation
                                && k.time == q.time
                        })
                    {
                        continue;
                    }
                    let score = model.score(h, q.relation, t, q.time);
                    better += usize::from(score < target);
                    tied += usize::from(score == target);
                }
                // Independent optimistic/pessimistic endpoints, then midpoint.
                ((1 + better) as f64 + (1 + better + tied) as f64) / 2.0
            };
            if reciprocals && q.relation >= original_relations {
                heads.push(rank(false));
            } else {
                tails.push(rank(false));
            }
            if !reciprocals {
                heads.push(rank(true));
            }
        }
        let rr = |r: &[f64]| r.iter().map(|r| 1.0 / r).sum::<f64>() / r.len() as f64;
        let all: Vec<f64> = heads.iter().chain(&tails).copied().collect();
        let stdout = String::from_utf8(output.stdout).unwrap();
        for (label, value, precision) in [
            ("MRR:", rr(&all), 4),
            ("head:", rr(&heads), 4),
            ("tail:", rr(&tails), 4),
            ("MR:", all.iter().sum::<f64>() / all.len() as f64, 1),
            (
                "Hits@1:",
                all.iter().filter(|&&r| r <= 1.0).count() as f64 / all.len() as f64,
                4,
            ),
            (
                "Hits@3:",
                all.iter().filter(|&&r| r <= 3.0).count() as f64 / all.len() as f64,
                4,
            ),
            (
                "Hits@10:",
                all.iter().filter(|&&r| r <= 10.0).count() as f64 / all.len() as f64,
                4,
            ),
        ] {
            let actual = stdout
                .lines()
                .find_map(|line| line.trim().strip_prefix(label))
                .unwrap()
                .trim();
            assert_eq!(
                actual,
                format!("{:.*}", precision, value as f32),
                "reciprocals={reciprocals}, {label}; {stdout}"
            );
        }
    }
}

#[test]
fn static_reciprocal_name_collision_exports_and_evaluates() {
    let sandbox = tempfile::tempdir().unwrap();
    let data = sandbox.path().join("data");
    let exported = sandbox.path().join("exported");
    std::fs::create_dir(&data).unwrap();
    std::fs::write(data.join("train.txt"), "a\tr\tb\nb\tr_inv\tc\n").unwrap();
    std::fs::write(data.join("valid.txt"), "").unwrap();
    std::fs::write(data.join("test.txt"), "c\tr_inv\ta\n").unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_tranz"))
        .args(["train", "--data"])
        .arg(&data)
        .args([
            "--model",
            "distmult",
            "--dim",
            "2",
            "--batch-size",
            "2",
            "--epochs",
            "0",
            "--reciprocals",
            "--eval",
            "--output",
        ])
        .arg(&exported)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");
    assert!(stderr.contains("Relations: 4"), "{stderr}");
    assert!(stderr.contains("zero epochs, no training loss"), "{stderr}");
    assert!(stderr.contains("Evaluating on test set"), "{stderr}");
    assert_embeddings(
        &exported.join("relations.tsv"),
        &["r", "r_inv", "r_inv_2", "r_inv_inv"],
        2,
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    for label in [
        "MRR:", "head:", "tail:", "MR:", "Hits@1:", "Hits@3:", "Hits@10:",
    ] {
        let value = stdout
            .lines()
            .find_map(|line| line.trim().strip_prefix(label))
            .unwrap_or_else(|| panic!("missing {label} in CLI output: {stdout}"))
            .trim()
            .parse::<f32>()
            .unwrap();
        assert!(value.is_finite(), "{label} was {value}: {stdout}");
    }
}
