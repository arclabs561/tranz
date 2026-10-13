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
