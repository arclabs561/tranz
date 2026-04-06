/// Benchmark f32 vs f64 accumulation in dot products.
///
/// Run with: cargo run --release --example bench_f32_vs_f64
use std::time::Instant;

fn dot_f64(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0_f64;
    for i in 0..a.len() {
        sum += a[i] as f64 * b[i] as f64;
    }
    sum as f32
}

fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

fn main() {
    let dim = 200;
    let n = 40943;

    // Random vectors
    let vecs: Vec<Vec<f32>> = (0..n)
        .map(|i| (0..dim).map(|j| ((i * 7 + j * 13) % 1000) as f32 / 1000.0 - 0.5).collect())
        .collect();
    let query: Vec<f32> = (0..dim).map(|j| (j * 17 % 1000) as f32 / 1000.0 - 0.5).collect();

    let iters = 20;

    // f64 accumulation (current)
    let start = Instant::now();
    for _ in 0..iters {
        for v in &vecs {
            std::hint::black_box(dot_f64(&query, v));
        }
    }
    let f64_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // f32 accumulation (proposed)
    let start = Instant::now();
    for _ in 0..iters {
        for v in &vecs {
            std::hint::black_box(dot_f32(&query, v));
        }
    }
    let f32_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // Accuracy check
    let mut max_err = 0.0_f32;
    for v in &vecs {
        let d64 = dot_f64(&query, v);
        let d32 = dot_f32(&query, v);
        max_err = max_err.max((d64 - d32).abs());
    }

    eprintln!("Dot product benchmark (dim={dim}, N={n}):");
    eprintln!("  f64 accum: {f64_ms:.2} ms");
    eprintln!("  f32 accum: {f32_ms:.2} ms");
    eprintln!("  speedup:   {:.2}x", f64_ms / f32_ms);
    eprintln!("  max error: {max_err:.2e}");
}
