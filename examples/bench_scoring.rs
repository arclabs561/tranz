//! Quick benchmark of scoring hot paths.
//!
//! Run with: `cargo run --release --example bench_scoring`
#![allow(missing_docs)]
use std::time::Instant;
use tranz::{ComplEx, DistMult, RotatE, Scorer, TransE};

fn bench<S: Scorer>(name: &str, model: &S, iters: usize) {
    let n = model.num_entities();

    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(model.score_all_tails(0, 0));
    }
    let tail_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(model.score_all_heads(0, 0));
    }
    let head_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(model.top_k_tails(0, 0, 10));
    }
    let top_tail_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(model.top_k_heads(0, 0, 10));
    }
    let top_head_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    eprintln!(
        "{name:<12} N={n:<6} tail={tail_ms:.2}ms  head={head_ms:.2}ms  top_tail={top_tail_ms:.2}ms  top_head={top_head_ms:.2}ms  ratio={:.2}x",
        head_ms / tail_ms
    );
}

fn main() {
    let n = 40943; // WN18RR entity count
    let r = 11;
    let dim = 100;
    let iters = 10;

    eprintln!("Scoring benchmark (dim={dim}, {iters} iterations each)\n");

    let complex = ComplEx::new(n, r, dim);
    bench("ComplEx", &complex, iters);

    let distmult = DistMult::new(n, r, dim);
    bench("DistMult", &distmult, iters);

    let transe = TransE::new(n, r, dim);
    bench("TransE", &transe, iters);

    let rotate = RotatE::new(n, r, dim, 12.0);
    bench("RotatE", &rotate, iters);

    // Eval cost estimate
    let test_triples = 6268;
    let tail_ms = {
        let start = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(complex.score_all_tails(0, 0));
        }
        start.elapsed().as_secs_f64() * 1000.0 / iters as f64
    };
    let head_ms = {
        let start = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(complex.score_all_heads(0, 0));
        }
        start.elapsed().as_secs_f64() * 1000.0 / iters as f64
    };
    let serial_s = test_triples as f64 * (tail_ms + head_ms) / 1000.0;
    eprintln!("\nComplEx eval estimate ({test_triples} triples):");
    eprintln!("  serial: {serial_s:.0}s");
    eprintln!("  rayon 8t: {:.0}s", serial_s / 8.0);
}
