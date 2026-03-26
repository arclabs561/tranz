//! Minimal scoring example -- no candle dependency needed.
//!
//! Usage: cargo run --example score

use tranz::{DistMult, Scorer};

fn main() {
    // Create a small model.
    let model = DistMult::new(100, 10, 50);

    // Score a single triple.
    let score = model.score(0, 0, 1);
    println!("score(0, 0, 1) = {score:.4}");

    // Top-k tail predictions.
    let top5 = model.top_k_tails(0, 0, 5);
    println!("\nTop-5 tails for (entity 0, relation 0, ?):");
    for (rank, (entity, score)) in top5.iter().enumerate() {
        println!("  #{}: entity {entity}, score {score:.4}", rank + 1);
    }

    // Top-k head predictions.
    let top5 = model.top_k_heads(0, 0, 5);
    println!("\nTop-5 heads for (?, relation 0, entity 0):");
    for (rank, (entity, score)) in top5.iter().enumerate() {
        println!("  #{}: entity {entity}, score {score:.4}", rank + 1);
    }
}
