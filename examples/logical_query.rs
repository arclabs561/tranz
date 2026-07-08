//! Compositional query answering over a hand-built TransE model.
//!
//! Usage: cargo run --example logical_query

use tnorms::LogicFamily;
use tranz::{
    query::{answer_query_topk, Query, QueryConfig, ScoreNorm},
    TransE,
};

fn main() {
    let names = ["dog", "cat", "mammal", "animal", "plant"];

    // One-dimensional TransE geometry. The relation vector is +1, so
    // h + is_a is closest to the direct superclass.
    let model = TransE::from_vecs(
        vec![
            vec![0.0], // dog
            vec![0.0], // cat
            vec![1.0], // mammal
            vec![2.0], // animal
            vec![5.0], // plant
        ],
        vec![vec![1.0]], // is_a
        1,
    );

    let config = QueryConfig {
        t_norm_projection: LogicFamily::Product,
        t_norm_intersection: LogicFamily::Godel,
        beam_k: names.len(),
        score_norm: ScoreNorm::Sigmoid,
    };

    let one_hop = Query::anchor(0, 0);
    let two_hop = Query::anchor(0, 0).then(0);
    let common_parent = Query::intersection(vec![Query::anchor(0, 0), Query::anchor(1, 0)]);

    print_top(
        "dog is_a ?",
        &names,
        &answer_query_topk(&model, &one_hop, &config, 3),
    );
    print_top(
        "dog is_a ? is_a ?",
        &names,
        &answer_query_topk(&model, &two_hop, &config, 3),
    );
    print_top(
        "(dog is_a ?) AND (cat is_a ?)",
        &names,
        &answer_query_topk(&model, &common_parent, &config, 3),
    );

    assert_eq!(answer_query_topk(&model, &one_hop, &config, 1)[0].0, 2);
    assert_eq!(answer_query_topk(&model, &two_hop, &config, 1)[0].0, 3);
    assert_eq!(
        answer_query_topk(&model, &common_parent, &config, 1)[0].0,
        2
    );
}

fn print_top(label: &str, names: &[&str], rows: &[(usize, f32)]) {
    println!("{label}");
    for (rank, (entity, score)) in rows.iter().enumerate() {
        println!("  #{} {:<8} {:.3}", rank + 1, names[*entity], score);
    }
}
