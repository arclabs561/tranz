# tranz

Point-embedding knowledge graph completion: TransE, RotatE, ComplEx, DistMult.

Entities are points in vector space. Relations are transformations (translation, rotation, diagonal scaling). Scoring functions rank candidate triples by distance or similarity.

```toml
[dependencies]
tranz = "0.1.0"
```

Dual-licensed under MIT or Apache-2.0.

## Models

| Model | Scoring function | Space | Reference |
|---|---|---|---|
| TransE | `\|\|h + r - t\|\|` | Real | Bordes et al., 2013 |
| RotatE | `\|\|h * r - t\|\|` | Complex | Sun et al., 2019 |
| ComplEx | `Re(h * r * conj(t))` | Complex | Trouillon et al., 2016 |
| DistMult | `h * r * t` | Real | Yang et al., 2015 |

## Usage

```rust
use tranz::{TransE, Scorer};
use tranz::dataset::load_dataset;
use tranz::eval::evaluate_link_prediction;

// Load WN18RR-format dataset (train.txt, valid.txt, test.txt)
let ds = load_dataset("data/WN18RR".as_ref()).unwrap();
let mut interned = ds.into_interned();

// Optional: add reciprocal relations (improves all models)
interned.add_reciprocals();

// Create model and evaluate
let model = TransE::new(
    interned.num_entities(),
    interned.num_relations(),
    200,
);
let metrics = evaluate_link_prediction(
    &model,
    &interned.test,
    &interned.all_triples(),
    interned.num_entities(),
);
println!("MRR: {:.4}, Hits@10: {:.4}", metrics.mrr, metrics.hits_at_10);
```

## Companion to subsume

[subsume](https://crates.io/crates/subsume) embeds entities as geometric regions (boxes, cones) where containment encodes subsumption. tranz embeds entities as points where distance/similarity encodes relational facts. Different geometric paradigms for different tasks:

- **subsume**: ontology completion, taxonomy expansion, logical query answering
- **tranz**: link prediction, relation extraction, knowledge base completion
