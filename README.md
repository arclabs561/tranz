# tranz

Point-embedding knowledge graph completion: TransE, RotatE, ComplEx, DistMult.

Entities are points in vector space. Relations are transformations (translation, rotation, diagonal scaling). Train on any triple file, export embeddings for downstream use.

```toml
[dependencies]
tranz = "0.2.0"
```

Dual-licensed under MIT or Apache-2.0.

## Models

| Model | Scoring function | Space | Reference |
|---|---|---|---|
| TransE | `\|\|h + r - t\|\|` | Real | Bordes et al., 2013 |
| RotatE | `\|\|h * r - t\|\|` | Complex | Sun et al., 2019 |
| ComplEx | `Re(h * r * conj(t))` | Complex | Trouillon et al., 2016 |
| DistMult | `h * r * t` | Real | Yang et al., 2015 |

## CLI

Install with `cargo install tranz --features candle`.

```sh
# Train on any TSV/CSV triple file
tranz train --triples data.tsv --model rotate --dim 200 --epochs 500 --output embeddings/

# Train on WN18RR-format directory
tranz train --data data/WN18RR/ --model transe --dim 200 --epochs 500 --output embeddings/ --eval

# Output: embeddings/entities.tsv, embeddings/relations.tsv (w2v format)
```

## Library usage

```rust
use tranz::{TransE, Scorer};
use tranz::dataset::load_dataset;
use tranz::eval::evaluate_link_prediction;

// Load WN18RR-format dataset
let ds = load_dataset("data/WN18RR".as_ref()).unwrap();
let mut interned = ds.into_interned();
interned.add_reciprocals(); // optional, improves all models

let model = TransE::new(
    interned.num_entities(),
    interned.num_relations(),
    200,
);

// Batch scoring: top-10 tail predictions
let top10 = model.top_k_tails(0, 0, 10);

// Filtered evaluation
let metrics = evaluate_link_prediction(
    &model,
    &interned.test,
    &interned.all_triples(),
    interned.num_entities(),
);
```

### Generic triple loading

```rust
use tranz::dataset::load_triples;

// Load any TSV or CSV file: head<TAB>relation<TAB>tail
let ds = load_triples("my_graph.tsv".as_ref()).unwrap();
let ds = ds.split(0.1, 0.1); // 80/10/10 train/valid/test
let interned = ds.into_interned();
```

### Embedding export

```rust
use tranz::io::export_embeddings;

// After training, export to w2v TSV format
export_embeddings(
    "output/".as_ref(),
    &interned.id_to_entity,
    &model.entities().to_vec(), // or entity_vecs() from TrainableModel
    &interned.id_to_relation,
    &model.relations().to_vec(),
).unwrap();
// -> output/entities.tsv, output/relations.tsv
```

## Training (requires `candle` feature)

```rust
use tranz::train::{train, TrainConfig, ModelType};

let config = TrainConfig {
    model_type: ModelType::RotatE,
    dim: 200,
    gamma: 12.0,
    epochs: 500,
    normalize_entities: true, // L2 normalization (TransE paper)
    ..TrainConfig::default()
};

let result = train(&interned.train, num_entities, num_relations, &config, &device).unwrap();
let scorer = result.model.to_rotate().unwrap();
```

## Companion to subsume

[subsume](https://crates.io/crates/subsume) embeds entities as geometric regions (boxes, cones) where containment encodes subsumption. tranz embeds entities as points where distance/similarity encodes relational facts. Different geometric paradigms for different tasks:

- **subsume**: ontology completion, taxonomy expansion, logical query answering
- **tranz**: link prediction, relation extraction, knowledge base completion
