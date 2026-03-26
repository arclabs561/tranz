# tranz

Point-embedding knowledge graph completion: TransE, RotatE, ComplEx, DistMult.

Train on any triple file, export embeddings, predict missing links. 1-N scoring with BCE loss for fast convergence.

```toml
[dependencies]
tranz = "0.3.0"
```

Dual-licensed under MIT or Apache-2.0.

## Models

| Model | Scoring function | Space | Reference |
|---|---|---|---|
| TransE | `\|\|h + r - t\|\|` | Real | Bordes et al., 2013 |
| RotatE | `\|\|h * r - t\|\|` | Complex | Sun et al., 2019 |
| ComplEx | `Re(h * r * conj(t))` | Complex | Trouillon et al., 2016 |
| DistMult | `h * r * t` | Real | Yang et al., 2015 |

## Quick start

Install with `cargo install tranz --features candle`.

```sh
# Train with 1-N scoring (recommended)
tranz train --data data/WN18RR/ --model distmult --dim 200 \
    --1n --label-smoothing 0.1 --reciprocals \
    --epochs 100 --lr 0.001 --output embeddings/ --eval

# Train with negative sampling (classic)
tranz train --triples my_graph.tsv --model transe --dim 200 \
    --epochs 500 --gamma 9.0 --alpha 0.5 --output embeddings/ --eval

# Predict from saved embeddings
tranz predict --embeddings embeddings/ --model distmult \
    --head "aspirin" --relation "treats" --k 10
```

## Benchmark: WN18RR

| Model | Mode | Epochs | MRR | H@1 | H@10 |
|-------|------|--------|-----|-----|------|
| DistMult | 1-N + label smoothing | 50 | 0.341 | 0.329 | 0.362 |
| TransE | neg. sampling (SANS) | 100 | 0.156 | 0.002 | 0.421 |

1-N scoring converges much faster than negative sampling. Published DistMult MRR on WN18RR is ~0.43 at convergence.

## Library usage

```rust
use tranz::{TransE, DistMult, Scorer};
use tranz::dataset::load_dataset;
use tranz::eval::evaluate_link_prediction;

// Load dataset
let ds = load_dataset("data/WN18RR".as_ref()).unwrap();
let mut interned = ds.into_interned();
interned.add_reciprocals();

// Create model and query
let model = DistMult::new(interned.num_entities(), interned.num_relations(), 200);
let top10 = model.top_k_tails(0, 0, 10);

// Evaluate
let metrics = evaluate_link_prediction(
    &model, &interned.test, &interned.all_triples(), interned.num_entities(),
);
```

### Generic triple loading

```rust
use tranz::dataset::load_triples;

let ds = load_triples("my_graph.tsv".as_ref()).unwrap();
let ds = ds.split(0.1, 0.1); // 80/10/10
let interned = ds.into_interned();
```

### Embedding export

```rust
use tranz::io::{export_embeddings, flatten_matrix};

// Export to w2v TSV
export_embeddings("output/".as_ref(), &names, &vecs, &rel_names, &rel_vecs).unwrap();

// Flat f32 matrix for FAISS/Qdrant
let flat: Vec<f32> = flatten_matrix(&vecs);
```

## Training (requires `candle` feature)

Two training modes:

**1-N scoring** (recommended): scores all entities per query via matmul + BCE loss. Faster convergence, no negative sampling noise.

**Negative sampling** (classic): samples k negatives per positive with self-adversarial weighting (SANS).

```rust
use tranz::train::{train, TrainConfig, ModelType};

let config = TrainConfig {
    model_type: ModelType::DistMult,
    dim: 200,
    one_to_n: true,
    label_smoothing: 0.1,
    embedding_dropout: 0.1,
    epochs: 100,
    ..TrainConfig::default()
};

let result = train(&triples, num_entities, num_relations, &config, &device).unwrap();
```

## Companion to subsume

[subsume](https://crates.io/crates/subsume) embeds entities as geometric regions (boxes, cones) where containment encodes subsumption. tranz embeds entities as points where distance/similarity encodes relational facts.

- **subsume**: ontology completion, taxonomy expansion, logical query answering
- **tranz**: link prediction, relation extraction, knowledge base completion
