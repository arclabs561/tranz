# tranz

Point-embedding knowledge graph models: TransE, RotatE, ComplEx,
and DistMult. GPU training via candle.

```toml
[dependencies]
tranz = "0.5"
```

Dual-licensed under MIT or Apache-2.0.

For context on how point embeddings relate to region-based approaches, see [Why Regions, Not Points](https://attobop.net/posts/region-embeddings/).

## Models

Each model scores a triple (head, relation, tail) differently:

| Model | Scoring function | Intuition | Reference |
|---|---|---|---|
| TransE | $\lVert \mathbf{h} + \mathbf{r} - \mathbf{t} \rVert$ | Translation: tail = head + relation | Bordes et al., 2013 |
| RotatE | $\lVert \mathbf{h} \circ \mathbf{r} - \mathbf{t} \rVert$ | Rotation in complex plane | Sun et al., 2019 |
| ComplEx | $\text{Re}(\langle \mathbf{h}, \mathbf{r}, \bar{\mathbf{t}} \rangle)$ | Asymmetric via complex conjugate | Trouillon et al., 2016 |
| DistMult | $\langle \mathbf{h}, \mathbf{r}, \mathbf{t} \rangle$ | Element-wise product, symmetric | Yang et al., 2015 |

$\mathbf{h}, \mathbf{r}, \mathbf{t}$ are learned embedding vectors for head, relation, and tail.
$\lVert \cdot \rVert$ is the L2 norm, $\circ$ is element-wise product, $\langle \cdot \rangle$ is the trilinear dot product, $\bar{\mathbf{t}}$ is the complex conjugate.

## Quick start

Install with `cargo install tranz --features candle`.

```sh
# Train with 1-N scoring (recommended)
tranz train --data data/WN18RR/ --model complex --dim 200 \
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

| Model | Config | Dim | Epochs | MRR | H@1 | H@10 |
|-------|--------|-----|--------|-----|-----|------|
| ComplEx | Adagrad + N3 + reciprocals | 100 | 100 | **0.438** | 0.400 | 0.512 |
| ComplEx | Adam + 1-N + reciprocals | 100 | 50 | 0.433 | 0.406 | 0.487 |
| DistMult | Adam + 1-N | 100 | 50 | 0.341 | 0.329 | 0.362 |

Published ComplEx MRR on WN18RR is 0.475 (Lacroix et al. 2018).
tranz reaches 92% of published with the same recipe (Adagrad, N3, reciprocals).

Commands to reproduce:

```sh
# Adagrad + N3 (best)
tranz train --data data/WN18RR/ --model complex --dim 100 \
    --1n --reciprocals --optimizer adagrad --init-scale 1e-3 \
    --n3 0.1 --lr 0.1 --epochs 100 --eval

# Adam + 1-N
tranz train --data data/WN18RR/ --model complex --dim 100 \
    --1n --label-smoothing 0.1 --reciprocals \
    --epochs 50 --lr 0.001 --eval
```

## Library usage

```rust
use tranz::{TransE, DistMult, Scorer};
use tranz::dataset::{load_dataset, FilterIndex, InternedDatasetExt};
use tranz::eval::evaluate_link_prediction;

// Load dataset (types from lattix::kge)
let ds = load_dataset("data/WN18RR".as_ref()).unwrap();
let mut interned = ds.into_interned();
interned.add_reciprocals();

// Create model and query
let model = DistMult::new(interned.num_entities(), interned.num_relations(), 200);
let top10 = model.top_k_tails(0, 0, 10);

// Evaluate (filtered link prediction)
let filter = FilterIndex::from_dataset(&interned);
let metrics = evaluate_link_prediction(&model, &interned.test, &filter, interned.num_entities());
```

### Generic triple loading

```rust
use tranz::dataset::{Dataset, DatasetExt};

let ds = Dataset::load_flexible("my_graph.tsv".as_ref()).unwrap();
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

### Multi-hop query answering

Answers conjunctive, disjunctive, and negation queries by decomposing them
into atomic link prediction calls composed with t-norm fuzzy logic
(CQD-Beam, Arakelyan et al. 2021). No complex-query training needed.

```rust
use tranz::query::{Query, QueryConfig, answer_query_topk};
use tranz::DistMult;

let model = DistMult::new(1000, 50, 200);

// 2-hop chain: entity 0 -rel 0-> V -rel 1-> ?
let q = Query::anchor(0, 0).then(1);

// Intersection: (0 -r0-> ?) AND (1 -r1-> ?)
let q = Query::intersection(vec![Query::anchor(0, 0), Query::anchor(1, 1)]);

// Intersect-then-project (pi): (0 -r0-> V AND 1 -r1-> V) -r2-> ?
let q = Query::intersection(vec![Query::anchor(0, 0), Query::anchor(1, 1)]).then(2);

let top10 = answer_query_topk(&model, &q, &QueryConfig::default(), 10);
```

### Ensemble scoring

Average scores from multiple models (snapshots, different seeds).

```rust
use tranz::{DistMult, EnsembledScorer, Scorer};

let models: Vec<Box<dyn Scorer>> = vec![
    Box::new(DistMult::new(100, 10, 50)),
    Box::new(DistMult::new(100, 10, 50)),
];
let ensemble = EnsembledScorer::new(models);
let top5 = ensemble.top_k_tails(0, 0, 5);
```

## Training

Two backends available:

| Feature | Backend | GPU | Best for |
|---------|---------|-----|----------|
| `candle` | Candle | CUDA | Production training, all 4 models |
| `burn-cpu` | Burn + ndarray | -- | CPU training, ComplEx |
| `burn-gpu` | Burn + WGPU | Metal/Vulkan | macOS GPU training, ComplEx |

1-N scoring (all entities per query via matmul + softmax CE) is recommended. Negative sampling with SANS weighting is also supported (candle backend only).

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
