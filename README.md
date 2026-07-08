# tranz

Knowledge graph embedding models.

```toml
[dependencies]
tranz = "0.8.0"
```

Dual-licensed under MIT or Apache-2.0.

`tranz` trains and scores TransE, RotatE, ComplEx, and DistMult models. It
includes dataset loading, filtered link-prediction evaluation, embedding export,
and Burn-backed CPU/GPU training.

Install with `cargo install tranz --features burn-ndarray` (CPU) or
`--features burn-wgpu` (GPU/Metal). Training uses Burn's 1-N (1vsAll)
cross-entropy with AdamW.

```sh
# Train (ComplEx is the strongest recipe with label smoothing + reciprocals)
tranz train --data data/WN18RR/ --model complex --dim 200 \
    --label-smoothing 0.1 --reciprocals \
    --epochs 100 --lr 0.001 --output embeddings/ --eval

# Train from a single triple file (auto-split 80/10/10)
tranz train --triples my_graph.tsv --model transe --dim 200 \
    --epochs 500 --output embeddings/ --eval

# Predict from saved embeddings
tranz predict --embeddings embeddings/ --model distmult \
    --head "aspirin" --relation "treats" --k 10
```

## Benchmark: WN18RR

Trained with the Burn 1-N (1vsAll) cross-entropy trainer (AdamW), full filtered
evaluation on the test split.

| Model | Config | Dim | Epochs | MRR | H@1 | H@10 |
|-------|--------|-----|--------|-----|-----|------|
| ComplEx | 1-N + label smoothing + reciprocals | 100 | 50 | **0.424** | 0.398 | 0.476 |

Published ComplEx MRR on WN18RR is 0.475 (Lacroix et al. 2018, with Adagrad + N3
regularization). The reproduced command below leaves N3 off; pass
`--n3-reg <coefficient>` to enable the Burn trainer's weighted nuclear-3 penalty
for ComplEx or DistMult.

Reproduce (about 20 min on Metal: dim 100, 50 epochs over full WN18RR):

```sh
cargo run --release --features "burn-ndarray,burn-wgpu" --bin tranz -- \
    train --data data/WN18RR/ --model complex --dim 100 \
    --label-smoothing 0.1 --reciprocals --epochs 50 --lr 0.001 --eval
```

The other three models train end to end on WN18RR too; see the
`wn18rr_kge_burn` example for a four-model relative comparison.

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

`tranz train` writes `entities.tsv`, `relations.tsv`, and `manifest.json`.
The manifest records model family, training config, split sizes, SHA-256
digests, byte sizes, and evaluation metrics when `--eval` is used. With the
`artifact-manifest` feature, `load_embedding_manifest` and
`verify_embedding_manifest` read the manifest back and check the exported files.

```rust
use tranz::io::{
    export_embeddings, flatten_matrix, load_embedding_manifest, verify_embedding_manifest,
};

// Export to w2v TSV
export_embeddings("output/".as_ref(), &names, &vecs, &rel_names, &rel_vecs).unwrap();

let manifest = load_embedding_manifest("output/".as_ref()).unwrap();
verify_embedding_manifest("output/".as_ref(), &manifest).unwrap();

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

## Models

Each model scores a triple (head, relation, tail) differently:

| Model | Scoring function | Intuition | Reference |
|---|---|---|---|
| TransE | $\lVert \mathbf{h} + \mathbf{r} - \mathbf{t} \rVert$ | Translation: tail = head + relation | Bordes et al., 2013 |
| RotatE | $\lVert \mathbf{h} \circ \mathbf{r} - \mathbf{t} \rVert$ | Rotation in complex plane | Sun et al., 2019 |
| ComplEx | $\text{Re}(\langle \mathbf{h}, \mathbf{r}, \bar{\mathbf{t}} \rangle)$ | Asymmetric via complex conjugate | Trouillon et al., 2016 |
| DistMult | $\langle \mathbf{h}, \mathbf{r}, \mathbf{t} \rangle$ | Element-wise product, symmetric | Yang et al., 2015 |

$\mathbf{h}, \mathbf{r}, \mathbf{t}$ are learned embedding vectors for head,
relation, and tail.

## Training

Training runs on the Burn backend, selected by feature:

| Feature | Backend | GPU | Best for |
|---------|---------|-----|----------|
| `burn-ndarray` | Burn + ndarray | -- | CPU training, all 4 models |
| `burn-wgpu` | Burn + WGPU | Metal/Vulkan | macOS/GPU training, all 4 models |

All four models train with 1-N (1vsAll) scoring: every entity is scored per
query via matmul + softmax cross-entropy, optimized with AdamW. Label smoothing
is optional.

```rust
use tranz::burn_train::{train_kge, BurnModelType, BurnTrainConfig};

type B = burn::backend::Autodiff<burn_ndarray::NdArray>;
let device = burn_ndarray::NdArrayDevice::Cpu;

let config = BurnTrainConfig {
    dim: 200,
    label_smoothing: 0.1,
    epochs: 100,
    ..BurnTrainConfig::default()
};

let result = train_kge::<B>(
    &triples,
    num_entities,
    num_relations,
    BurnModelType::DistMult,
    &config,
    &device,
);
```

## Examples

See [`examples/README.md`](examples/README.md) for the full gallery, where each
example states the question it answers, the run command, and real sample output.
Highlights:

- `wn18rr_kge_burn` trains all four models on real WN18RR with the Burn backend (Metal-accelerated) and reports MRR/Hits, the real-data check that the Burn trainers learn.
- `wn18rr_vicinity` trains point embeddings and serves nearest-neighbour queries through a [vicinity](https://crates.io/crates/vicinity) HNSW index.
- `score` is the smallest way to use a trained model; `bench_wgpu` times Metal vs CPU; `bench_scoring` and `bench_f32_vs_f64` measure the scoring hot path.

## Companion to subsume

[subsume](https://crates.io/crates/subsume) embeds entities as geometric regions (boxes, cones) where containment encodes subsumption. tranz embeds entities as points where distance/similarity encodes relational facts.

- **subsume**: ontology completion, taxonomy expansion, logical query answering
- **tranz**: link prediction, relation extraction, knowledge base completion
