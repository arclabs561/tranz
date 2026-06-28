# tranz examples

Each example answers one question and is runnable as-is. Examples that need a
dataset are **data-gated**: they exit 0 with a message if the data is absent, so
they are safe to compile/run in CI.

Backend selection (Burn examples): `--features burn-ndarray` runs on CPU;
add `burn-wgpu` to run on Metal/Vulkan (`--features "burn-ndarray,burn-wgpu"`).
The feature names follow Burn's own convention (named after the backend, not the
device class).

## Training (Burn backend)

### `wn18rr_kge_burn` — do the four Burn KGE models actually learn on real data?

Trains TransE, RotatE, ComplEx, and DistMult on full WN18RR via `train_kge` and
reports filtered MRR/Hits.

Cost: full WN18RR (40943 entities, 173670 triples); ~30-60s per model on Metal,
minutes per model on CPU. Shrink with `TRAIN_CAP=20000 DIM=32 EPOCHS=5`.

```bash
cargo run --release --features "burn-ndarray,burn-wgpu" --example wn18rr_kge_burn
```
```text
model      train_s   loss[0]->loss[-1]      test_MRR   H@10
DistMult     39.1    10.386 ->   3.979        0.5706   0.6459
TransE       32.5     9.954 ->   5.013        0.5488   0.6407
ComplEx      41.8    10.317 ->   4.077        0.5560   0.6390
RotatE       57.1     9.596 ->   4.774        0.5532   0.6307
```
MRR here is sampled-eval (200 candidates per test triple); it reads higher than
full-entity-ranking MRR, so use it for relative comparison, not as a benchmark.

### `wn18rr_vicinity` — can trained point embeddings serve nearest-neighbour queries?

Trains DistMult, then indexes the entity vectors in a [vicinity](https://crates.io/crates/vicinity)
HNSW and runs cosine nearest-neighbour queries. (tranz produces *point*
embeddings, so a vector index is the natural serving layer.)

```bash
cargo run --release --features "burn-ndarray,burn-wgpu" --example wn18rr_vicinity
```
```text
indexed 40943 entity vectors in vicinity HNSW
nearest entities by cosine similarity (id: [neighbours]):
      0: [13635(0.683), 24364(0.662), 14327(0.657), 5495(0.653), 29144(0.645)]
   1000: [222(0.695), 28193(0.671), 16838(0.609), 17370(0.590), 8282(0.575)]
```

### `bench_wgpu` — how much faster is Metal than CPU?

Times a Burn training epoch on wgpu (Metal) vs ndarray (CPU).

```bash
cargo run --release --features "burn-ndarray,burn-wgpu" --example bench_wgpu
```

## Utility (no ML backend needed)

| Example | Question | Run |
|---|---|---|
| `score` | Score triples with a CPU scorer (no training backend needed) | `cargo run --release --example score` |
| `bench_scoring` | Scoring hot-path throughput | `cargo run --release --example bench_scoring` |
| `bench_f32_vs_f64` | Does f64 accumulation matter for dot products? | `cargo run --release --example bench_f32_vs_f64` |

## Datasets

`data/WN18RR/` and `data/FB15k-237/` are not tracked. The data-gated examples
no-op without them.
