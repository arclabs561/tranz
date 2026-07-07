# Benchmarks

Reproducible WN18RR runs for the current Burn trainer.

## Setup

```sh
# Install tranz for CPU training
cargo install tranz --features burn-ndarray

# Install tranz with GPU support
cargo install tranz --features "burn-ndarray,burn-wgpu"

# Or build from source with GPU support
git clone https://github.com/arclabs561/tranz
cd tranz
cargo build --release --features "burn-ndarray,burn-wgpu" --bin tranz

# Download WN18RR
mkdir -p data/WN18RR && cd data/WN18RR
curl -sLO https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/WN18RR/original/train.txt
curl -sLO https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/WN18RR/original/valid.txt
curl -sLO https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/WN18RR/original/test.txt
cd ../..
```

## WN18RR Results

### ComplEx + Burn 1-N (AdamW) -- MRR 0.424

Full filtered evaluation on the WN18RR test split.

```sh
tranz train --data data/WN18RR --model complex --dim 100 \
    --label-smoothing 0.1 --reciprocals \
    --epochs 50 --lr 0.001 \
    --output output/complex-burn --eval
```

| Metric | Value |
|--------|-------|
| MRR | 0.424 |
| Hits@1 | 0.398 |
| Hits@10 | 0.476 |

The published ComplEx result for WN18RR is 0.475 MRR (Lacroix et al. 2018,
with Adagrad and N3 regularization). The current trainer supports N3 for
ComplEx and DistMult with `--n3-reg <coefficient>`, but it uses AdamW rather
than Adagrad.

### Four-model relative comparison

`examples/wn18rr_kge_burn.rs` reports sampled evaluation over 200 candidates per
test triple. These numbers are useful for comparing models inside one run, but
they read higher than full-entity-ranking MRR.

```sh
cargo run --release --features "burn-ndarray,burn-wgpu" --example wn18rr_kge_burn
```

| Model | Train seconds | Loss | Sampled MRR | H@10 |
|-------|---------------|------|-------------|------|
| DistMult | 39.1 | 10.386 -> 3.979 | 0.5706 | 0.6459 |
| TransE | 32.5 | 9.954 -> 5.013 | 0.5488 | 0.6407 |
| ComplEx | 41.8 | 10.317 -> 4.077 | 0.5560 | 0.6390 |
| RotatE | 57.1 | 9.596 -> 4.774 | 0.5532 | 0.6307 |

## Published reference numbers (WN18RR)

| Model | MRR | H@1 | H@10 | Source |
|-------|-----|-----|------|--------|
| ComplEx | 0.475 | 0.438 | 0.547 | Lacroix et al. 2018 / LibKGE |
| DistMult | 0.452 | 0.413 | 0.530 | Ruffinelli et al. 2020 |
| RotatE | 0.476 | 0.428 | 0.571 | Sun et al. 2019 |
| TransE | 0.226 | 0.053 | 0.501 | Sun et al. 2019 |

## Notes

- The binary uses wgpu when built with `burn-wgpu`; otherwise it uses ndarray.
- CPU eval uses rayon for parallelism. Set `RAYON_NUM_THREADS` to control.
- Reciprocal relations (`--reciprocals`) doubles the training data and
  number of relations. Test metrics are computed on the original test set
  including reciprocal triples.
- 1-N scoring uses more memory than sampled negative training because each
  batch scores every entity.
