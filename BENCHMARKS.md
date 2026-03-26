# Benchmarks

Reproducible benchmark results on WN18RR and FB15k-237.

## Setup

```sh
# Install tranz with GPU support
cargo install tranz --features cuda
# Or build from source
git clone https://github.com/arclabs561/tranz
cd tranz
cargo build --release --features cuda --bin tranz

# Download WN18RR
mkdir -p data/WN18RR && cd data/WN18RR
curl -sLO https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/WN18RR/original/train.txt
curl -sLO https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/WN18RR/original/valid.txt
curl -sLO https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/WN18RR/original/test.txt
cd ../..

# Download FB15k-237
mkdir -p data/FB15k-237 && cd data/FB15k-237
curl -sLO https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237/train.txt
curl -sLO https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237/valid.txt
curl -sLO https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237/test.txt
cd ../..
```

## WN18RR Results

### ComplEx + Adagrad (Lacroix recipe) -- MRR 0.438

The closest to published results (0.475). Uses the exact recipe from
Lacroix et al. 2018 (`facebookresearch/kbc`).

```sh
tranz train --data data/WN18RR --model complex --dim 100 \
    --optimizer adagrad --lr 0.1 --init-scale 0.001 \
    --1n --n3 0.1 --reciprocals \
    --epochs 100 --batch-size 100 --log-interval 10 \
    --output output/complex-lacroix --eval --gpu
```

| Metric | Value |
|--------|-------|
| MRR | 0.438 |
| MR | -- |
| Hits@1 | 0.400 |
| Hits@3 | 0.449 |
| Hits@10 | 0.512 |

Hardware: NVIDIA A10G (g5.2xlarge), ~2 hours total (training + eval).
CPU-only: ~8 hours on Apple M-series.

### ComplEx + Adam -- MRR 0.429

Simpler config, slightly lower MRR but faster convergence.

```sh
tranz train --data data/WN18RR --model complex --dim 100 \
    --1n --label-smoothing 0.1 --reciprocals \
    --epochs 50 --batch-size 128 --lr 0.001 --log-interval 10 \
    --output output/complex-adam --eval --gpu
```

| Metric | Value |
|--------|-------|
| MRR | 0.429 |
| Hits@1 | 0.407 |
| Hits@10 | 0.469 |

Note: increasing to dim=200, 200 epochs yields MRR=0.421 (slightly worse).
The Adagrad recipe above is strictly better for ComplEx on WN18RR.

### DistMult + 1-N -- MRR 0.341

```sh
tranz train --data data/WN18RR --model distmult --dim 100 \
    --1n --label-smoothing 0.1 \
    --epochs 50 --batch-size 128 --lr 0.001 --log-interval 10 \
    --output output/distmult --eval
```

| Metric | Value |
|--------|-------|
| MRR | 0.341 |
| Hits@1 | 0.329 |
| Hits@10 | 0.362 |

### RotatE + 1-N -- MRR 0.402

```sh
tranz train --data data/WN18RR --model rotate --dim 100 \
    --1n --reciprocals \
    --epochs 100 --batch-size 128 --lr 0.001 --log-interval 10 \
    --output output/rotate --eval --gpu
```

| Metric | Value |
|--------|-------|
| MRR | 0.402 |
| MR | 7950 |
| Hits@1 | 0.388 |
| Hits@10 | 0.429 |

### TransE + Negative Sampling -- MRR 0.156

TransE with SANS (self-adversarial negative sampling). Lower MRR is
expected -- TransE cannot model symmetric relations on WN18RR.

```sh
tranz train --data data/WN18RR --model transe --dim 100 \
    --epochs 100 --batch-size 1024 --negatives 128 \
    --gamma 9.0 --alpha 1.0 --lr 0.001 --log-interval 25 \
    --output output/transe --eval
```

| Metric | Value |
|--------|-------|
| MRR | 0.156 |
| Hits@1 | 0.002 |
| Hits@10 | 0.421 |

## Published reference numbers (WN18RR)

| Model | MRR | H@1 | H@10 | Source |
|-------|-----|-----|------|--------|
| ComplEx | 0.475 | 0.438 | 0.547 | Lacroix et al. 2018 / LibKGE |
| DistMult | 0.452 | 0.413 | 0.530 | Ruffinelli et al. 2020 |
| RotatE | 0.476 | 0.428 | 0.571 | Sun et al. 2019 |
| TransE | 0.226 | 0.053 | 0.501 | Sun et al. 2019 |

## Notes

- All tranz results use default features unless `--gpu` is specified.
- `--gpu` requires building with `--features cuda` and an NVIDIA GPU.
- CPU eval uses rayon for parallelism. Set `RAYON_NUM_THREADS` to control.
- Reciprocal relations (`--reciprocals`) doubles the training data and
  number of relations. Test metrics are computed on the original test set
  including reciprocal triples.
- 1-N scoring (`--1n`) is dramatically faster than negative sampling for
  convergence but uses more memory (~batch_size * num_entities * 4 bytes).
- The Lacroix recipe (Adagrad + N3 + small init) is proven to work for
  ComplEx. Other models may need different hyperparameters.
