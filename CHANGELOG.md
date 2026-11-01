# Changelog

## Source history (unversioned)

These source changes are not assigned to a package release here.

- Numeric CLI options report missing or malformed values with exit status 1
  before loading input, instead of panicking. Negative or overflowing unsigned
  integers are rejected; floating-point domain policy is unchanged.
  ([source](https://github.com/arclabs561/tranz/commit/e2e5cf1f7dea096de74e1944366de1e1432f3a28))
- `train` and `train-temporal` accept zero epochs and export initialized
  embeddings, reporting that no training loss exists. Burn training preserves
  the batch axis for singleton batches, including a final one-item remainder.
  ([source](https://github.com/arclabs561/tranz/commit/e2e5cf1f7dea096de74e1944366de1e1432f3a28))
- Benchmark documentation labels the existing WN18RR result as historical,
  predating realistic-tie and reciprocal-evaluation corrections. It requires
  a rerun before comparison; no new benchmark measurement is reported.
  ([source](https://github.com/arclabs561/tranz/commit/ef9728d853e9e9b6e1e2636b1e35942b7c29a516))

## [Unreleased]

## [0.9.0] - 2026-11-01

### Added

- `evaluate_temporal_link_prediction_reciprocal` evaluates each augmented quad
  once as a tail query, with timestamp-aware filtering and realistic ties.
  `train-temporal --reciprocals --eval` uses this protocol and reports original
  and inverse queries as tail and head metrics, respectively. The existing
  temporal evaluator and nonreciprocal CLI path still score both directions.

### Changed

- Burn 0.20 -> 0.22, and the CPU backend from `ndarray` to `flex`. Backends
  are selected through the `burn` facade's features now, so `burn-ndarray` and
  `burn-wgpu` are no longer direct dependencies; `burn = "0.22"` plus
  `burn/flex` or `burn/wgpu` replaces them. The CPU feature is renamed
  `burn-ndarray` -> `burn-flex`, a breaking change for feature selection:
  build with `--features burn-flex`, or `--features "burn-flex,burn-wgpu"` for
  GPU/Metal.
- Burn 0.22 removed the backend type parameter. `train_complex`,
  `train_kge` and `train_tcomplex` (and their `_seeded` variants) no longer
  take a `B: AutodiffBackend` parameter, and `BurnComplEx`, `BurnKge` and
  `BurnTComplEx` are no longer generic over a backend. The device argument is
  now `&burn::tensor::Device`, chosen at runtime (`Device::flex()`,
  `Device::wgpu(..)`, `Device::default()`); the trainers enable autodiff on it
  themselves. Seeding is `device.seed(seed)` rather than a backend associated
  function.
- The minimum supported Rust version is 1.95, Burn 0.22.1's MSRV.

### Burn record (checkpoint) migration

Nothing in tranz persists a Burn record, so no checkpoint held by a tranz user
is affected by Burn 0.22's move from the MessagePack record format to
burnpack. Training output is plain vectors: `train*` returns
`Vec<Vec<f32>>`, and the CLI writes TSV/bin embeddings, `vocab.txt` and
`manifest.json`, all unchanged. The Burn module structs are per-run internals
with private fields and no persistence path, and the crate carries no recorder
code to delete.

A checkpoint you made yourself *with Burn 0.20* is a different matter: a
`.mpk` file written through `NamedMpkFileRecorder` (or any 0.20 record) cannot
be read by 0.22, which knows only burnpack. To keep it, export its tensors
under Burn 0.20 with `burn-store`/`NamedMpkFileRecorder` (safetensors, or plain
tensor data) and import those tensors under 0.22, or retrain: the seeded
trainers make a run reproducible.

### Fixed

- Static and temporal reciprocal augmentation now assigns a distinct inverse
  relation name when an original relation already uses the conventional
  `_inv` suffix. The fixed offset IDs and ordinary inverse names are retained.
- Temporal link-prediction evaluation now uses realistic midranks for ties,
  matching static evaluation while preserving timestamp-aware filtering.
- Sampled evaluation documentation now describes candidate-subset metrics
  and their sampling/filtering behavior without claiming unbiased full ranks.

## [0.8.0] - 2026-07-08

### Added

- `Scorer::score_all_tails_batch` and `Scorer::score_all_heads_batch` score
  many same-relation projections at once. `answer_query` uses the tail batch
  path for beam projection.

### Changed

- Query configuration now uses `tnorms::LogicFamily` directly instead of the
  local `query::TNorm` shim. `TNorm::Min` becomes `LogicFamily::Godel`.
- Negation in query answering now delegates to `LogicFamily::Lukasiewicz`.
- `Scorer` and `TemporalScorer` now require `Send + Sync`, so scorer objects
  can move across worker threads as well as be shared by reference.

## [0.7.4] - 2026-07-07

### Changed

- Query-answering t-norm and t-conorm helpers now delegate their standard
  Gödel/min and Product formulas to `tnorms`.

## [0.7.3] - 2026-07-04

### Fixed

- `BurnTrainConfig::n3_reg` was declared but never applied by the static
  trainer (`train_kge`); it now applies the Lacroix et al. (ICML 2018)
  weighted nuclear-3 penalty for the CP-family models (ComplEx, DistMult)
  and is ignored with a CLI warning for the distance models. `tranz train`
  gains `--n3-reg`.

## [0.7.2] - 2026-07-04

### Added

- `TemporalScorer::score_all_tails_over`: per-entity minimum energy over a
  timestamp set (the existential fold behind timestamp-set hops), with a
  rayon-parallel `TComplEx` override. Not-during hops on ICEWS05-15's
  4017-day axis are the hot path this exists for.

## [0.7.1] - 2026-07-04

### Added

- Temporal training: `--n3-reg` enables the weighted nuclear-3 regularizer
  Ω³ (Lacroix et al., ICLR 2020, Eq. 4), penalizing the
  `(relation ∘ timestamp)` product as one factor per the paper's order-4
  unfolding argument. Measured on ICEWS14 at dim 256: valid MRR 0.34 → 0.53
  with both regularizers on (use `--init-scale 0.01`; with tiny init the
  origin is a fixed point of the trilinear score and Ω³ pins the model
  there).
- `TemporalScorer::score_all_times`: the timestamp-answering direction
  (time projection), with a hoisted `TComplEx` override.
- CLI: `--eval-split valid|test`, so hyperparameter selection can run on
  the validation split.

### Changed

- `--time-smooth` now applies the paper's Λ₃ penalty (cubed absolute
  discrete derivative, normalized by `|T|−1`) instead of a squared-L2
  mean; results for nonzero values differ from 0.7.0. The previous form
  was a deviation from the cited reference.

## [0.7.0] - 2026-07-04

### Added

- `temporal` module: quad datasets (`load_temporal_dataset` with
  chronologically interned timestamps, reciprocal augmentation), the
  `TemporalScorer` trait (same lower-is-better convention as `Scorer`), the
  `TComplEx` CPU scorer (Lacroix et al., ICLR 2020), and time-aware filtered
  link-prediction evaluation.
- `burn_temporal::train_tcomplex`: 1-N cross-entropy trainer for TComplEx in
  both directions with a temporal smoothness penalty, plus the
  `tranz train-temporal` CLI subcommand exporting `entities.tsv`,
  `relations.tsv`, and `times.tsv`.
- `tranz train` now writes `manifest.json` beside `entities.tsv` and
  `relations.tsv`, recording model family, training config, split sizes,
  SHA-256 artifact digests, byte sizes, and aggregate evaluation metrics when
  `--eval` is used.
- `load_embedding_manifest` and `verify_embedding_manifest` behind the
  `artifact-manifest` feature read `manifest.json` back and verify exported
  artifact paths, byte lengths, and SHA-256 digests.

### Fixed

- Burn trainer ComplEx head scoring (`score_1n_heads`, `score_1n_heads_kge`)
  added the imaginary product where `Re(h * r * conj(t))` requires subtracting
  it, training a mis-signed head-prediction objective that the CPU evaluator
  did not share. Both sites now match the CPU reference; a Burn-vs-CPU score
  parity test guards the convention.

## [0.6.0] - 2026-06-28

### Removed

- `candle` and `cuda` features, the candle `train` module, and the candle-only
  examples (`train_wn18rr`, `bench_training`, `bench_burn`). Training now runs
  entirely on Burn. Breaking: the `candle`/`cuda` feature flags and the
  `tranz::train` module no longer exist.

### Changed

- `tranz train` trains via Burn `train_kge` (1-N/1vsAll cross-entropy, AdamW) on
  the `burn-ndarray` (CPU) or `burn-wgpu` (Metal/Vulkan) backend. The candle-only
  CLI flags (`--optimizer`, `--gamma`, `--negatives`, `--alpha`, `--n3`,
  `--norm`, `--dropout`, `--subsampling`, `--normalize`, `--warmup`,
  `--cosine-cycles`, `--checkpoint`, `--l2`, `--swa`, `--rel-pred`, `--gpu`) are
  removed; 1-N is the only training mode (`--1n` is accepted as a no-op).
- The bin and training examples require a Burn backend feature
  (`--features burn-ndarray` for CPU, add `burn-wgpu` for GPU/Metal).

## [0.5.3] - 2026-06-10

### Deprecated

- `candle` feature is deprecated and will be removed in 0.6.0 (next major). Burn upstream deprecated `burn-candle` (PR tracel-ai/burn#4416), so the dual-framework pattern no longer composes cleanly. Migrate to `burn-cpu` (CPU) or `burn-gpu` (Wgpu cross-platform GPU). For NVIDIA-specific deployments, a `burn-cuda` feature is on the roadmap.

## 0.4.0 (2026-03-26)

- Adagrad optimizer (proven for KGE by Lacroix et al. 2018)
- Configurable init scale (N(0, 1e-3) default, matching kbc reference)
- Fixed N3 regularization for ComplEx: operates on complex moduli
- 1vsAll mode (single-target CE, default) vs KvsAll (multi-hot) toggle
- Bidirectional 1-N scoring (head + tail prediction)
- Multi-hot label support for KvsAll mode
- SnapE cosine annealing with snapshot collection
- L2 regularization on embeddings
- Mean Rank metric in evaluation
- Per-epoch timing and embedding RMS diagnostics
- Preallocated target buffers (eliminates ~55GB/epoch allocation)
- CUDA support via --gpu flag
- 69 tests
- ComplEx achieves MRR=0.438 on WN18RR (92% of published 0.475)

## 0.3.1 (2026-03-26)

- Fix ComplEx 1-N training: switch from BCE to softmax cross-entropy loss
- Add eval subcommand for evaluating saved embeddings
- Add checkpoint saving during training
- Add scoring example (no candle dependency needed)
- Verify FB15k-237 dataset support

## 0.3.0 (2026-03-26)

- Add 1-N scoring with label smoothing (5-10x faster convergence)
- Add configurable L1/L2 distance norm
- Add embedding dropout, subsampling weights
- Add per-relation evaluation breakdown
- Optimize batch scoring for all 4 models
- Parallelize evaluation via rayon
- Add LR warmup and training progress logging
- Complete CLI: train + eval + predict subcommands
- DistMult achieves MRR=0.341 on WN18RR (50 epochs, dim=100)

## 0.2.1 (2026-03-25)

- Parallelize evaluation with rayon
- Add embedding export (w2v TSV) and import
- Add batch scoring and top-k retrieval to Scorer trait
- Add generic TSV/CSV triple loader with Dataset::split
- Add predict subcommand to CLI
- Add contiguous matrix export for vector DB handoff

## 0.2.0 (2026-03-25)

- Add RotatE, ComplEx, DistMult models
- Add candle-based GPU training with SANS and N3 regularization
- Add reciprocal relation augmentation
- Make TransE fields private with validated constructors
- Add `#[non_exhaustive]` on Error enum
- Per-model initialization: Xavier (TransE), gamma-based (RotatE)
- f64 accumulators in scoring inner loops
- Head/tail corruption and epoch shuffling
- CI (stable + MSRV) and OIDC release workflow
- 34 tests

## 0.1.0 (2026-03-25)

- Initial release: TransE model, dataset loading, filtered evaluation
- WN18RR-format support
- Scorer trait for triple scoring
