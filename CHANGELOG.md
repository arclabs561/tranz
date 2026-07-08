# Changelog

## [Unreleased]

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
