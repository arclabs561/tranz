# Changelog

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
