# Contributing to tranz

Thanks for your interest. tranz is point-embedding knowledge graph models: TransE, RotatE, ComplEx, DistMult. GPU training via Burn (wgpu/Metal).

## Before you start

For non-trivial work (new APIs, features, large refactors), open an issue first to align on scope. Drive-by bug fixes and doc patches don't need an issue.

## Setup

- Rust toolchain: stable, MSRV `1.89`. Use `rustup` to manage.
- Optional: `cargo-nextest` for faster test runs (`cargo install cargo-nextest`).

```
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

## Style

- Direct, lowercase prose in commits. No marketing words ("powerful", "robust", "elegant"). No em-dashes in prose.
- Commit messages: `tranz: short lowercase description`. One commit per logical change.
- `cargo fmt` and `cargo clippy --all-targets --all-features -- -D warnings` must pass before `git add`.

## Testing

- `cargo test --all-features` for the full matrix.
- Test names should describe the property under test, not the function under test.

## Pull requests

- Keep PRs scoped to one concern.
- Show before/after for behavior changes.
- Link the related issue.
- CI must be green before requesting review.

## License

Dual-licensed under MIT or Apache-2.0 at your option. By contributing you agree your contributions are licensed under both.
