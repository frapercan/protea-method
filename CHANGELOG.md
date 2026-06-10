# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The public feature schema is SemVer-coordinated with `protea-contracts`; a
breaking change to the schema forces a major bump here and re-training of every
downstream LightGBM booster.

## [Unreleased]

### Added
- Exquisite Sphinx documentation: narrative guide pages (overview,
  quickstart, LAFA inference flow, own-reference temporal-cutoff design,
  container usage, contributing) alongside the full API autodoc reference.
- `cutoff` module wired into the API reference.
- Optional `docs` Poetry dependency group (sphinx + alabaster) and a
  `docs` CI workflow that builds with warnings treated as errors.

### Changed
- Deduplicated the lambdarank sigmoid-calibration logic shared by
  `reranker.predict` and `reranker.apply_reranker` into a single
  `_calibrate_scores` helper.
- Aligned the packaged version to the latest release tag.

### Removed
- Unused `_ensure_text_stream` helper (and its `io` import) from
  `io/loaders.py`; the live copy lives in `io/lafa_tsv.py`.

## [0.3.1] - 2026-06-03

### Added
- Torch GPU KNN backend: chunked `cdist` + `topk` with automatic CPU
  fallback and an OOM-driven chunk-halving retry.
- Minimal Sphinx autodoc scaffold.

### Fixed
- Release CUDA tensors between `_search_torch` calls to bound VRAM growth.

### Changed
- Repin `protea-contracts` from `master` to `main`.

## [0.3.0] - 2026-05-11

### Added
- Production inference path packaged as the slim, standalone LAFA
  submission layer (KNN search, feature enrichment, LightGBM re-ranker
  apply, the `predict` orchestrator).
- LAFA container entrypoint (`method_main.py`) with the standard generic
  flags, 3-column TSV output, and `/app/data` + `/app/output` binds.
- In-container PLM embedder for the self-contained container mode
  (`embed/`), with a disk cache keyed by `(backend_id, fasta_sha256)`.

## [0.2.0] - 2026-05-09

### Added
- Initial extraction of the pure inference modules from PROTEA core into
  a dependency-light package (no FastAPI, no SQLAlchemy, no protea-core).

[Unreleased]: https://github.com/frapercan/protea-method/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/frapercan/protea-method/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/frapercan/protea-method/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/frapercan/protea-method/releases/tag/v0.2.0
