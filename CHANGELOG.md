# Changelog

All notable changes to `hifd` will be documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-05-23

### Added
- Initial release.
- Per-task agreement primitives: `score_age`, `score_categorical`,
  `score_landmark`, `score_gaze`, `score_bvp`, `score_hr`.
- Privacy (`privacy_score`, `privacy_ensemble`) and quality (`quality_score`)
  primitives, with batch variants for the scalar-input ones.
- Utility levels `U1` / `U2` / `U3` and per-method aggregation.
- Composite HiFD score (`composite`, `compose_profiles`) under three default
  application profiles (Privacy-First, Balanced, Clinical).
- JSON envelope IO (schema version `1.0`) with full validation.
- High-level `run_pipeline` orchestrator producing per_sample / per_method /
  composite CSVs plus a LaTeX-ready composite table.
- `hifd` CLI: `validate`, `compute-agreements`, `aggregate`, `compose`,
  `run-all`, `version`.
- PEP 561 type information (`py.typed`).
