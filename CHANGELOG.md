# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Phase 11: Comprehensive test suite (683 tests, 0 failures).
- Phase 11: Severity calculator zero-day score-boost logic (+0.2 boost, capped at 1.0).
- Phase 11: Detection engine single-sample support via `decision_function` fallback.
- Phase 11: `anomaly_rate` property on `DetectionStats`.
- Phase 11: Config mock fixtures for unit tests.
- Notebooks 01–09: Full pipeline walkthrough and visualization.
- Documentation: 25 pages covering architecture, models, API, research, and reports.
- Deployment: Dockerfile, docker-compose, systemd service, Prometheus config.
- CI/CD: GitHub Actions workflows for tests, lint, and build.
- README.md with project overview, installation, usage, and architecture.

## [0.10.0] — 2026-06-10

### Fixed
- Phase 10: Full repository audit and remediation (all 11 execution groups).
- Source bug fixes: severity.py, engine.py, stream_reader.py.
- Test API alignment with production class signatures.

## [0.9.0] — 2026-06-10

### Added
- Phase 9: Dashboard metrics calculator and Streamlit pages.

## [0.8.0] — 2026-06-10

### Added
- Phase 8: Alerting subsystem (JSON logger, CSV logger, formatter, generator, notifier).

## [0.7.0] — 2026-06-10

### Added
- Phase 7: Streaming simulation (streamer, scheduler, dispatcher, microbatch, simulator, stream_reader).

## [0.6.0] — 2026-06-10

### Added
- Phase 6: Real-time detection engine (engine, alert_manager, severity, decision_logic, orchestrator, response_handler).

## [0.5.0] — 2026-06-10

### Added
- Phase 5: Attack classification with Random Forest (trainer, inference, importance, evaluator).

## [0.4.0] — 2026-06-10

### Added
- Phase 4: Anomaly detection with Isolation Forest (trainer, inference, threshold, evaluator).

## [0.3.0] — 2026-06-10

### Added
- Phase 3: Preprocessing pipeline (cleaner, encoder, scaler, imputer) and feature engineering (selector, transformer, schema, extractor, importance).

## [0.2.0] — 2026-06-10

### Added
- Phase 2: Dataset management (loader, validator, profiler, splitter) with CICIDS2017 support.

## [0.1.0] — 2026-06-10

### Added
- Phase 1: Project foundation (pyproject.toml, 8 YAML configs, core/ and utils/ modules).
- Initial project structure and directory layout.
