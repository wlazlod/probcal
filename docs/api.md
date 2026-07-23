# API reference

Rendered from the numpy-style docstrings, split across three pages (a single page with
all modules triggers a third-party rendering pathology — DECISIONS 50):

- [Calibrators](api/calibrators.md) — the base contract and all eleven calibration methods.
- [Metrics and tests](api/metrics.md) — the full `probcal.metrics` catalog and `evaluate`.
- [Tools](api/tools.md) — offset, wrapper, selection, curves, plots, attribution,
  thresholds, datasets.

The public surface is exported flat from `probcal`; metrics live under `probcal.metrics`.
