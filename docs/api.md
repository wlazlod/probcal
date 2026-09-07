# API reference

Rendered from the numpy-style docstrings, split across four pages (one page with
every module renders too slowly to use):

- [Calibrators](api/calibrators.md): the base contract and all thirteen calibrators.
- [Metrics and tests](api/metrics.md): the full `probcal.metrics` catalog and `evaluate`.
- [Tools](api/tools.md): offset, wrapper, selection, curves, plots, attribution,
  thresholds, datasets, `Chain`, the monitor, and the `expit`/`logit` helpers.
- [Integrations](api/integrations.md): the scikit-learn adapter (`probcal.sklearn`)
  and optbinning scorecards (`probcal.integrations.optbinning`).

The public surface is exported flat from `probcal`; metrics live under `probcal.metrics`
and monitoring under `probcal.monitor`.
