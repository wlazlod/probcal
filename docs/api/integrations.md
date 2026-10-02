# API: integrations

Optional extras; each module imports its dependency only when used.

::: probcal.sklearn

## `CalibratedScorecard` serialization and scope

`CalibratedScorecard` (from `calibrate_scorecard`) serializes its calibration
layer with `to_dict`/`to_json` and checks provenance with `fingerprint()`, like
every probcal artifact. The scorecard itself is optbinning's object and is not
serialized, so loading needs it back:
`CalibratedScorecard.from_json(path_or_str, scorecard=sc)` (or `from_dict(d,
scorecard=sc)`). For the same reason the class is deliberately **not** in the
generic `probcal.load()` registry: `load(d)` could not rebuild it without the
scorecard object.

The stored scorecard fingerprint is SHA-256 of a canonical JSON rendering of
`table(style="detailed")` (floats to 12 significant digits), no longer of
pandas' CSV output, so it does not depend on pandas' formatting. Payloads
written before 0.3.4 still load: a stored fingerprint matching the legacy CSV
hash is accepted, and re-saving writes the new form; any other mismatch
raises `ValueError`.

If the scorecard's probability is constant on the calibration data, the
points-to-log-odds map is unidentifiable: `calibrate_scorecard` warns,
`points_affine_coeffs_` is `None`, and `masterscale()` refuses, exactly as
for `rounding=True`. `predict_proba` returns a 1-D array of calibrated PDs
(the probcal convention, so it feeds `probcal.metrics` directly); a scorecard
is not an sklearn estimator, and `probcal.sklearn.CalibratedClassifier` is the
route when sklearn's `(n, 2)` semantics are needed.

::: probcal.integrations.optbinning
