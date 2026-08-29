# Serialization

Every fitted probcal object serializes to versioned, human-readable JSON:

```python
from probcal import BetaCalibrator, make_pd_portfolio

port = make_pd_portfolio(n=8000, random_state=42)
cal = BetaCalibrator().fit(port.scores, port.y)

cal.to_json("beta.json")                       # or text = cal.to_json()
loaded = BetaCalibrator.from_json("beta.json") # bit-identical predictions
cal.fingerprint()                              # sha-256 provenance id
```

## What is stored

`to_dict()` produces one envelope for every class:

| Key | Content |
|---|---|
| `probcal_schema` | Serialization schema version (currently 1) |
| `probcal_version` | The probcal release that wrote the file |
| `class` | Registered class name, used for `from_dict` dispatch |
| `params` | Constructor parameters (`get_params()`) |
| `state` | The fitted attributes — everything `predict_proba` needs |
| `fit_meta` | `n_obs`, `n_events`, `weight_sum`, `fitted_at_utc` (ISO 8601), `data_fingerprint`, convergence flags where they exist |

State is JSON-native throughout: floats, ints, bools, strings; 1-D float64
arrays as plain lists; any other array as a nested list tagged with explicit
`dtype`/`shape`; nested probcal objects (a scaling-binning stage's Platt map,
a selector's winner, a wrapper's offsets) embed their own full envelopes.
`VennAbersCalibrator` stores its sorted calibration set and both
cumulative-sum-diagram sweeps — O(n) by design, because that *is* the fitted
map. `CalibratedModel` stores a *reference* to the base model (class name,
user-supplied `model_id`, `get_params()` when JSON-encodable), never the
model object; reattach on load with
`CalibratedModel.from_dict(d, model=...)`.

## Why JSON — and never pickle

Pickle executes arbitrary code on load: a poisoned artifact is a remote-code
path, and no reviewer can read one. A probcal JSON is inert data a validator
can open, diff, and archive — the parameters a regulator asks about are in
plain sight, and loading it can only ever build registered probcal classes
via their documented constructors. There is no pickle anywhere in the
package, and there will not be.

## The registry

`from_dict` dispatches through a registry of serializable classes
(`class` name → class), filled by a decorator on every public calibrator,
`LogitOffset`, `CalibratorSelector`, and `CalibratedModel`. Calling
`BaseCalibrator.from_dict(d)` loads whatever class wrote `d`; calling
`SomeCalibrator.from_dict(d)` additionally requires the payload to have
been written by that class. Unknown schema versions and unknown class names
raise `ValueError` naming what was found.

## The compatibility promise

**Every 0.x release reads schema 1.** A schema bump ships only together
with a converter for old files and a changelog entry. The promise is
enforced, not aspirational: `tests/golden/` holds one committed JSON per
registered class, written at the release that introduced serialization, and
CI loads each one and reproduces its stored predictions (to 1e-12) on every
run. A change that breaks reading old files breaks the build.

## Fingerprints

Two hashes serve provenance:

- **`fingerprint()`** — SHA-256 of the canonical JSON of `to_dict()`,
  excluding `probcal_version` and the fit timestamps. Two identical fits on
  identical data produce the same fingerprint; any change to parameters or
  fitted state changes it. Consumers (model registries, the monitoring
  workstream, recourse engines) record it to name exactly which calibrator
  a decision was computed against.
- **`fit_meta["data_fingerprint"]`** — SHA-256 of the row-sorted training
  triple `(s, y, w)` (probabilities and weights only, for `LogitOffset`).
  Sorting makes it permutation-invariant: the same sample in any order
  hashes identically.

## Storing the JSON next to the model artifact

The intended deployment pattern keeps the calibrator's JSON beside the
model in whatever registry the model lives in:

```python
artifact_dir = registry.upload(model, name="pd-model", version="2026.09")
cal.to_json(artifact_dir / "calibrator.json")
registry.tag(name="pd-model", version="2026.09",
             calibrator_fingerprint=cal.fingerprint())
```

At scoring time, load both, and let downstream consumers verify
`cal.fingerprint()` against the tag before trusting a threshold or a
recourse certificate computed from it.
