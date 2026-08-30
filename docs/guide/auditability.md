# Auditability

What a validator, an internal auditor, or a supervisor can check without
trusting you — and the call that checks it. Nothing on this page is a new
feature; it is the existing surface arranged around one question: *given
these files, what can I re-derive myself?*

## The artifacts

| Artifact | What it proves | How to verify it |
|---|---|---|
| `cal.interpret()` → `Interpretation` | The fitted map's parameters and a plain-language reading of each (`method`, `param_names`, `param_values`, `messages`) — including the awkward facts, such as a monotonicity constraint that forced a refit or an IRLS fit that did not converge | Call it on the *loaded* object and compare with the report's appendix; the messages are generated from the parameters, not stored prose |
| `cal.fit_meta_` | Provenance of the fit: `n_obs`, `n_events`, `weight_sum`, `fitted_at_utc`, `data_fingerprint` (SHA-256 of the row-sorted `(s, y, w)` triple), plus convergence flags | Recompute `data_fingerprint` from the calibration data you were given: a mismatch means the file was not fitted on that data |
| `cal.to_json()` / `from_json()` | Exact reproduction: the file is inert JSON — no pickle, no code execution on load — carrying every parameter `predict_proba` needs | Reload and re-predict; every number must match to 1e-12. The compatibility promise (every 0.x release reads schema 1) is pinned by `tests/golden/` in CI, not asserted in prose |
| `cal.fingerprint()` | Identity: a version- and timestamp-blind SHA-256 of the canonical form. Two identical fits on identical data agree; any change to parameters or state does not | Compare the hash a consumer recorded (a report header, a monitor's audit trail, a recourse certificate) against the hash of the file in front of you |
| `MonitorStep` / `MonitorReport` | Anytime-valid evidence: per batch `e_global`, `p_anytime`, `alarm`, `delta_ci`, `delta_hat`, `slope_hat`, per-grade e-values; per report `alarm_at`, `onset_label`, `recommendation`, `reasoning` | Reload `monitor.json` and reproduce the trajectory batch for batch. The alarm rule is a threshold on a martingale, so it can be re-checked by hand: alarm when `e_global >= 1/alpha` |
| `AppliedAction.audit` | What was changed, when, and on what evidence: `alarm_at`, `onset_label`, `delta`, `se`, `old_monitor_fingerprint`, `new_monitor_fingerprint`, `offset_fingerprint`, `old_target_fingerprint`, `new_target_fingerprint` — with `window` naming the batches the estimate used | Re-estimate the offset from the named window and compare `delta`; check the old-target fingerprint against the calibrator that was actually deployed before the change |
| treecf certificate + `check_certificate(cert, calibrator=...)` | That a recourse recommendation was computed against *this* calibrator: the certificate's calibrated-target block carries `{embedded, fingerprint, type, buffer_logit}` | Load the calibrator from its probcal JSON, match `fingerprint()` against the certificate, then let treecf re-check the stored `lo`/`hi` against the stored raw interval ([treecf guide](treecf.md)) |
| `validation_report(...)` | The assembled pack: reliability, metrics, CORP decomposition, grade backtests, grouped evaluation, monitoring trajectory, calibrator appendix — one self-contained HTML file, no external requests, no scripts | Re-run with the same inputs and `seed`: the document is byte-identical apart from its single `Generated ... UTC` line (pinned by `tests/test_report.py`) |

The chain these form is short and each link is a hash: the report names a
calibrator fingerprint, the calibrator JSON reproduces that fingerprint
and its predictions, the monitor's audit trail names the fingerprints
before and after every change, and a recourse certificate names the
calibrator it was solved against.

## The parameters, in words

`interpret()` is the first thing to read, because it is the only artifact
that states what the fit *did* rather than what it is:

```python
# s_cal, y_cal: held-out calibration scores and outcomes
from probcal import BetaCalibrator

cal = BetaCalibrator().fit(s_cal, y_cal)
interp = cal.interpret()
for message in interp.messages:
    print("-", message)

print(cal.fit_meta_["n_obs"], cal.fit_meta_["n_events"],
      cal.fit_meta_["data_fingerprint"][:12])
```

## What changed, and on what evidence

An offset applied after an alarm is the one moment a deployed pipeline
changes without a re-fit, so it carries its own record:

```python
import numpy as np
from probcal import expit, logit
from probcal.datasets import make_pd_portfolio
from probcal.monitor import CalibrationMonitor

drifted = CalibrationMonitor(alpha=0.05)
for k in range(6):                      # six batches, a +0.8 log-odds drift
    batch = make_pd_portfolio(n=1000, random_state=k)
    rng = np.random.default_rng(k + 1000)
    y_batch = (rng.random(1000) < expit(logit(batch.scores) + 0.8)).astype(float)
    drifted.update(y_batch, batch.scores, label=f"m{k}")

action = drifted.apply_recommendation()
print(action.kind, action.window)       # 're-offset', and the batches it used
print({k: action.audit[k] for k in ("alarm_at", "onset_label", "delta", "se")})
```

`window` is the answer to "which data justified this?", `delta`/`se` the
answer to "how big, and how sure?", and the fingerprint pairs the answer
to "what exactly was replaced by what?". [Monitor and act](monitoring.md)
covers the decision rule; the statistics are in the
[monitoring chapter](../concepts/monitoring.md).

## A verification session

The realistic hand-off is three files: an HTML report, the calibrator's
JSON, and the monitor's JSON. The block below writes that pack into a
temporary directory and then verifies it the way a reviewer would — load,
fingerprint-match, re-predict, re-invert one cutoff, replay the e-process:

```python
# s_cal, y_cal, s_new, grades, mon: held-out calibration data, new scores,
# rating labels, and the deployed monitor; cal is the fit from the block above.
import pathlib
import tempfile

import numpy as np

from probcal import BaseCalibrator
from probcal.monitor import CalibrationMonitor
from probcal.report import validation_report      # probcal[viz]

# --- the modeller's side: produce the pack ------------------------------
pack = pathlib.Path(tempfile.mkdtemp())
cal.to_json(pack / "calibrator.json")
mon.to_json(pack / "monitor.json")                # mon: the deployed monitor
validation_report(y_cal, cal.predict_proba(s_cal), calibrator=cal, monitor=mon,
                  grades=grades, n_boot=20, seed=42, path=pack / "validation.html")

# --- the reviewer's side: nothing but the three files -------------------
loaded = BaseCalibrator.from_json(pack / "calibrator.json")
assert loaded.fingerprint() == cal.fingerprint()                    # identity
np.testing.assert_allclose(loaded.predict_proba(s_new),
                           cal.predict_proba(s_new), atol=1e-12)    # reproduction
assert (loaded.interval_inverse(0.0, 0.02, space="logit")
        == cal.interval_inverse(0.0, 0.02, space="logit"))          # the 2% cutoff
replay = CalibrationMonitor.from_json(pack / "monitor.json")
np.testing.assert_allclose([s.e_global for s in replay.steps_],
                           [s.e_global for s in mon.steps_])        # the evidence
assert cal.fingerprint() in (pack / "validation.html").read_text()  # same object
print("verified:", cal.fingerprint()[:12])
```

One caveat on how that block reads: here the right-hand side of every
assertion is a live object still in memory, because the page produces and
verifies the pack in one process — in a real review the right-hand sides
are numbers printed in the report the reviewer was handed, copied out of
its fingerprint block, and the assertions are the same.

Note what the reviewer never needed: the training data, the scoring model,
the modeller's environment, or any code from the modelling team. Two of
the five checks (`from_json`, `fingerprint`) are also what the
[serialization chapter](../concepts/serialization.md) promises across
releases, so the same session still runs against an artifact archived two
minors ago.

## What this does not prove

The chain above is about *artifacts*, and it is worth being exact about
where it stops.

It does not prove that the calibration data was the right data: that the
outcomes had matured, that the sample represents the portfolio the model
scores, or that nothing leaked from the training split. `data_fingerprint`
pins *which rows* were used only if you still hold those rows to hash. It
does not say anything about the underlying model — a well-calibrated
forecast from a discriminatively useless score is still well calibrated,
and calibration metrics will not tell you otherwise. It does not prove the
deployed system actually calls this calibrator; only a fingerprint
recorded by the deployed system at scoring time can do that, which is why
`fingerprint()` exists and why the monitor and treecf both record it. And
it does not prove that the policy is right: a cutoff exactly inverted from
a wrong PD target is exactly wrong.

What it does prove is narrower and worth having: that the object in the
file is the object that produced the numbers in the report, that anyone
can reproduce those numbers from it, and that every change made to it
since is named, dated, sized, and attributed to specific evidence.

## Related

- [Build a validation report](report.md) — the assembled document.
- [Serialize and persist](serialization.md) — the how-to behind the JSON.
- [Serialization](../concepts/serialization.md) — schema, registry, and
  the golden-file compatibility promise.
- [Set cutoffs and invert maps](cutoffs.md) — re-deriving the cutoff the
  session above re-checks.
- [Choose a calibrator](choosing.md) — which fitted parameters there are
  to interpret in the first place.
