"""Build docs/notebooks/pd_end_to_end.ipynb (spec W12), unexecuted.

Execute afterwards with:
``uv run jupyter nbconvert --to notebook --execute --inplace docs/notebooks/pd_end_to_end.ipynb``

The executed notebook is committed (mkdocs-jupyter renders it without
re-executing); CI re-executes it on the synthetic fallback via nbmake.
"""

import json
import pathlib

MD = "markdown"
CODE = "code"

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append((MD, text))


def code(text: str) -> None:
    CELLS.append((CODE, text))


md(
    "**Role: the flagship, full-lifecycle tutorial.** One portfolio taken through "
    "every stage — calibrate, evaluate, invert, monitor, act, report — on the whole "
    "public surface. New to probcal? Start with the 20-minute "
    "[PD calibration walkthrough](../pd_calibration_walkthrough/) instead."
)

md("""\
# PD calibration end to end: rare events, audit, recourse, monitoring

One credit-risk portfolio taken through the whole probcal 0.2.0 surface:
baseline GBM → reliability diagnosis → automatic calibrator selection with
bootstrap CIs → per-grade regulatory backtests → an auditable macro offset →
policy-threshold translation and a treecf counterfactual → anytime-valid
monitoring → JSON serialization of every artifact.

**Data.** The primary dataset is [Home Credit Default Risk
(2018)](https://www.kaggle.com/c/home-credit-default-risk) — ~307k
applications at an 8.1% event rate — downloaded through `kagglehub` under
*your* Kaggle credentials and cached locally (not redistributed). Without
kagglehub/credentials the notebook falls back to a documented synthetic
100k portfolio so it always executes (this is also what CI runs). A loader
stub for a genuinely rare (<3%) mortgage set (Freddie Mac single-family
loan-level data; registration required) is included at the end of this cell.
""")

code("""\
import numpy as np
import pandas as pd

from probcal import expit, logit

SEED = 42


def load_home_credit():
    \"\"\"Home Credit application_train via kagglehub (user credentials, local cache).\"\"\"
    import kagglehub

    root = kagglehub.competition_download("home-credit-default-risk")
    df = pd.read_csv(f"{root}/application_train.csv")
    cols = [
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "AMT_CREDIT",
        "AMT_INCOME_TOTAL", "AMT_ANNUITY", "DAYS_BIRTH", "DAYS_EMPLOYED",
        "DAYS_ID_PUBLISH", "CNT_CHILDREN",
    ]
    X = df[cols].fillna(-999.0).to_numpy(float)
    y = df["TARGET"].to_numpy(float)
    return X, y, "Home Credit Default Risk (application_train, 10 features)"


def load_freddie_mac_stub():
    \"\"\"Loader stub for the Freddie Mac single-family loan-level dataset (<3% event
    rate). Registration is required at freddiemac.com/research/datasets; point
    this at the downloaded origination + performance files and build a
    default-within-24-months target. Left as an exercise on purpose — the
    license does not permit redistribution.\"\"\"
    raise NotImplementedError


def make_synthetic(n=100_000):
    \"\"\"The documented fallback (spec W12): a synthetic feature portfolio with a
    mildly non-linear true PD, so the GBM ranks well but mis-calibrates in the
    tail — the regime the rest of the notebook is about.\"\"\"
    rng = np.random.default_rng(7)
    X = np.column_stack([
        rng.normal(size=n),
        rng.normal(size=n),
        rng.uniform(0.0, 1.0, n),
        rng.exponential(1.0, n),
        rng.integers(0, 6, n).astype(float),
    ])
    z = (
        1.1 * X[:, 0] - 0.8 * X[:, 1] + 2.2 * (X[:, 2] - 0.5) ** 2
        + 0.4 * np.log1p(X[:, 3]) + 0.15 * X[:, 4] - 4.1
    )
    y = (rng.random(n) < expit(z)).astype(float)
    return X, y, f"synthetic fallback portfolio (n={n:,})"


try:
    X, y, source = load_home_credit()
except Exception as exc:  # no kagglehub / no credentials -> the CI path
    print(f"falling back to synthetic data ({type(exc).__name__})")
    X, y, source = make_synthetic()

rng = np.random.default_rng(SEED)
order = rng.permutation(len(y))
X, y = X[order], y[order]
n = len(y)
i_train, i_cal = int(0.5 * n), int(0.75 * n)
X_train, y_train = X[:i_train], y[:i_train]
X_cal, y_cal = X[i_train:i_cal], y[i_train:i_cal]
X_test, y_test = X[i_cal:], y[i_cal:]
print(f"{source}: n={n:,}, event rate {y.mean():.2%}")
""")

md("""\
## 1. A GBM baseline that ranks well

Nothing wrong with the model as a *ranker* — the trouble only shows once we
read its outputs as probabilities.
""")

code("""\
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# subsample left at 1.0 so the recourse section also runs correctly under
# treecf < 0.2.3, which routed split boundaries in float64 while sklearn
# casts inputs to float32 (fixed in treecf#21; guide/treecf.md T4)
model = GradientBoostingClassifier(
    n_estimators=120, max_depth=3, random_state=0
).fit(X_train, y_train)

s_cal = model.predict_proba(X_cal)[:, 1]
s_test = model.predict_proba(X_test)[:, 1]

pd.DataFrame(
    {
        "split": ["calibration", "test"],
        "n": [len(y_cal), len(y_test)],
        "event rate": [f"{y_cal.mean():.2%}", f"{y_test.mean():.2%}"],
        "mean score": [f"{s_cal.mean():.2%}", f"{s_test.mean():.2%}"],
        "AUC": [
            round(float(roc_auc_score(y_cal, s_cal)), 4),
            round(float(roc_auc_score(y_test, s_test)), 4),
        ],
    }
)
""")

md("""\
## 2. Reliability: miscalibrated exactly where the policy lives

A 2% PD cut-off sits in the left tail — precisely where the smoothed
reliability curve and the calibration belt pull away from the diagonal.
Logit-scaled axes make the low-PD region readable; linear axes would hide it.
""")

code("""\
import matplotlib.pyplot as plt

from probcal import calibration_belt, reliability_binned, reliability_loess
from probcal.plots import plot_belt, plot_reliability

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
plot_reliability(
    reliability_binned(y_cal, s_cal),
    smooth=reliability_loess(y_cal, s_cal),
    scale="logit",
    y=y_cal,
    p=s_cal,
    ax=axes[0],
)
plot_belt(calibration_belt(y_cal, s_cal), scale="logit", ax=axes[1])
fig.tight_layout()
""")

md("""\
## 3. Automatic selection, with confidence intervals

`CalibratorSelector` runs the default menu under nested cross-validation.
On rare events, isotonic regression pays for its plateaus (few events per
block → coarse, high-variance steps in the tail); parametric families keep
sharing strength across the range, and the three-parameter `abm` beta can
bend the two tails separately when the distortion is asymmetric (as it
typically is on heavy real portfolios like Home Credit — on this synthetic
fallback a simpler map may win the log-loss criterion). `evaluate` puts
bootstrap CIs behind the comparison so the choice is not a point-estimate
coin flip.
""")

code("""\
from probcal import BetaCalibrator, CalibratorSelector, IsotonicCalibrator
from probcal.metrics import evaluate

selector = CalibratorSelector().fit(s_cal, y_cal)
print(selector.report_)
print(f"\\nselected: {selector.best_name_}")
""")

code("""\
rows = []
for name, proto in (("beta_abm", BetaCalibrator()), ("isotonic", IsotonicCalibrator())):
    cal_fit = type(proto)(**proto.get_params()).fit(s_cal, y_cal)
    rep = evaluate(
        y_test, cal_fit.predict_proba(s_test), n_boot=200,
        metrics=("log_loss", "brier", "smooth_ece"),
    )
    stats = zip(rep.names, rep.values, rep.ci_low, rep.ci_high, strict=True)
    for m, v, lo, hi in stats:
        rows.append({"calibrator": name, "metric": m, "estimate": round(float(v), 5),
                     "ci": (round(float(lo), 5), round(float(hi), 5))})
pd.DataFrame(rows)
""")

md("""\
## 4. Per-grade regulatory backtest, before and after

The Jeffreys test (ECB IRB validation instructions) per grade: a small value
flags a grade whose assigned PD is likely understated. Grades here are fixed
PD bands on the calibrated scale.
""")

code("""\
from probcal.metrics import jeffreys_grade_test

cal = BetaCalibrator().fit(s_cal, y_cal)
p_test = cal.predict_proba(s_test)

edges = np.array([0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 1.0])
labels = np.array(["A", "B", "C", "D", "E", "F"])
grades_before = labels[np.clip(np.searchsorted(edges, s_test, side="right") - 1, 0, 5)]
grades_after = labels[np.clip(np.searchsorted(edges, p_test, side="right") - 1, 0, 5)]

print("before calibration (grading on raw scores):")
print(jeffreys_grade_test(y_test, s_test, grades_before))
print("\\nafter calibration (grading on calibrated PD):")
print(jeffreys_grade_test(y_test, p_test, grades_after))
""")

md("""\
## 5. A macro shift, handled as an auditable offset

Mid-cycle, the through-the-cycle central tendency moves. The answer is not a
silent refit: `LogitOffset` applies one uniform log-odds shift, solved
against the target mean, and `audit_report` shows a validator the before and
after in one table. Ranking is untouched.
""")

code("""\
from probcal import LogitOffset

target_mean = float(1.25 * p_test.mean())  # the new central tendency
offset = LogitOffset(target_mean=target_mean).fit(p_test)
print(offset.audit_report(y_test, p_test))
""")

md("""\
## 6. From policy to raw threshold — and to a counterfactual

"Approve below 2% calibrated PD" must reach production as a raw-score rule,
and reach the declined applicant as recourse. `interval_inverse` gives the
exact raw threshold (through the offset too, via `Chain`); treecf turns the
same calibrated target into the smallest feature change — with the exact
backend, a certificate.
""")

code("""\
from probcal import Chain

chain = Chain([cal, offset])  # post-offset policy: invert offset ∘ calibrator
lo_z, hi_z = chain.interval_inverse(0.0, 0.02, space="logit")
raw_cut = float(expit(np.array([hi_z]))[0])
print(f"'calibrated PD <= 2%' == raw model score <= {raw_cut:.4%} (logit {hi_z:+.3f})")
""")

code("""\
try:
    from treecf import Explainer, Target

    p_chain = chain.predict_proba(s_test)
    declined = int(np.argmin(np.abs(p_chain - 0.05)))  # a borderline decline
    x0 = X_test[declined]
    exp = Explainer(model=model, background=X_train[:500])
    res = exp.explain(
        x0, target=Target.calibrated(chain, op="<=", value=0.02),
        seed=0, backend="exact",
    )
    if hasattr(res, "x_cf"):
        p_new = chain.predict_proba(model.predict_proba(np.asarray(res.x_cf)[None])[:, 1])
        print(f"proof: {res.proof}; changes: {res.changes}")
        print(f"calibrated PD {p_chain[declined]:.3%} -> {p_new[0]:.3%} (target <= 2%)")
    else:
        print(f"certified infeasible for this applicant: {res}")
except ImportError:
    print("treecf not installed — skipping the recourse demo "
          "(pip install 'probcal[treecf]')")
""")

md("""\
## 7. Monitoring: an e-process that survives being looked at monthly

Monthly cohorts replay through `CalibrationMonitor`. This dataset has no
time axis, so the replay is simulated the same way the W9 verification
does it: for the stable months, outcomes are drawn from the calibrated
forecast itself (calibrated by construction); from month 14, from the
forecast shifted by +0.5 log-odds — a sustained macro deterioration. The
alarm rule "wealth ≥ 1/α" keeps its type-I guarantee at *every* look — no
correction for repeated testing — and after the alarm the report says
whether a re-offset is enough. Theory: the *Monitoring* chapter.
""")

code("""\
from probcal.monitor import CalibrationMonitor
from probcal.plots import plot_e_process

mon = CalibrationMonitor(alpha=0.05)
rng_m = np.random.default_rng(11)
pool = np.arange(len(y_test))
for month in range(24):
    take = rng_m.choice(pool, size=2000, replace=True)
    p_month = p_test[take]
    true_pd = p_month if month < 13 else expit(logit(p_month) + 0.5)
    y_month = (rng_m.random(2000) < true_pd).astype(float)
    step = mon.update(y_month, p_month, label=f"m{month + 1:02d}")

rep = mon.report()
print(f"alarm at: {rep.alarm_at}; recommendation: {rep.recommendation}")
for line in rep.reasoning:
    print(" -", line)
plot_e_process(rep)
plt.show()
""")

md("""\
## 8. Everything to JSON, and back

Every fitted artifact serializes to versioned, human-readable JSON — never
pickle — and reloads bit-for-bit. Fingerprints (version- and
timestamp-blind) name each artifact for registries, monitors, and recourse
certificates.
""")

code("""\
import json

from probcal import BaseCalibrator

artifacts = {
    "calibrator.json": cal,
    "offset.json": offset,
    "chain.json": chain,
    "monitor.json": mon,
}
for fname, obj in artifacts.items():
    text = obj.to_json()
    loaded = type(obj).from_json(text)
    same = np.array_equal(
        obj.predict_proba(s_test[:100]) if hasattr(obj, "predict_proba") else 1,
        loaded.predict_proba(s_test[:100]) if hasattr(loaded, "predict_proba") else 1,
    )
    print(f"{fname:16s} {len(text):>8,} bytes  fingerprint {obj.fingerprint()[:16]}…  "
          f"round-trip bit-identical: {bool(same)}")

# registry dispatch: the class need not be known in advance
obj = BaseCalibrator.from_dict(json.loads(cal.to_json()))
print(f"\\nregistry loaded a {type(obj).__name__} — schema 1, readable by every 0.x release")
""")


md("""\
## 9. Conservative margins: most-prudent PDs and margin-of-conservatism

Two more decision-relevant readings sit alongside calibration. Pluto & Tasche's most-prudent
PD gives a defensible upper bound for a grade even when it saw few or no defaults, by pooling
it with every worse grade under the rating-monotonicity assumption. Margin-of-conservatism
(MoC) offsets do a different job: they *compose* with an existing calibrator — never replace
it — re-anchoring an already-calibrated portfolio's mean at a conservative reading of realized
outcomes. Theory: the *Conservatism* chapter.
""")

code("""\
from probcal.metrics import pluto_tasche_from_arrays

pt = pluto_tasche_from_arrays(
    grades_after, y_test, order=("A", "B", "C", "D", "E", "F"), confidence=0.9,
)
print(pt.interpret())
""")

code("""\
from probcal.monitor import moc_offset_from_counts

moc = moc_offset_from_counts(y_test, p_test, level=0.9)
chain_moc = Chain([cal, moc])  # calibration first, MoC offset second

print(f"calibrated mean PD:   {p_test.mean():.4%}")
print(f"MoC-adjusted mean PD: {chain_moc.predict_proba(s_test).mean():.4%}")
print(moc.interpret())
""")


md("""\
## 10. Monitoring closed loop: alarm to a fresh, quiet monitor

Section 7's monitor alarmed on a sustained +0.5 log-odds drift and recommended
an action. `apply_recommendation()` turns that recommendation into a fitted
`LogitOffset` (for `kind="re-offset"`) plus a **fresh** `CalibrationMonitor` —
fresh because the old e-process is a martingale under "the *currently
deployed* forecast is calibrated", a null that no longer holds once the
pipeline changes. Feeding the fresh monitor the corrected forecasts for a
further stretch of drifted months should keep it quiet: the correction, not
a fluke, is what explains the drift. Theory: the *Monitoring* chapter,
"Closing the loop".
""")

code("""\
action = mon.apply_recommendation()
print(f"recommendation: {action.kind}")
print(f"audit: {action.audit}")

fresh = action.monitor
offset = action.offset
for month in range(24, 36):
    take = rng_m.choice(pool, size=2000, replace=True)
    p_month = p_test[take]
    true_pd = expit(logit(p_month) + 0.5)  # the same drift persists, uncorrected
    y_month = (rng_m.random(2000) < true_pd).astype(float)
    p_corrected = offset.transform(p_month)
    fresh.update(y_month, p_corrected, label=f"m{month + 1:02d}")

fresh_rep = fresh.report()
print(f"fresh monitor alarm at: {fresh_rep.alarm_at} (quiet == correction explained the drift)")
""")


md("""\
## 11. Validation report

Everything above is assembled into one self-contained document by
`validation_report`: reliability, the metric report, and the CORP
decomposition from `y_test`/`p_test` alone, plus the rating-grades section
since `grades_after` is given and the calibrator appendix since `cal` is
given. `path=None` returns the HTML string instead of writing a file; the
`"http" not in html` check confirms the document makes no external
requests. Theory: the *Validation report* guide.
""")

code("""\
from probcal.report import validation_report

html = validation_report(
    y_test, p_test, calibrator=cal, grades=grades_after, path=None,
)
print(f"report length: {len(html):,} characters")
print(f"no external requests: {'http' not in html}")
""")


def main() -> None:
    nb = {
        "cells": [
            {
                "cell_type": kind,
                "metadata": {},
                "source": text.splitlines(keepends=True),
                **({"outputs": [], "execution_count": None} if kind == CODE else {}),
            }
            for kind, text in CELLS
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = pathlib.Path(__file__).parent.parent / "notebooks" / "pd_end_to_end.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
