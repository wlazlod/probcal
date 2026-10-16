"""Tests for probcal.monitor._onset and the since-onset recommendation window."""

import importlib.util
import pathlib
from dataclasses import replace

import numpy as np
import pytest

from probcal._math import bern_log_lr, expit, logit
from probcal.datasets import make_pd_portfolio
from probcal.monitor import CalibrationMonitor
from probcal.monitor._onset import estimate_onset

_SPEC = importlib.util.spec_from_file_location(
    "monitor_sim", pathlib.Path(__file__).parent.parent / "docs" / "scripts" / "monitor_sim.py"
)
sim = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sim)


def _batch(n=2000, shift=0.0, slope=1.0, seed=0, event_rate=0.05):
    d = make_pd_portfolio(n=n, event_rate=event_rate, random_state=seed)
    p = d.scores
    true = expit(slope * logit(p) + shift)
    rng = np.random.default_rng(seed + 1000)
    y = (rng.random(n) < true).astype(float)
    return y, p


# ---------------------------------------------------------------- estimate_onset (unit)


@pytest.mark.parametrize(
    ("increments", "expected"),
    [
        ([-0.1, -0.2, -0.15, 5.0, 4.5, 4.8], 3),  # backward-CUSUM argmax
        ([3.0], 0),
        ([0.0, 3.0, 2.0], 0),  # ties resolve to the earliest batch
        ([-1.0, 0.0, 3.0], 1),
        ([0.0, 0.0, 0.0], 0),  # no positive evidence: the whole history
        ([-1.0, -0.5], 0),
    ],
)
def test_estimate_onset(increments: list[float], expected: int) -> None:
    assert estimate_onset(np.array(increments)) == expected


def test_estimate_onset_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        estimate_onset(np.array([]))


def test_onset_increment_sums_only_active_components() -> None:
    # MON-10: the onset series is the plug-in log-LR of the monitored components.
    mon = CalibrationMonitor(components=("offset",))
    for k in range(5):
        y, p = _batch(n=1000, shift=0.5, seed=60 + k)
        step = mon.update(y, p, label=f"m{k}")
        d = step.delta_hat
        expected = 0.0 if d == 0.0 else bern_log_lr(y, p, expit(logit(p) + d), np.ones_like(p))
        assert step.log_e_increment == pytest.approx(expected, abs=1e-12)


def _drift_from_start(**kw: object) -> CalibrationMonitor:
    mon = CalibrationMonitor(**kw)  # type: ignore[arg-type]
    for k in range(6):
        y, p = _batch(shift=0.6, seed=40 + k)
        mon.update(y, p, label=f"m{k}")
    return mon


def test_onset_never_after_alarm() -> None:
    mon = _drift_from_start()
    rep = mon.report()
    labels = [s.label for s in mon.steps_]
    assert rep.alarm_at is not None and rep.onset_label is not None
    assert labels.index(rep.onset_label) <= labels.index(rep.alarm_at)
    assert any("estimated drift onset" in r for r in rep.reasoning)


def test_no_plug_in_evidence_falls_back_to_whole_history() -> None:
    # min_history so large the plug-ins never engage: every increment is 0
    # and the alarm comes from the fixed-grid mixture alone.
    mon = _drift_from_start(min_history=100, components=("offset",))
    assert all(s.log_e_increment == 0.0 for s in mon.steps_)
    rep = mon.report()
    assert rep.alarm_at == "m0"
    assert rep.onset_label == "m0"
    action = mon.apply_recommendation()
    assert action.window == tuple(s.label for s in mon.steps_)
    assert action.audit["onset_label"] == rep.onset_label


# ---------------------------------------------------------------- recommendation_window


def test_since_onset_window_respects_plug_in_window() -> None:
    # plug_in_window bounds the since-onset window: an onset estimated far
    # earlier than the trailing window must not widen the window used by
    # delta_now/_slope_ci/_residual_shape_lr beyond plug_in_window batches.
    mon = CalibrationMonitor(alpha=0.05, plug_in_window=3, recommendation_window="since_onset")
    mon_trailing = CalibrationMonitor(
        alpha=0.05, plug_in_window=3, recommendation_window="trailing"
    )
    for k in range(10):
        y, p = _batch(shift=0.8, seed=k)
        mon.update(y, p, label=f"m{k}")
        mon_trailing.update(y, p, label=f"m{k}")
    assert mon.report().alarm_at is not None  # both monitors saw identical batches

    # Force the onset estimate to batch 2 of 10 (index 2), well before the
    # plug_in_window=3 trailing start (index 7 = 10 - 3).
    forced = [-1.0, -1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    assert estimate_onset(np.array(forced)) == 2
    # The onset is read up to the first alarm; place that alarm after it.
    mon.steps_ = [
        replace(s, log_e_increment=v, alarm=i >= 5)
        for i, (s, v) in enumerate(zip(mon.steps_, forced, strict=True))
    ]

    mon_trailing.steps_ = [replace(s, alarm=i >= 5) for i, s in enumerate(mon_trailing.steps_)]

    rep = mon.report()
    assert rep.onset_label == "m2"  # the onset estimate itself is unaffected
    rep_trailing = mon_trailing.report()

    # start = max(onset_idx=2, n_batches=10 - plug_in_window=3) = 7, i.e. the
    # last 3 batches -- exactly the "trailing" window with plug_in_window=3 --
    # so every non-onset reasoning number coincides bit-for-bit.
    since_numbers = rep.reasoning[:4] + rep.reasoning[5:]
    trailing_numbers = rep_trailing.reasoning[:4] + rep_trailing.reasoning[5:]
    assert since_numbers == trailing_numbers


# ---------------------------------------------------------------- injected-drift localization


@pytest.mark.slow
def test_onset_localizes_injected_drift() -> None:
    n_runs = 40
    onset_idx = 12
    z = logit(sim._scores(2000, seed=42))
    p = expit(z)
    errors = []
    for seed in range(n_runs):
        mon = CalibrationMonitor(delta_ci_grid=(-2.0, 2.0, 41))
        rng = np.random.default_rng(seed)
        for k in range(24):
            p_true = expit(z + 0.6) if k >= onset_idx else p
            y = (rng.random(2000) < p_true).astype(float)
            mon.update(y, p, label=f"m{k}")
        rep = mon.report()
        assert rep.alarm_at is not None, seed
        errors.append(abs(int(rep.onset_label[1:]) - onset_idx))
    median_err = float(np.median(errors))
    assert median_err <= 2.0, (median_err, sorted(errors))
