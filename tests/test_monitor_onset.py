"""Tests for probcal.monitor._onset and the since-onset recommendation window (spec M3)."""

import importlib.util
import pathlib
from dataclasses import replace

import numpy as np
import pytest

from probcal._math import expit, logit
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


def test_estimate_onset_backward_cusum_argmax() -> None:
    inc = np.array([-0.1, -0.2, -0.15, 5.0, 4.5, 4.8])
    assert estimate_onset(inc) == 3


def test_estimate_onset_ties_break_to_latest() -> None:
    assert estimate_onset(np.zeros(5)) == 4


def test_estimate_onset_single_element() -> None:
    assert estimate_onset(np.array([3.0])) == 0


def test_estimate_onset_all_negative_picks_last_index() -> None:
    # Every suffix sum is negative; the least-negative (smallest loss) is the
    # final singleton, since dropping any earlier term only adds more loss.
    inc = np.array([-1.0, -2.0, -0.5])
    assert estimate_onset(inc) == 2


def test_estimate_onset_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        estimate_onset(np.array([]))


# ---------------------------------------------------------------- onset_label wiring


def test_no_onset_under_null() -> None:
    mon = CalibrationMonitor(alpha=0.05)
    for k in range(12):
        y, p = _batch(shift=0.0, seed=500 + k)
        mon.update(y, p, label=f"m{k}")
    rep = mon.report()
    assert rep.alarm_at is None
    assert rep.onset_label is None


def test_onset_label_present_after_alarm() -> None:
    mon = CalibrationMonitor(alpha=0.05)
    for k in range(8):
        y, p = _batch(shift=0.8, seed=k)
        mon.update(y, p, label=f"m{k}")
    rep = mon.report()
    assert rep.alarm_at is not None
    assert rep.onset_label is not None
    assert any("estimated drift onset" in r for r in rep.reasoning)


def test_log_e_increment_round_trips_through_json(tmp_path) -> None:
    mon = CalibrationMonitor()
    for k in range(3):
        y, p = _batch(seed=10 + k)
        mon.update(y, p, label=f"m{k}")
    path = tmp_path / "state.json"
    mon.to_json(path)
    mon2 = CalibrationMonitor.from_json(path)
    assert [s.log_e_increment for s in mon.steps_] == [s.log_e_increment for s in mon2.steps_]


# ---------------------------------------------------------------- recommendation_window


def test_recommendation_window_validation() -> None:
    with pytest.raises(ValueError, match="recommendation_window"):
        CalibrationMonitor(recommendation_window="nope")


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
    mon.steps_ = [replace(s, log_e_increment=v) for s, v in zip(mon.steps_, forced, strict=True)]

    rep = mon.report()
    assert rep.onset_label == "m2"  # the onset estimate itself is unaffected
    rep_trailing = mon_trailing.report()

    # start = max(onset_idx=2, n_batches=10 - plug_in_window=3) = 7, i.e. the
    # last 3 batches -- exactly the "trailing" window with plug_in_window=3 --
    # so every non-onset reasoning number coincides bit-for-bit.
    since_numbers = rep.reasoning[:4] + rep.reasoning[5:]
    trailing_numbers = rep_trailing.reasoning[:4] + rep_trailing.reasoning[5:]
    assert since_numbers == trailing_numbers


def test_trailing_window_matches_since_onset_when_onset_is_batch_zero() -> None:
    # A single batch, drifted hard enough to alarm on the first look: the
    # first batch's plug-in is always the identity (no strictly-earlier
    # data), so its increment is exactly 0.0 and it is trivially the only
    # step — onset_idx == 0 by construction, and "since onset" and
    # "trailing" (all past batches) select the identical single-batch
    # window, so the recommendation numbers must coincide exactly.
    def run(window: str):
        mon = CalibrationMonitor(alpha=0.05, recommendation_window=window)
        y, p = _batch(n=2000, shift=1.2, seed=0)
        mon.update(y, p, label="m0")
        return mon.report()

    rep_since = run("since_onset")
    rep_trailing = run("trailing")
    assert rep_since.alarm_at == "m0"
    assert rep_since.onset_label == "m0"
    assert rep_since.reasoning == rep_trailing.reasoning


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
