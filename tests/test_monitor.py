"""Fast tests for probcal.monitor; simulations live in test_monitor_sim.py."""

import json

import numpy as np
import pytest

from probcal._math import expit, logit
from probcal.datasets import make_pd_portfolio
from probcal.monitor import CalibrationMonitor, MonitorReport, MonitorStep

RNG = np.random.default_rng(17)


def _batch(n=2000, shift=0.0, slope=1.0, seed=0, event_rate=0.05):
    d = make_pd_portfolio(n=n, event_rate=event_rate, random_state=seed)
    p = d.scores
    true = expit(slope * logit(p) + shift)
    rng = np.random.default_rng(seed + 1000)
    y = (rng.random(n) < true).astype(float)
    return y, p


def test_first_batch_plug_ins_are_identity() -> None:
    mon = CalibrationMonitor()
    y, p = _batch(seed=3)
    step = mon.update(y, p, label="b1")
    assert isinstance(step, MonitorStep)
    assert step.delta_hat == 0.0
    assert step.slope_hat == 1.0
    assert step.e_shape == 1.0  # identity plug-in -> every factor is exactly 1
    assert step.n == 2000 and step.label == "b1"


def test_mixture_hand_computed_on_tiny_batch() -> None:
    # Two observations at p = 0.5, y = (1, 0). For grid point d the batch
    # factor is q(1-q)/0.25 with q = sigma(d); the plug-in factor is 1.
    mon = CalibrationMonitor(mixture_grid=(0.5, 1.0))
    step = mon.update(np.array([1.0, 0.0]), np.array([0.5, 0.5]), label="t")
    deltas = np.array([-1.0, -0.5, 0.5, 1.0])
    q = expit(deltas)
    e_mix = float(np.mean(q * (1 - q) / 0.25))
    expected_off = 0.5 * (1.0 + e_mix)
    assert step.e_offset == pytest.approx(expected_off, rel=1e-12)


def test_alarm_and_reoffset_recommendation_under_level_drift() -> None:
    mon = CalibrationMonitor(alpha=0.05)
    fired_at = None
    for k in range(8):
        y, p = _batch(shift=0.8, seed=k)
        step = mon.update(y, p, label=f"m{k}")
        if step.alarm and fired_at is None:
            fired_at = k
    assert fired_at is not None and fired_at <= 4
    rep = mon.report()
    assert rep.alarm_at == f"m{fired_at}"
    assert rep.recommendation == "re-offset"
    assert any("shape" in r for r in rep.reasoning)
    lo, hi = mon.steps_[-1].delta_ci
    assert lo <= 0.8 <= hi  # CS covers the injected offset
    assert lo > 0.0  # and excludes zero after sustained drift


def test_refit_recommendation_under_slope_drift() -> None:
    mon = CalibrationMonitor(alpha=0.05)
    for k in range(12):
        y, p = _batch(shift=0.0, slope=0.55, seed=100 + k)
        mon.update(y, p, label=f"m{k}")
    rep = mon.report()
    assert rep.alarm_at is not None
    assert rep.recommendation == "re-fit"


def test_no_alarm_on_calibrated_stream() -> None:
    mon = CalibrationMonitor(alpha=0.05)
    for k in range(12):
        y, p = _batch(shift=0.0, seed=500 + k)
        step = mon.update(y, p, label=f"m{k}")
    assert step.alarm is False
    assert mon.report().recommendation == "none"
    assert step.p_anytime > 0.05


def test_grade_component_catches_single_grade_drift() -> None:
    mon = CalibrationMonitor(alpha=0.05)
    for k in range(8):
        y_a, p_a = _batch(n=1000, shift=1.2, seed=200 + k)
        y_b, p_b = _batch(n=1000, shift=0.0, seed=300 + k)
        y = np.concatenate([y_a, y_b])
        p = np.concatenate([p_a, p_b])
        g = np.array(["A"] * 1000 + ["B"] * 1000)
        step = mon.update(y, p, grade=g, label=f"m{k}")
    assert set(step.e_grades) == {"A", "B"}
    assert step.e_grades["A"] > step.e_grades["B"]
    assert "A" in mon.report().grade_table


def test_weights_warn_once_and_enter_as_exponents() -> None:
    mon = CalibrationMonitor()
    y, p = _batch(seed=7)
    w = np.ones_like(p)
    w[:10] = 2.5
    with pytest.warns(UserWarning, match="martingale"):
        mon.update(y, p, sample_weight=w, label="w1")
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as rec:
        _warnings.simplefilter("always")
        mon.update(y, p, sample_weight=w, label="w2")
    assert not any("martingale" in str(r.message) for r in rec)  # warned once


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="components"):
        CalibrationMonitor(components=("offset", "nope"))
    with pytest.raises(ValueError, match="alpha"):
        CalibrationMonitor(alpha=1.5)
    mon = CalibrationMonitor()
    with pytest.raises(ValueError, match="0, 1"):
        mon.update(np.array([0.0, 2.0]), np.array([0.1, 0.2]))
    # a single-class batch is legal (a quiet month has no defaults)
    mon.update(np.zeros(50), np.full(50, 0.02), label="quiet")


def test_predictability_future_batches_do_not_change_past() -> None:
    def run(later_shift: float) -> list[float]:
        mon = CalibrationMonitor()
        out = []
        for k in range(6):
            shift = 0.0 if k < 3 else later_shift
            y, p = _batch(shift=shift, seed=40 + k)
            out.append(mon.update(y, p, label=f"m{k}").e_global)
        return out

    a = run(0.0)
    b = run(2.0)
    assert a[:3] == b[:3]  # bit-identical prefixes
    assert a[3:] != b[3:]


def test_persistence_reproduces_trajectory_bit_for_bit(tmp_path) -> None:
    batches = [_batch(shift=0.3, seed=60 + k) for k in range(5)]
    mon = CalibrationMonitor()
    direct = [mon.update(y, p, label=f"m{k}").e_global for k, (y, p) in enumerate(batches)]

    resumed: list[float] = []
    mon2 = CalibrationMonitor()
    for k, (y, p) in enumerate(batches):
        resumed.append(mon2.update(y, p, label=f"m{k}").e_global)
        path = tmp_path / f"state{k}.json"
        mon2.to_json(path)
        mon2 = CalibrationMonitor.from_json(path)
    assert direct == resumed
    rep = mon2.report()
    assert isinstance(rep, MonitorReport)
    assert [s.e_global for s in rep.steps] == direct


def test_delayed_labels_are_opaque() -> None:
    batches = [_batch(seed=80 + k) for k in range(3)]
    lab_order = ["2026Q3", "2026Q1", "2026Q2"]  # out of calendar order
    mon_a = CalibrationMonitor()
    mon_b = CalibrationMonitor()
    pairs = zip(batches, lab_order, strict=True)
    ea = [mon_a.update(y, p, label=lab).e_global for (y, p), lab in pairs]
    eb = [mon_b.update(y, p, label=f"x{k}").e_global for k, (y, p) in enumerate(batches)]
    assert ea == eb


def test_to_frame_is_list_of_dicts_or_frame() -> None:
    mon = CalibrationMonitor()
    for k in range(3):
        y, p = _batch(seed=90 + k)
        mon.update(y, p, label=f"m{k}")
    frame = mon.report().to_frame()
    try:
        import pandas as pd

        assert isinstance(frame, pd.DataFrame) and len(frame) == 3
    except ImportError:
        assert isinstance(frame, list) and len(frame) == 3 and isinstance(frame[0], dict)


def test_monitor_serialization_registered() -> None:
    from probcal._registry import SERIALIZABLE

    assert "CalibrationMonitor" in SERIALIZABLE
    mon = CalibrationMonitor()
    y, p = _batch(seed=95)
    mon.update(y, p, label="m0")
    d = json.loads(mon.to_json())
    assert d["class"] == "CalibrationMonitor" and d["probcal_schema"] == 1
