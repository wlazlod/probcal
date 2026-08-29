"""Per-grade offset confidence sequences (spec M2); slow power sim excluded from -m "not slow"."""

import pathlib

import numpy as np
import pytest

from probcal._math import expit, logit
from probcal.datasets import make_pd_portfolio
from probcal.monitor import CalibrationMonitor

_DATA_DIR = pathlib.Path(__file__).parent / "data"


def _grade_batch(
    n: int = 1500, shift_a: float = 0.0, shift_b: float = 0.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d_a = make_pd_portfolio(n=n, random_state=seed)
    d_b = make_pd_portfolio(n=n, random_state=seed + 1)
    p = np.concatenate([d_a.scores, d_b.scores])
    true = np.concatenate([expit(logit(d_a.scores) + shift_a), expit(logit(d_b.scores) + shift_b)])
    rng = np.random.default_rng(seed + 5000)
    y = (rng.random(len(p)) < true).astype(float)
    g = np.array(["A"] * n + ["B"] * n)
    return y, p, g


def test_grade_delta_ci_empty_without_grades() -> None:
    mon = CalibrationMonitor()
    y, p, _ = _grade_batch(n=200, seed=1)
    step = mon.update(y, p, label="m0")
    assert step.grade_delta_ci == {}


def test_grade_delta_ci_populated_and_shrinks_toward_injected_offset() -> None:
    mon = CalibrationMonitor(alpha=0.05, delta_ci_grid=(-2.0, 2.0, 81))
    step = None
    for k in range(6):
        y, p, g = _grade_batch(n=1500, shift_a=0.6, shift_b=0.0, seed=10 + k)
        step = mon.update(y, p, grade=g, label=f"m{k}")
    assert set(step.grade_delta_ci) == {"A", "B"}
    lo_a, hi_a = step.grade_delta_ci["A"]
    lo_b, hi_b = step.grade_delta_ci["B"]
    assert lo_a <= 0.6 <= hi_a
    assert lo_b <= 0.0 <= hi_b
    assert lo_a > 0.0  # sustained drift on A eventually excludes zero


@pytest.mark.slow
def test_two_grade_drift_confidence_sequence_coverage() -> None:
    n_runs = 20
    hits_a = hits_b = 0
    for r in range(n_runs):
        mon = CalibrationMonitor(alpha=0.05, delta_ci_grid=(-2.0, 2.0, 81))
        step = None
        for k in range(6):
            y, p, g = _grade_batch(n=1500, shift_a=0.6, shift_b=0.0, seed=1000 * r + k)
            step = mon.update(y, p, grade=g, label=f"m{k}")
        lo_a, hi_a = step.grade_delta_ci["A"]
        lo_b, hi_b = step.grade_delta_ci["B"]
        hits_a += lo_a <= 0.6 <= hi_a
        hits_b += lo_b <= 0.0 <= hi_b
    assert hits_a / n_runs >= 0.9
    assert hits_b / n_runs >= 0.9


def test_persistence_reproduces_grade_ci_bit_for_bit(tmp_path) -> None:
    batches = [_grade_batch(n=300, shift_a=0.3, shift_b=0.0, seed=60 + k) for k in range(5)]
    mon = CalibrationMonitor()
    direct = [
        mon.update(y, p, grade=g, label=f"m{k}").grade_delta_ci
        for k, (y, p, g) in enumerate(batches)
    ]

    resumed = []
    mon2 = CalibrationMonitor()
    for k, (y, p, g) in enumerate(batches):
        resumed.append(mon2.update(y, p, grade=g, label=f"m{k}").grade_delta_ci)
        path = tmp_path / f"state{k}.json"
        mon2.to_json(path)
        mon2 = CalibrationMonitor.from_json(path)
    assert direct == resumed


def test_0_2_0_monitor_file_loads_and_continues() -> None:
    path = _DATA_DIR / "monitor_0_2_0.json"
    mon = CalibrationMonitor.from_json(path)
    assert len(mon.steps_) == 3
    assert all(s.grade_delta_ci == {} for s in mon.steps_)

    for k in range(2):
        y, p, g = _grade_batch(n=300, shift_a=0.5, shift_b=0.0, seed=900 + k)
        step = mon.update(y, p, grade=g, label=f"new{k}")
        assert set(step.grade_delta_ci) == {"A", "B"}

    assert mon.steps_[0].grade_delta_ci == {}
    assert mon.steps_[-1].grade_delta_ci["A"] is None or isinstance(
        mon.steps_[-1].grade_delta_ci["A"], tuple
    )

    d = mon.to_dict()
    mon2 = CalibrationMonitor.from_dict(d)
    assert mon2.to_dict() == d


def test_plot_e_process_grades_panel_adds_second_axes() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from probcal.plots import plot_e_process

    mon = CalibrationMonitor(delta_ci_grid=(-2.0, 2.0, 21))
    for k in range(3):
        y, p, g = _grade_batch(n=200, shift_a=0.4, shift_b=0.0, seed=2000 + k)
        mon.update(y, p, grade=g, label=f"m{k}")
    rep = mon.report()

    ax_default = plot_e_process(rep)
    assert len(ax_default.figure.axes) == 1

    ax_panel = plot_e_process(rep, grades_panel=True)
    assert len(ax_panel.figure.axes) == 2
