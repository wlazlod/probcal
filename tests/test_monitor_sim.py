"""Slow simulation gates for probcal.monitor (reduced sizes).

The full-size table (2000 runs) is produced by ``docs/scripts/monitor_sim.py``
and pasted into ``docs/concepts/monitoring.md``; this suite enforces the same
gates at reduced run counts with tolerances widened accordingly
(``alpha + 2*sqrt(alpha*(1-alpha)/runs)``), plus a cross-check that the
vectorized fleet replay agrees with the shipped ``CalibrationMonitor`` — the
simulations verify the shipped math, not a parallel implementation.
"""

import importlib.util
import pathlib

import numpy as np
import pytest

from probcal._math import expit, logit
from probcal.monitor import CalibrationMonitor

pytestmark = pytest.mark.slow

_SPEC = importlib.util.spec_from_file_location(
    "monitor_sim", pathlib.Path(__file__).parent.parent / "docs" / "scripts" / "monitor_sim.py"
)
sim = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sim)


def test_fleet_replay_matches_calibration_monitor() -> None:
    # Same batches through the vectorized fleet and the shipped class: the
    # e-process trajectories must agree (plug-ins differ only by the fleet's
    # tabulated-inverse delta and aggregated-count IRLS -> tight tolerance).
    rng = np.random.default_rng(9)
    z = logit(sim._scores(800, seed=21))
    fleet = sim.FleetReplay(3, z)
    monitors = [CalibrationMonitor(delta_ci_grid=(-2.0, 2.0, 21)) for _ in range(3)]
    p_true = expit(z + 0.35)
    for k in range(6):
        y = sim.draw_outcomes(rng, p_true, 3)
        out = fleet.update(y)
        for r, mon in enumerate(monitors):
            step = mon.update(y[r], expit(z), label=f"b{k}")
            np.testing.assert_allclose(np.exp(out["global"][r]), step.e_global, rtol=1e-3)


def test_type1_gate_reduced() -> None:
    runs = 300
    res = sim.sim_type1(runs=runs, batches=24, n=2000, seed=1)
    for alpha in (0.05, 0.01):
        bound = alpha + 2.0 * np.sqrt(alpha * (1.0 - alpha) / runs)
        for comp in ("offset", "shape", "global"):
            assert res[alpha][comp] <= bound, (alpha, comp, res[alpha][comp], bound)


def test_type1_grades_and_hetero_reduced() -> None:
    runs = 300
    for res in (
        sim.sim_type1_grades(runs=runs, batches=24, n=2000, seed=3),
        sim.sim_type1_hetero(runs=runs, batches=24, seed=2),
    ):
        for alpha in (0.05, 0.01):
            bound = alpha + 2.0 * np.sqrt(alpha * (1.0 - alpha) / runs)
            assert res[alpha]["global"] <= bound


def test_power_gates_reduced() -> None:
    strong = sim.sim_power(shift=0.4, runs=120, onset=12)
    assert strong["detect_rate"] >= 0.9
    assert strong["median_delay"] <= 6.0
    slope = sim.sim_power(slope=0.8, runs=120, onset=12)
    assert slope["detect_rate"] >= 0.9
    assert slope["median_delay"] <= 12.0


def test_cs_time_uniform_coverage_reduced() -> None:
    assert sim.sim_cs_coverage(runs=150, true_delta=0.0) >= 0.95
    assert sim.sim_cs_coverage(runs=150, true_delta=0.4, seed=6) >= 0.95


def _recommendation_run(shift: float, slope: float, seed: int) -> str:
    mon = CalibrationMonitor(delta_ci_grid=(-2.0, 2.0, 41))
    z = logit(sim._scores(2000, seed=42))
    p = expit(z)
    rng = np.random.default_rng(seed)
    p_true = expit(slope * z + shift)
    for k in range(10):
        y = (rng.random(2000) < p_true).astype(float)
        mon.update(y, p, label=f"m{k}")
    return mon.report().recommendation


def test_recommendation_correct_on_pure_drift() -> None:
    n_runs = 20
    offset_ok = sum(
        _recommendation_run(shift=0.4, slope=1.0, seed=1000 + r) == "re-offset"
        for r in range(n_runs)
    )
    slope_ok = sum(
        _recommendation_run(shift=0.0, slope=0.8, seed=2000 + r) == "re-fit" for r in range(n_runs)
    )
    assert offset_ok >= int(0.9 * n_runs), offset_ok
    assert slope_ok >= int(0.9 * n_runs), slope_ok
