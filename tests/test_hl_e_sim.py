"""Slow type-I/power simulation gates for probcal.metrics.hl_e_test (spec M1).

The full-size table (2000 runs) is produced by ``docs/scripts/hl_e_sim.py``
and pasted into ``docs/concepts/monitoring.md``; this suite enforces the
type-I gates at a reduced run count, with the tolerance widened accordingly
(the same ``alpha + 2*sqrt(alpha*(1-alpha)/runs)`` pattern as
``tests/test_monitor_sim.py``), plus a power smoke floor.
"""

import importlib.util
import pathlib

import numpy as np
import pytest

pytestmark = pytest.mark.slow

_SPEC = importlib.util.spec_from_file_location(
    "hl_e_sim", pathlib.Path(__file__).parent.parent / "docs" / "scripts" / "hl_e_sim.py"
)
sim = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sim)


def test_type1_gate_reduced() -> None:
    runs = 300
    res = sim.type1(runs=runs, n=2000, seed=42)
    bound_05 = 0.05 + 2.0 * np.sqrt(0.05 * 0.95 / runs)
    bound_01 = 0.01 + 2.0 * np.sqrt(0.01 * 0.99 / runs)
    assert res["p_ge_20"] <= bound_05, (res["p_ge_20"], bound_05)
    assert res["p_ge_100"] <= bound_01, (res["p_ge_100"], bound_01)
    assert res["mean_e"] <= 1.0 + 3.0 * res["se_mean"], (res["mean_e"], res["se_mean"])


def test_power_smoke_floor() -> None:
    # Not a calibrated power gate -- just a floor confirming the test has
    # visible power against an obvious level shift, at the reduced run count
    # but full n=2000 sample size.
    res = sim.power(shift=0.4, runs=300, n=2000, seed=43)
    assert res["detect_rate"] >= 0.5, res["detect_rate"]
