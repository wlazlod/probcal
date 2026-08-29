"""Slow recovery-simulation gates for SegmentedCalibrator (empirical-Bayes shrinkage).

The full-size table (2000 runs) is produced by
``docs/scripts/segmented_sim.py`` and pasted into
``docs/concepts/segmented.md``; this suite enforces the same gates at a
reduced run count in CI.
"""

import importlib.util
import pathlib

import numpy as np
import pytest

pytestmark = pytest.mark.slow

_SPEC = importlib.util.spec_from_file_location(
    "segmented_sim",
    pathlib.Path(__file__).parent.parent / "docs" / "scripts" / "segmented_sim.py",
)
sim = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sim)


def test_eb_mse_beats_no_pooling_and_complete_pooling_reduced() -> None:
    res = sim.recovery(100, sim.N_PER_SEGMENT, sim.TRUE_DELTAS, seed=42)
    assert res["mean_mse_eb"] <= res["mean_mse_no_pooling"]
    assert res["mean_mse_eb"] <= res["mean_mse_complete_pooling"]


def test_eb_degrades_to_complete_pooling_as_spread_shrinks_to_zero() -> None:
    res = sim.recovery(100, (3000,) * 6, (0.0,) * 6, seed=123)
    assert np.max(np.abs(res["mean_delta_tilde"])) < 0.05
