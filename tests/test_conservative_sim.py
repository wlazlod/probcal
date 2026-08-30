"""Slow coverage-simulation gate for probcal.metrics.pluto_tasche.

The full-size table (2000 runs) is produced by
``docs/scripts/conservative_sim.py`` and pasted into
``docs/concepts/conservatism.md``; this suite enforces the same gate at a
reduced run count, with the tolerance widened accordingly.
"""

import importlib.util
import pathlib

import numpy as np
import pytest

pytestmark = pytest.mark.slow

_SPEC = importlib.util.spec_from_file_location(
    "conservative_sim",
    pathlib.Path(__file__).parent.parent / "docs" / "scripts" / "conservative_sim.py",
)
sim = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sim)


@pytest.mark.parametrize("confidence", [0.9, 0.95])
def test_coverage_gate_reduced(confidence: float) -> None:
    runs = 300
    res = sim.coverage(runs, sim.N_PER_GRADE, sim.PD_TRUE, confidence=confidence)
    bound = confidence - 2.0 * np.sqrt(confidence * (1.0 - confidence) / runs)
    assert res["min_per_grade"] >= bound, (confidence, res["min_per_grade"], bound)
