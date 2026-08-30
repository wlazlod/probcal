"""Slow coverage-simulation gate for probcal.curves.corp_reliability bands.

The full-size table (500 runs, n in {1000, 5000}, level in {0.8, 0.9}) is
produced by ``docs/scripts/corp_sim.py`` and pasted into
``docs/concepts/corp.md``; this suite enforces the pointwise gate at a
reduced size. Pointwise coverage at level 0.9 should be close to 0.9;
uniform coverage is expected to be lower (bands are pointwise, not
simultaneous) and is reported, not gated.
"""

import importlib.util
import pathlib

import pytest

pytestmark = pytest.mark.slow

_SPEC = importlib.util.spec_from_file_location(
    "corp_sim", pathlib.Path(__file__).parent.parent / "docs" / "scripts" / "corp_sim.py"
)
sim = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sim)


def test_consistency_band_pointwise_coverage_near_nominal() -> None:
    out = sim.coverage(n=1000, runs=60, n_resamples=100, level=0.9, seed=0)
    assert out["pointwise_coverage"] >= 0.85
