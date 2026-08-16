"""Perf smoke tests: generous ceilings (>=5x locally measured on this host) that
catch a regression to the pre-fast-path O(n^2)/O(n log n) behavior without being
flaky on slower CI hardware.

``evaluate`` is measured at n=5,000/n_boot=20 rather than the larger n=50,000/
n_boot=50 scale: its default metric catalog includes ``ece_sweep``, which scans
~99 candidate bin counts per call (untouched by this perf pass), so it -- not
the grid-anchored LOESS / binned smECE fast paths this suite guards -- would
dominate the timing at n=50,000 (measured ~537s on this host). The smaller
scale still exercises the same bootstrap loop and metric dispatch and stays a
practical smoke test.
"""

import time

import pytest

from probcal._math import loess
from probcal.datasets import make_pd_portfolio
from probcal.metrics import evaluate, ici

pytestmark = pytest.mark.slow


def test_ici_n200k_under_30s() -> None:
    d = make_pd_portfolio(n=200_000, random_state=42)
    t0 = time.perf_counter()
    ici(d.y, d.scores)
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0


def test_evaluate_n5k_boot20_under_130s() -> None:
    d = make_pd_portfolio(n=5_000, random_state=42)
    t0 = time.perf_counter()
    evaluate(d.y, d.scores, n_boot=20)
    elapsed = time.perf_counter() - t0
    assert elapsed < 130.0


def test_loess_grid512_n1m_under_150s() -> None:
    d = make_pd_portfolio(n=1_000_000, random_state=42)
    t0 = time.perf_counter()
    loess(d.scores, d.y, grid_size=512)
    elapsed = time.perf_counter() - t0
    assert elapsed < 150.0
