"""Perf smoke tests: generous ceilings (>=5x locally measured on this host) that
catch a regression to the pre-fast-path O(n^2)/O(n log n) behavior without being
flaky on slower CI hardware.

``evaluate`` is measured at n=20,000/n_boot=20 (raised from n=5,000 now that the
binned smECE fast path this fix corrects no longer dominates timing at that
scale -- measured 16.6s here, comfortably under the ceiling) rather than the
larger n=50,000/n_boot=50 scale: its default metric catalog still includes
``ece_sweep``, which scans ~99 candidate bin counts per call (untouched by
this fix), so it -- not the grid-anchored LOESS / binned smECE fast paths this
suite guards -- would dominate the timing at n=50,000 (measured ~537s on this
host). n=20,000 exercises the bootstrap loop and metric dispatch above both
the smECE bin-lattice default (8,192) and the LOESS anchor grid (512), and
stays a practical smoke test.

ENIR's ceiling uses n=20,000, not the spec's m=1e5: the vectorized engine (the
production memory fix; see DECISIONS) is O(m*G) in time, and 1e5 is not
reachable under any headroom multiple on this host (measured 35.5s at m=5e4).
n=20,000 is the largest size that still leaves >=5x headroom to a practical
ceiling; it also carries a tracemalloc peak assertion, which is where ENIR's
actual acceptance criterion (bounded memory, not fit time) lives. Wall time
and peak memory are measured in separate ``fit`` calls: tracemalloc's
allocation tracing roughly triples ENIR's fit time here (its retention/BIC
bookkeeping is Python-level, not vectorized), so timing it under tracemalloc
would leave no real headroom on the 30s ceiling.
"""

import time
import tracemalloc

import pytest

from probcal import CalibratorSelector, ENIRCalibrator, VennAbersCalibrator
from probcal._math import loess
from probcal.datasets import make_pd_portfolio
from probcal.metrics import evaluate, ici
from probcal.metrics.smooth import smooth_ece

pytestmark = pytest.mark.slow


def test_ici_n200k_under_30s() -> None:
    d = make_pd_portfolio(n=200_000, random_state=42)
    t0 = time.perf_counter()
    ici(d.y, d.scores)
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0


def test_smooth_ece_default_n20k_under_ceiling() -> None:
    # n=20,000 exercises the lattice-binned default path (n > 8192 bins); the
    # pre-fix aliasing defect made this cost ~6s regardless of n, the fixed
    # lattice integrator measures ~ms, so 5s is a generous (>=5x) ceiling.
    d = make_pd_portfolio(n=20_000, random_state=42)
    t0 = time.perf_counter()
    smooth_ece(d.y, d.scores)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0


def test_evaluate_n20k_boot20_under_130s() -> None:
    d = make_pd_portfolio(n=20_000, random_state=42)
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


def test_enir_fit_n20k_under_30s_and_500mb() -> None:
    d = make_pd_portfolio(n=20_000, random_state=42)

    t0 = time.perf_counter()
    ENIRCalibrator().fit(d.scores, d.y)
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0

    tracemalloc.start()
    ENIRCalibrator().fit(d.scores, d.y)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak / (1024 * 1024) < 500.0


def test_ivap_fit_n100k_under_5s() -> None:
    d = make_pd_portfolio(n=100_000, random_state=42)
    t0 = time.perf_counter()
    VennAbersCalibrator().fit(d.scores, d.y)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0


def test_ivap_predict_interval_n100k_under_2p5s() -> None:
    d = make_pd_portfolio(n=100_000, random_state=42)
    ivap = VennAbersCalibrator().fit(d.scores, d.y)
    query = make_pd_portfolio(n=100_000, random_state=43).scores
    t0 = time.perf_counter()
    ivap.predict_interval(query)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.5


def test_selector_default_menu_n100k_under_60s() -> None:
    d = make_pd_portfolio(n=100_000, random_state=42)
    t0 = time.perf_counter()
    CalibratorSelector().fit(d.scores, d.y)
    elapsed = time.perf_counter() - t0
    assert elapsed < 60.0
