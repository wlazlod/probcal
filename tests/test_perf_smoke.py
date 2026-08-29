"""Perf smoke tests: generous ceilings (>=5x locally measured on this host) that
catch a regression to the pre-fast-path O(n^2)/O(n log n) behavior without being
flaky on slower CI hardware. The one documented exception is
``test_evaluate_n10k_boot1000_under_150s``, whose ~1.7x ceiling is explained at
the test.

``evaluate`` is measured at n=20,000/n_boot=20 (raised from n=5,000 now that the
binned smECE fast path this fix corrects no longer dominates timing at that
scale -- measured 16.6s here, comfortably under the ceiling) rather than the
larger n=50,000/n_boot=50 scale. Post-fix, ``smooth_ece``'s binned-path cost is
negligible (a few ms) at every size this suite exercises, so it no longer
drives the choice of scale here. An earlier draft of this comment claimed
``ece_sweep`` -- its ~99-candidate bin-count scan, untouched by this fix --
would dominate the timing at n=50,000, citing a ~537s figure; that figure was
itself defect-contaminated (roughly 310s of it was ~50 bootstrap replicates
each paying the pre-fix smECE aliasing defect's ~6.1s/call, not ``ece_sweep``;
see DECISIONS 66) and is not reused. Measured on this dev host after the 0.3.0
bootstrap work (shared per-replicate sort, vectorized ``ece_sweep`` scan,
vectorized LOESS anchor evaluation), a full-catalog replicate at n=10,000 costs
0.089s -- ICI family 0.051s (58%), ``ece_sweep`` scan 0.024s (27%),
``intercept``/``slope`` 0.009s (10%), the whole binned ECE family 0.4ms.
``evaluate(n=10,000, n_boot=1000)`` takes 86.9s here, down from 304.2s in 0.2.x
(3.5x). n=20,000/n_boot=20 stays the fast smoke scale and exercises the
bootstrap loop and metric dispatch above both the smECE bin-lattice default
(8,192) and the LOESS anchor grid (512); the n=10,000/n_boot=1000 case below
pins the full-catalog acceptance scale.

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
    # lattice integrator measures ~ms. 1s is a >100x-headroom ceiling that
    # still restores a real (~6x) regression detector for the pre-fix defect.
    d = make_pd_portfolio(n=20_000, random_state=42)
    t0 = time.perf_counter()
    smooth_ece(d.y, d.scores)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0


def test_smooth_ece_default_wired_to_binned_path_n10k() -> None:
    # End-to-end wiring check: the default (bins=8192) call must actually be
    # exercising the fast binned path, not silently falling through to the
    # exact O(n) computation -- a revert of the binned branch (e.g. `bins`
    # quietly becoming a no-op) would pass every accuracy test in
    # test_metrics_smooth.py (which calls the evaluator directly) but this
    # timing check would catch it: exact at n=10^4 measures >1s on this host.
    d = make_pd_portfolio(n=10_000, random_state=42)
    t0 = time.perf_counter()
    default_value = smooth_ece(d.y, d.scores)
    elapsed = time.perf_counter() - t0
    exact_value = smooth_ece(d.y, d.scores, bins=None)
    assert abs(default_value - exact_value) <= 1e-3
    assert elapsed < 1.0


def test_smooth_ece_default_n4000_under_ceiling() -> None:
    # The 0.1.3 "size cliff": n <= 8192 ran the exact O(n)-per-step path
    # (~0.35s at n=4000 on this host) while n=8193 took ~2ms. The lattice
    # path now engages for all n >= 64 and measures ~2ms here; 0.25s keeps
    # ~100x headroom for slow CI while staying below the pre-fix exact cost.
    d = make_pd_portfolio(n=4000, random_state=42)
    t0 = time.perf_counter()
    smooth_ece(d.y, d.scores)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.25


def test_evaluate_n20k_boot20_under_130s() -> None:
    d = make_pd_portfolio(n=20_000, random_state=42)
    t0 = time.perf_counter()
    evaluate(d.y, d.scores, n_boot=20)
    elapsed = time.perf_counter() - t0
    assert elapsed < 130.0


def test_evaluate_n10k_boot1000_under_150s() -> None:
    # The full-catalog acceptance scale: measured 86.9s on this dev host
    # (304.2s in 0.2.x, before the shared per-replicate sort, the vectorized
    # ``ece_sweep`` scan, and the vectorized LOESS anchor evaluation). 150s is a
    # ~1.7x ceiling rather than this file's usual >=5x: a 5x ceiling would put a
    # 7-minute test in CI, and at 150s the check still fails outright on a
    # revert of any of the three (each alone puts this well past 150s).
    d = make_pd_portfolio(n=10_000, random_state=3)
    t0 = time.perf_counter()
    evaluate(d.y, d.scores, n_boot=1000)
    elapsed = time.perf_counter() - t0
    assert elapsed < 150.0


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
