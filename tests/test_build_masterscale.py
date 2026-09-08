"""build_masterscale: an exact DP over pre-bins, checked against brute force."""

import itertools
import time

import numpy as np
import pytest

from probcal import (
    BetaCalibrator,
    Masterscale,
    build_masterscale,
    calibrated_bands_to_raw,
    make_pd_portfolio,
)
from probcal.metrics import jeffreys_grade_test, pluto_tasche_from_arrays
from probcal.thresholds import _prebin, _segment_objective

PORT = make_pd_portfolio(n=8000, random_state=42)


def _brute_force(y, p, k, objective, min_count, min_events, prebins, target=None):
    p_arr = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    W, E, _cuts = _prebin(p_arr, np.asarray(y, float), np.ones(len(p_arr)), prebins)
    B = len(W)
    best = None
    for combo in itertools.combinations(range(1, B), k - 1):
        bounds = (0, *combo, B)
        total = 0.0
        ok = True
        for gi, (a, b) in enumerate(zip(bounds, bounds[1:], strict=False)):
            w, e = W[a:b].sum(), E[a:b].sum()
            if w < min_count or e < min_events:
                ok = False
                break
            t = None if target is None else float(target[gi])
            total += _segment_objective(w, e, W.sum(), objective, t)
        if ok and (best is None or total > best):
            best = total
    return best


@pytest.mark.parametrize("objective", ["likelihood", "target_shares"])
@pytest.mark.parametrize(
    ("k", "prebins", "min_count", "min_events"),
    [(2, 12, 0, 0), (3, 16, 0, 0), (4, 20, 40, 1), (5, 24, 30, 0)],
)
def test_dp_equals_brute_force(objective, k, prebins, min_count, min_events) -> None:
    d = make_pd_portfolio(n=600, event_rate=0.15, random_state=5)
    target = np.full(k, 1.0 / k) if objective == "target_shares" else None
    ms = build_masterscale(
        d.y,
        d.scores,
        n_grades=k,
        min_count=min_count,
        min_events=min_events,
        objective=objective,
        target_shares=target,
        prebins=prebins,
    )
    brute = _brute_force(d.y, d.scores, k, objective, min_count, min_events, prebins, target)
    assert ms.provenance is not None
    assert ms.provenance["objective_value"] == pytest.approx(brute, abs=1e-12)
    assert ms.n_grades == k and ms.names == tuple(f"G{i + 1}" for i in range(k))


def test_edges_are_exact_and_floors_hold() -> None:
    ms = build_masterscale(PORT.y, PORT.scores, n_grades=6, min_count=200, min_events=5)
    tab = ms.table(PORT.y, PORT.scores)
    assert np.all(tab.n >= 200) and np.all(tab.events >= 5)
    assert np.all(np.diff(tab.mean_pd) > 0)  # monotone grade PDs by construction
    assert ms.edges[0] == 0.0 and ms.edges[-1] == 1.0
    p_clip = np.clip(PORT.scores, 1e-12, 1 - 1e-12)
    assert np.all(np.isin(ms.edges[1:-1], p_clip))  # edges are observed p values
    # The assigned partition is the one the optimizer scored: counts match pre-bin sums.
    W, _E, cuts = _prebin(p_clip, PORT.y, np.ones(len(p_clip)), 512)
    positions = [int(np.searchsorted(cuts, e)) + 1 for e in ms.edges[1:-1]]
    seg = [W[a:b].sum() for a, b in zip([0, *positions], [*positions, len(W)], strict=True)]
    np.testing.assert_allclose(tab.n, seg)


def test_infeasible_names_the_binding_floor() -> None:
    with pytest.raises(ValueError, match="min_events"):
        build_masterscale(PORT.y, PORT.scores, n_grades=5, min_events=1000)
    with pytest.raises(ValueError, match="min_count"):
        build_masterscale(PORT.y, PORT.scores, n_grades=5, min_count=3000)


def test_target_shares_and_objective_validation() -> None:
    with pytest.raises(ValueError, match="target_shares"):
        build_masterscale(PORT.y, PORT.scores, n_grades=3, objective="target_shares")
    with pytest.raises(ValueError, match="sum to 1"):
        build_masterscale(
            PORT.y,
            PORT.scores,
            n_grades=3,
            objective="target_shares",
            target_shares=[0.5, 0.5, 0.5],
        )
    with pytest.raises(ValueError, match="objective"):
        build_masterscale(PORT.y, PORT.scores, n_grades=3, objective="entropy")


def test_prebin_sensitivity_below_one_prebin_width() -> None:
    a = build_masterscale(PORT.y, PORT.scores, n_grades=5, prebins=512)
    b = build_masterscale(PORT.y, PORT.scores, n_grades=5, prebins=1024)
    p_sorted = np.sort(np.clip(PORT.scores, 1e-12, 1 - 1e-12))
    cuts_512 = p_sorted[(np.arange(1, 512) * len(p_sorted)) // 512]
    width = np.max(np.diff(cuts_512))
    assert np.all(np.abs(a.edges[1:-1] - b.edges[1:-1]) <= width)
    assert a.provenance["prebins"] == 512 and b.provenance["prebins"] == 1024


def test_built_scale_round_trips_and_feeds_downstream() -> None:
    ms = build_masterscale(PORT.y, PORT.scores, n_grades=5, min_count=100, names=list("ABCDE"))
    back = Masterscale.from_json(ms.to_json())
    assert back == ms and back.provenance == ms.provenance
    res = jeffreys_grade_test(PORT.y, PORT.scores, ms)
    assert res.grades == ("A", "B", "C", "D", "E")
    pt = pluto_tasche_from_arrays(ms, PORT.y, p=PORT.scores)
    assert pt.grades == ("A", "B", "C", "D", "E")
    cal = BetaCalibrator().fit(PORT.scores, PORT.y)
    raw = calibrated_bands_to_raw(cal, ms)
    assert list(raw) == list("ABCDE")
    assert any("likelihood" in m for m in ms.interpret().messages)


def test_weights_names_and_speed() -> None:
    w = np.where(PORT.y == 1, 2.0, 1.0)
    ms = build_masterscale(PORT.y, PORT.scores, n_grades=4, sample_weight=w)
    assert ms.n_grades == 4
    with pytest.raises(ValueError, match="names"):
        build_masterscale(PORT.y, PORT.scores, n_grades=4, names=["a", "b"])
    t0 = time.perf_counter()
    build_masterscale(PORT.y, PORT.scores, n_grades=8)
    assert time.perf_counter() - t0 < 2.0
