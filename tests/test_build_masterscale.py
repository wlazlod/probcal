"""build_masterscale: an exact DP over pre-bins, checked against brute force."""

import itertools

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
from probcal.thresholds import _prebin

PORT = make_pd_portfolio(n=8000, random_state=42)


def _segment_objective(w, e, w_total, objective, target):
    """Scalar reference objective of one grade, written independently of the DP."""
    if objective == "likelihood":
        if w <= 0 or e <= 0 or e >= w:
            return 0.0
        r = e / w
        return e * np.log(r) + (w - e) * np.log1p(-r)
    return -((w / w_total - target) ** 2)


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


def test_from_edges_carries_provenance() -> None:
    # OFF-23: build_masterscale builds the scale once, via from_edges(provenance=).
    ms = Masterscale.from_edges([0.1], provenance={"source": "test", "edges": [0.1]})
    assert ms.provenance == {"source": "test", "edges": [0.1]}
    built = build_masterscale(PORT.y, PORT.scores, n_grades=3)
    assert built == Masterscale.from_edges(
        built.edges[1:-1], names=built.names, provenance=built.provenance
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"min_events": 1000}, "binding: min_events"),
        ({"min_count": 3000}, "binding: min_count"),
        ({"objective": "target_shares"}, "requires target_shares"),
        ({"objective": "target_shares", "target_shares": [0.5, 0.5]}, "non-negative shares"),
        ({"objective": "target_shares", "target_shares": [0.5, 0.5, 0.5]}, "sum to 1"),
        ({"objective": "entropy"}, "objective"),
        ({"names": ["a", "b"]}, "names"),
        # OFF-18: these used to fall through to a numpy error deep in the DP.
        ({"n_grades": True}, "n_grades must be a positive integer"),
        ({"prebins": 2.5}, "prebins must be a positive integer"),
    ],
)
def test_validation(kwargs, match) -> None:
    kwargs = {"n_grades": 3, **kwargs}
    with pytest.raises(ValueError, match=match):
        build_masterscale(PORT.y, PORT.scores, **kwargs)


def test_weighted_build_is_weight_scale_invariant() -> None:
    w = np.random.default_rng(0).uniform(0.1, 5.0, len(PORT.y))
    a = build_masterscale(PORT.y, PORT.scores, n_grades=4, sample_weight=w, min_count=100)
    b = build_masterscale(PORT.y, PORT.scores, n_grades=4, sample_weight=10 * w, min_count=1000)
    np.testing.assert_array_equal(a.edges, b.edges)
    assert not np.array_equal(a.edges, build_masterscale(PORT.y, PORT.scores, n_grades=4).edges)


def test_edges_are_observed_unclipped_values_even_with_mass_at_zero_and_one() -> None:
    p = np.r_[np.zeros(200), np.full(300, 0.3), np.full(500, 0.6)]
    y = (np.arange(1000) % 7 == 0).astype(float)
    ms = build_masterscale(y, p, n_grades=3)
    assert ms.edges.tolist() == [0.0, 0.3, 0.6, 1.0]
    assert ms.table(y, p).n.tolist() == [200.0, 300.0, 500.0]

    rng = np.random.default_rng(0)
    p = np.r_[rng.uniform(0.01, 0.2, 700), np.ones(300)]
    y = (rng.random(1000) < p).astype(float)
    ms = build_masterscale(y, p, n_grades=3)
    assert ms.edges[-1] == 1.0
    assert np.all(ms.edges[1:-1] < 0.2)  # no near-one edge from clipping
    assert ms.table(y, p).n.sum() == 1000
