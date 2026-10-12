"""Masterscale: one boundary convention, one object, accepted everywhere grades are."""

import copy
import json
import pathlib
import pickle

import numpy as np
import pytest

from probcal import BetaCalibrator, GradeTable, Masterscale, make_pd_portfolio
from probcal._registry import load
from probcal.metrics import (
    binomial_grade_test,
    hl_e_test,
    jeffreys_grade_test,
    jeffreys_upper_bands,
    pluto_tasche_from_arrays,
)
from probcal.monitor import CalibrationMonitor

BANDS = {
    "A": (0.0, 0.01),
    "B": (0.01, 0.02),
    "C": (0.02, 0.05),
    "D": (0.05, 0.10),
    "E": (0.10, 1.0),
}
MS = Masterscale(BANDS)
PORT = make_pd_portfolio(n=4000, random_state=3)
P_TEST = make_pd_portfolio(n=6000, random_state=11)


# ----------------------------------------------------------------- construction


def test_dict_and_from_edges_agree() -> None:
    ms2 = Masterscale.from_edges([0.01, 0.02, 0.05, 0.10], names=list("ABCDE"))
    assert MS == ms2
    assert MS.names == ("A", "B", "C", "D", "E")
    assert MS.n_grades == 5
    np.testing.assert_array_equal(MS.edges, [0.0, 0.01, 0.02, 0.05, 0.10, 1.0])
    assert MS.bands == BANDS
    assert Masterscale.from_edges([0.05, 0.01]).names == ("G1", "G2", "G3")
    assert Masterscale(dict(reversed(list(BANDS.items())))) == MS  # unsorted dict input


@pytest.mark.parametrize(
    ("make", "fragment"),
    [
        (lambda: Masterscale({"A": (0.0, 0.01), "B": (0.02, 1.0)}), "B"),  # gap: next band
        (lambda: Masterscale({"A": (0.0, 0.02), "B": (0.01, 1.0)}), "B"),  # overlap
        (lambda: Masterscale({"A": (0.01, 0.01), "B": (0.01, 1.0)}), "A"),  # lo >= hi
        (lambda: Masterscale({"A": (-0.1, 0.5), "B": (0.5, 1.0)}), "A"),  # outside [0, 1]
        (lambda: Masterscale({}), "at least one"),
        (lambda: Masterscale.from_edges([0.5], names=["A", "A"]), "unique"),
    ],
    ids=["gap", "overlap", "empty-band", "outside", "no-bands", "duplicate-names"],
)
def test_construction_errors_name_the_band(make, fragment) -> None:
    with pytest.raises(ValueError, match=fragment):
        make()


# ----------------------------------------------------------------- assign / table


def test_assign_matches_digitize_right_false() -> None:
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.random(100_000), [0.0, 0.01, 0.02, 0.05, 0.10, 1.0]])
    inner = [0.01, 0.02, 0.05, 0.10]
    expected = np.asarray(list("ABCDE"))[np.digitize(p, inner, right=False)]
    np.testing.assert_array_equal(MS.assign(p), expected)
    np.testing.assert_array_equal(MS.index(p), np.digitize(p, inner, right=False))


def test_assign_validates_and_accepts_two_columns() -> None:
    ms = Masterscale({"lo": (0.001, 0.02), "hi": (0.02, 0.5)})
    for bad in ([0.0005], [0.6]):
        with pytest.raises(ValueError, match="outside"):
            ms.assign(bad)
    for bad in ([np.nan], [1.5], [-0.1]):
        with pytest.raises(ValueError):
            MS.assign(bad)
    p = np.array([0.005, 0.015, 0.03, 0.07, 0.2])
    np.testing.assert_array_equal(MS.assign(np.column_stack([1 - p, p])), MS.assign(p))


def test_assign_does_not_clip_probabilities_across_a_close_edge() -> None:
    low = Masterscale({"A": (0.0, 1e-13), "B": (1e-13, 1.0)})
    assert low.assign([0.0]).tolist() == ["A"]
    high = Masterscale({"A": (0.0, 1 - 1e-13), "B": (1 - 1e-13, 1.0)})
    assert high.assign([1.0]).tolist() == ["B"]


def test_table_lists_every_grade_and_counts_match_assign() -> None:
    tab = MS.table(PORT.y, PORT.scores)
    assert isinstance(tab, GradeTable)
    assert tab.grades == MS.names
    labels = MS.assign(PORT.scores)
    for i, g in enumerate(MS.names):
        assert tab.n[i] == np.sum(labels == g)
        assert tab.events[i] == np.sum(PORT.y[labels == g])
    np.testing.assert_array_equal(tab.lo, MS.edges[:-1])
    np.testing.assert_array_equal(tab.hi, MS.edges[1:])


def test_table_weighted_empty_grade_equality_and_unclipped_mean_pd() -> None:
    ms = Masterscale.from_edges([0.5, 0.9])
    p = np.array([0.1, 0.2, 0.6])
    y = np.array([0.0, 1.0, 1.0])
    tab = ms.table(y, p, sample_weight=[2.0, 1.0, 1.0])
    np.testing.assert_allclose(tab.n, [3.0, 1.0, 0.0])
    np.testing.assert_allclose(tab.events, [1.0, 1.0, 0.0])
    assert np.isnan(tab.observed_rate[2]) and np.isnan(tab.mean_pd[2])
    t = Masterscale.from_edges([0.5]).table([0.0, 1.0, 1.0], [0.0, 1.0, 1.0])
    assert t.mean_pd.tolist() == [0.0, 1.0]
    # OFF-3 / OFF-21: array-aware equality; repr is the generic result table.
    assert ms.table(y, p) == ms.table(y, p)  # nan observed_rate in the empty grade compares equal
    assert ms.table(y, p) != ms.table(y, np.array([0.1, 0.6, 0.6]))
    with pytest.raises(TypeError):
        hash(tab)
    assert repr(tab).startswith("GradeTable\ngrade")


def test_interpret_states_the_convention() -> None:
    interp = MS.interpret()
    assert interp.method == "Masterscale"
    # 0.3.4: each band contributes its (lo, hi) pair, not just its upper edge.
    assert interp.param_names[:4] == ("A.lo", "A.hi", "B.lo", "B.hi")
    assert interp.param_values[:4] == (0.0, 0.01, 0.01, 0.02)
    assert len(interp.param_values) == 2 * MS.n_grades
    assert any("lo <= p < hi" in m for m in interp.messages)
    assert "half-open" in interp.messages[-1] and "closed intervals" not in interp.messages[-1]


# ----------------------------------------------------------------- identity / serialization


def test_serialization_round_trip_fingerprint_and_provenance() -> None:
    d = MS.to_dict()
    assert d["class"] == "Masterscale" and d["probcal_schema"] == 1
    back = Masterscale.from_dict(json.loads(json.dumps(d)))
    assert back == MS and back.names == MS.names
    assert load(d) == MS
    assert MS.fingerprint() == Masterscale(dict(reversed(list(BANDS.items())))).fingerprint()
    built = Masterscale(BANDS, provenance={"objective": "likelihood", "prebins": 8})
    assert Masterscale.from_dict(built.to_dict()).provenance == {
        "objective": "likelihood",
        "prebins": 8,
    }
    assert MS.provenance is None
    assert any("likelihood" in m for m in built.interpret().messages)


def test_immutable_deep_frozen_and_copyable() -> None:
    # OFF-2: both raised "Masterscale is immutable".
    ms = Masterscale(BANDS, provenance={"objective": "likelihood", "binding": ["min_count"]})
    for clone in (copy.deepcopy(ms), copy.copy(ms), pickle.loads(pickle.dumps(ms))):
        assert clone == ms and clone.provenance == ms.provenance
        assert clone.fingerprint() == ms.fingerprint()
    with pytest.raises(AttributeError, match="immutable"):
        ms._bands = ()  # type: ignore[misc]
    # OFF-12: the provenance was copied one level deep only.
    src = {"objective": "likelihood", "binding": ["min_count"], "nested": {"k": [1, 2]}}
    ms = Masterscale(BANDS, provenance=src)
    src["binding"].append("min_events")
    src["nested"]["k"].append(3)
    got = ms.provenance
    assert got == {"objective": "likelihood", "binding": ["min_count"], "nested": {"k": [1, 2]}}
    got["binding"].append("x")  # the returned copy is the caller's to mutate
    assert ms.provenance["binding"] == ["min_count"]


def test_equality_and_fingerprint_follow_one_rule() -> None:
    # OFF-13: __eq__ ignored provenance while fingerprint covered it.
    plain = Masterscale(BANDS)
    built = Masterscale(BANDS, provenance={"objective": "likelihood"})
    assert plain.bands == built.bands
    assert plain != built and plain.fingerprint() != built.fingerprint()
    twin = Masterscale(dict(reversed(list(BANDS.items()))), provenance={"objective": "likelihood"})
    assert twin == built and hash(twin) == hash(built)
    assert twin.fingerprint() == built.fingerprint()
    assert len({plain, built, twin}) == 2
    # -0.0 and 0.0 bounds are the same scale.
    a = Masterscale({"A": (-0.0, 0.5), "B": (0.5, 1.0)})
    b = Masterscale({"A": (0.0, 0.5), "B": (0.5, 1.0)})
    assert a == b and hash(a) == hash(b)
    assert a.fingerprint() == b.fingerprint() and a.to_json() == b.to_json()


# ----------------------------------------------------------------- consumers


def _aligned(res_a, res_b, fields):
    """Rows of two grade results aligned by grade name, for bit-identity checks."""
    ia = {g: i for i, g in enumerate(res_a.grades)}
    ib = {g: i for i, g in enumerate(res_b.grades)}
    assert set(ia) == set(ib)
    for f in fields:
        a, b = getattr(res_a, f), getattr(res_b, f)
        for g in ia:
            assert a[ia[g]] == b[ib[g]], (f, g)


_FIELDS = ("n", "k", "pd", "p_exact", "p_normal", "p_value", "light", "ci_low", "ci_high")


@pytest.mark.parametrize("fn", [binomial_grade_test, jeffreys_grade_test])
def test_grade_tests_equivalence_gate(fn) -> None:
    with_ms = fn(P_TEST.y, P_TEST.scores, MS)
    with_labels = fn(P_TEST.y, P_TEST.scores, MS.assign(P_TEST.scores))
    _aligned(with_ms, with_labels, [f for f in _FIELDS if hasattr(with_ms, f)])
    assert with_ms.grades == tuple(g for g in MS.names if g in with_labels.grades)


def test_hl_e_pluto_tasche_and_jeffreys_bands_equivalence_gate() -> None:
    labels = MS.assign(P_TEST.scores)
    order = tuple(g for g in MS.names if g in set(labels))

    hl_ms = hl_e_test(P_TEST.y, P_TEST.scores, MS)
    hl_labels = hl_e_test(P_TEST.y, P_TEST.scores, labels)
    assert hl_ms.e_value == hl_labels.e_value and hl_ms.p_value == hl_labels.p_value
    _aligned(hl_ms, hl_labels, ["e_grade"])

    pt_ms = pluto_tasche_from_arrays(MS, P_TEST.y, p=P_TEST.scores)
    pt_labels = pluto_tasche_from_arrays(labels, P_TEST.y, order=order)
    assert pt_ms.grades == pt_labels.grades == order
    for f in ("n", "d", "n_pooled", "d_pooled", "pd_upper"):
        np.testing.assert_array_equal(getattr(pt_ms, f), getattr(pt_labels, f))
    with pytest.raises(ValueError, match="p is required"):
        pluto_tasche_from_arrays(MS, P_TEST.y)
    with pytest.raises(ValueError, match="order is required"):
        pluto_tasche_from_arrays(labels, P_TEST.y)

    jb_ms = jeffreys_upper_bands(P_TEST.y, P_TEST.scores, MS)
    assert jb_ms == jeffreys_upper_bands(P_TEST.y, P_TEST.scores, labels, order=order)
    assert list(jb_ms) == list(order)


def test_grade_tests_report_best_to_worst_and_skip_empty_grades() -> None:
    ms = Masterscale.from_edges([0.01, 0.05], names=["AAA", "AA", "A"])
    res = jeffreys_grade_test(P_TEST.y, P_TEST.scores, ms)
    assert res.grades == ("AAA", "AA", "A")
    assert np.all(np.diff(res.pd) > 0)
    res_labels = jeffreys_grade_test(P_TEST.y, P_TEST.scores, ms.assign(P_TEST.scores))
    assert res_labels.grades == ("A", "AA", "AAA")  # label arrays keep sorted order
    assert hl_e_test(P_TEST.y, P_TEST.scores, ms).grades == ("AAA", "AA", "A")
    sparse = Masterscale.from_edges([0.01, 0.05, 0.999], names=["A", "B", "C", "D"])
    assert "D" not in jeffreys_grade_test(P_TEST.y, P_TEST.scores, sparse).grades


def _batches():
    deployed = BetaCalibrator().fit(PORT.scores, PORT.y)
    other = make_pd_portfolio(n=3000, random_state=21)
    p = deployed.predict_proba(other.scores)
    return [(other.y[i * 500 : (i + 1) * 500], p[i * 500 : (i + 1) * 500]) for i in range(6)]


@pytest.mark.filterwarnings("ignore:grade arrays were passed:UserWarning")
def test_monitor_update_masterscale_equivalence_and_fingerprint() -> None:
    a, b = CalibrationMonitor(alpha=0.05), CalibrationMonitor(alpha=0.05)
    for k, (y, p) in enumerate(_batches()):
        sa = a.update(y, p, grade=MS, label=f"b{k}")
        sb = b.update(y, p, grade=MS.assign(p), label=f"b{k}")
        assert sa.e_global == sb.e_global and sa.e_grades == sb.e_grades
        assert sa.grade_delta_ci == sb.grade_delta_ci
    assert a.masterscale_fingerprint_ == MS.fingerprint()
    assert b.masterscale_fingerprint_ is None
    d = a.to_dict()
    assert d["state"]["masterscale_fingerprint"] == MS.fingerprint()
    assert "masterscale_fingerprint" not in b.to_dict()["state"]
    back = CalibrationMonitor.from_dict(d)
    assert back.masterscale_fingerprint_ == MS.fingerprint()
    assert back.fingerprint() == a.fingerprint()


def test_pre_0_3_3_monitor_payload_loads_without_the_field() -> None:
    golden = pathlib.Path(__file__).parent / "golden" / "CalibrationMonitor.json"
    d = json.loads(golden.read_text())["object"]
    assert "masterscale_fingerprint" not in d["state"]
    assert CalibrationMonitor.from_dict(d).masterscale_fingerprint_ is None


def test_validation_report_with_masterscale(tmp_path, monkeypatch) -> None:
    pytest.importorskip("matplotlib")
    from probcal.report import validation_report

    monkeypatch.chdir(tmp_path)
    cal = BetaCalibrator().fit(PORT.scores, PORT.y)
    p = cal.predict_proba(P_TEST.scores)
    html = validation_report(P_TEST.y, p, grades=MS, n_boot=20, seed=1)
    assert MS.fingerprint() in html
    assert "masterscale" in html
    assert "Grade table" in html
    tab = MS.table(P_TEST.y, p)
    assert f"{int(tab.n[0])}" in html
    assert "ordered best to worst by the masterscale" in html
    md = validation_report(
        P_TEST.y, p, grades=MS, n_boot=20, seed=1, format="markdown", path=tmp_path / "r.md"
    )
    assert MS.fingerprint() in md and "Grade table" in md
    # A label array keeps today's wording.
    html_labels = validation_report(P_TEST.y, p, grades=MS.assign(p), n_boot=20, seed=1)
    assert "by mean predicted probability" in html_labels and "Grade table" not in html_labels
