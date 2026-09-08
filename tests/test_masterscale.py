"""Masterscale: one boundary convention, one object, accepted everywhere grades are."""

import numpy as np
import pytest

from probcal import Masterscale

BANDS = {
    "A": (0.0, 0.01),
    "B": (0.01, 0.02),
    "C": (0.02, 0.05),
    "D": (0.05, 0.10),
    "E": (0.10, 1.0),
}


def test_dict_and_from_edges_agree() -> None:
    ms = Masterscale(BANDS)
    ms2 = Masterscale.from_edges([0.01, 0.02, 0.05, 0.10], names=list("ABCDE"))
    assert ms == ms2
    assert ms.names == ("A", "B", "C", "D", "E")
    assert ms.n_grades == 5
    np.testing.assert_array_equal(ms.edges, [0.0, 0.01, 0.02, 0.05, 0.10, 1.0])
    assert ms.bands == BANDS
    assert list(ms.bands) == list("ABCDE")


def test_from_edges_default_names_and_unsorted_dict_input() -> None:
    ms = Masterscale.from_edges([0.05, 0.01])
    assert ms.names == ("G1", "G2", "G3")
    shuffled = Masterscale(
        {
            "E": (0.10, 1.0),
            "A": (0.0, 0.01),
            "C": (0.02, 0.05),
            "B": (0.01, 0.02),
            "D": (0.05, 0.10),
        }
    )
    assert shuffled == Masterscale(BANDS)


@pytest.mark.parametrize(
    ("bands", "fragment"),
    [
        ({"A": (0.0, 0.01), "B": (0.02, 1.0)}, "B"),  # gap, named by the band after it
        ({"A": (0.0, 0.02), "B": (0.01, 1.0)}, "B"),  # overlap
        ({"A": (0.01, 0.01), "B": (0.01, 1.0)}, "A"),  # lo >= hi
        ({"A": (-0.1, 0.5), "B": (0.5, 1.0)}, "A"),  # outside [0, 1]
        ({"A": (0.0, 0.5), "B": (0.5, 1.5)}, "B"),
        ({}, "at least one"),
    ],
)
def test_construction_errors_name_the_band(bands, fragment) -> None:
    with pytest.raises(ValueError, match=fragment):
        Masterscale(bands)


def test_duplicate_names_raise() -> None:
    with pytest.raises(ValueError, match="unique"):
        Masterscale.from_edges([0.5], names=["A", "A"])


def test_assign_matches_digitize_right_false() -> None:
    ms = Masterscale(BANDS)
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.random(100_000), [0.0, 0.01, 0.02, 0.05, 0.10, 1.0]])
    inner = [0.01, 0.02, 0.05, 0.10]
    expected = np.asarray(list("ABCDE"))[np.digitize(p, inner, right=False)]
    np.testing.assert_array_equal(ms.assign(p), expected)
    np.testing.assert_array_equal(ms.index(p), np.digitize(p, inner, right=False))


def test_boundary_convention_pinned() -> None:
    ms = Masterscale(BANDS)
    assert ms.assign([0.01])[0] == "B"  # p == lo -> this band
    assert ms.assign([0.0199999])[0] == "B"
    assert ms.assign([0.02])[0] == "C"  # p == hi -> next band
    assert ms.assign([1.0])[0] == "E"  # top band closed at its upper edge
    assert ms.assign([0.0])[0] == "A"


def test_out_of_range_raises() -> None:
    ms = Masterscale({"lo": (0.001, 0.02), "hi": (0.02, 0.5)})
    with pytest.raises(ValueError, match="outside"):
        ms.assign([0.0005])
    with pytest.raises(ValueError, match="outside"):
        ms.assign([0.6])


def test_two_column_input_equals_column_one() -> None:
    ms = Masterscale(BANDS)
    p = np.array([0.005, 0.015, 0.03, 0.07, 0.2])
    np.testing.assert_array_equal(ms.assign(np.column_stack([1 - p, p])), ms.assign(p))


def test_frozen_and_hashable() -> None:
    ms = Masterscale(BANDS)
    with pytest.raises(AttributeError):
        ms.names = ("X",)  # type: ignore[misc]
    assert hash(ms) == hash(Masterscale(BANDS))
    assert "Masterscale" in repr(ms) and "A" in repr(ms)


# ----------------------------------------------------------------- Task 2

import json  # noqa: E402

from probcal import GradeTable, make_pd_portfolio  # noqa: E402
from probcal._registry import SERIALIZABLE, load  # noqa: E402

PORT = make_pd_portfolio(n=4000, random_state=3)


def test_table_lists_every_grade_and_counts_match_assign() -> None:
    ms = Masterscale(BANDS)
    tab = ms.table(PORT.y, PORT.scores)
    assert isinstance(tab, GradeTable)
    assert tab.grades == ms.names
    labels = ms.assign(PORT.scores)
    for i, g in enumerate(ms.names):
        assert tab.n[i] == np.sum(labels == g)
        assert tab.events[i] == np.sum(PORT.y[labels == g])
    np.testing.assert_array_equal(tab.lo, ms.edges[:-1])
    np.testing.assert_array_equal(tab.hi, ms.edges[1:])
    text = str(tab)
    assert "grade" in text and "observed_rate" in text and "A" in text


def test_table_weighted_and_empty_grade() -> None:
    ms = Masterscale.from_edges([0.5, 0.9])
    p = np.array([0.1, 0.2, 0.6])
    y = np.array([0.0, 1.0, 1.0])
    tab = ms.table(y, p, sample_weight=[2.0, 1.0, 1.0])
    np.testing.assert_allclose(tab.n, [3.0, 1.0, 0.0])
    np.testing.assert_allclose(tab.events, [1.0, 1.0, 0.0])
    assert np.isnan(tab.observed_rate[2]) and np.isnan(tab.mean_pd[2])


def test_interpret_states_the_convention() -> None:
    ms = Masterscale(BANDS)
    interp = ms.interpret()
    assert interp.method == "Masterscale"
    assert interp.param_names == ms.names
    assert any("lo <= p < hi" in m for m in interp.messages)
    assert any("A" in m and "0.01" in m for m in interp.messages)


def test_serialization_round_trip_and_fingerprint() -> None:
    ms = Masterscale(BANDS)
    d = ms.to_dict()
    assert d["class"] == "Masterscale" and d["probcal_schema"] == 1
    back = Masterscale.from_dict(json.loads(json.dumps(d)))
    assert back == ms and back.names == ms.names
    assert load(d) == ms
    assert Masterscale.from_json(ms.to_json()) == ms
    assert ms.fingerprint() == Masterscale(dict(reversed(list(BANDS.items())))).fingerprint()
    other = Masterscale.from_edges([0.01, 0.02, 0.05, 0.11], names=list("ABCDE"))
    assert ms.fingerprint() != other.fingerprint()
    assert "Masterscale" in SERIALIZABLE


def test_provenance_round_trips() -> None:
    ms = Masterscale(BANDS, provenance={"objective": "likelihood", "prebins": 8})
    back = Masterscale.from_dict(ms.to_dict())
    assert back.provenance == {"objective": "likelihood", "prebins": 8}
    assert Masterscale(BANDS).provenance is None
    assert any("likelihood" in m for m in ms.interpret().messages)


def test_to_json_path(tmp_path) -> None:
    ms = Masterscale(BANDS)
    ms.to_json(tmp_path / "ms.json")
    assert Masterscale.from_json(tmp_path / "ms.json") == ms


# ----------------------------------------------------------------- Task 3

from probcal.metrics import binomial_grade_test, hl_e_test, jeffreys_grade_test  # noqa: E402


def _aligned(res_a, res_b, fields):
    """Rows of two grade results aligned by grade name, for bit-identity checks."""
    ia = {g: i for i, g in enumerate(res_a.grades)}
    ib = {g: i for i, g in enumerate(res_b.grades)}
    assert set(ia) == set(ib)
    for f in fields:
        a, b = getattr(res_a, f), getattr(res_b, f)
        for g in ia:
            assert a[ia[g]] == b[ib[g]], (f, g)


MS = Masterscale(BANDS)
P_TEST = make_pd_portfolio(n=6000, random_state=11)
_FIELDS = ("n", "k", "pd", "p_exact", "p_normal", "p_value", "light", "ci_low", "ci_high")


@pytest.mark.parametrize("fn", [binomial_grade_test, jeffreys_grade_test])
def test_grade_tests_equivalence_gate(fn) -> None:
    with_ms = fn(P_TEST.y, P_TEST.scores, MS)
    with_labels = fn(P_TEST.y, P_TEST.scores, MS.assign(P_TEST.scores))
    _aligned(with_ms, with_labels, [f for f in _FIELDS if hasattr(with_ms, f)])
    assert with_ms.grades == tuple(g for g in MS.names if g in with_labels.grades)


def test_hl_e_test_equivalence_gate() -> None:
    with_ms = hl_e_test(P_TEST.y, P_TEST.scores, MS)
    with_labels = hl_e_test(P_TEST.y, P_TEST.scores, MS.assign(P_TEST.scores))
    assert with_ms.e_value == with_labels.e_value and with_ms.p_value == with_labels.p_value
    _aligned(with_ms, with_labels, ["e_grade"])


def test_grade_tests_report_best_to_worst_not_alphabetically() -> None:
    ms = Masterscale.from_edges([0.01, 0.05], names=["AAA", "AA", "A"])
    res = jeffreys_grade_test(P_TEST.y, P_TEST.scores, ms)
    assert res.grades == ("AAA", "AA", "A")
    assert np.all(np.diff(res.pd) > 0)
    res_labels = jeffreys_grade_test(P_TEST.y, P_TEST.scores, ms.assign(P_TEST.scores))
    assert res_labels.grades == ("A", "AA", "AAA")  # today's behaviour, unchanged
    assert hl_e_test(P_TEST.y, P_TEST.scores, ms).grades == ("AAA", "AA", "A")


def test_grade_tests_skip_empty_grades_like_labels_do() -> None:
    ms = Masterscale.from_edges([0.01, 0.05, 0.999], names=["A", "B", "C", "D"])
    res = jeffreys_grade_test(P_TEST.y, P_TEST.scores, ms)
    assert "D" not in res.grades
