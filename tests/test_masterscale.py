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
