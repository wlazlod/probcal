"""Tests for probcal._results dataclasses."""

import copy
import pickle

import numpy as np
import pytest

from probcal import calibration_belt, make_pd_portfolio
from probcal._results import (
    GroupedMetricReport,
    Interpretation,
    MetricReport,
    SelectionReport,
    _values_equal,
)


def _interpretation() -> Interpretation:
    return Interpretation(
        method="PlattCalibrator",
        param_names=("a", "b"),
        param_values=(0.85, -0.12),
        messages=("slope a < 1: overconfident spread shrunk toward the base rate",),
    )


def _metric_report(v: float = 0.131) -> MetricReport:
    return MetricReport(
        names=("log_loss", "brier"),
        values=np.array([v, 0.028]),
        ci_low=np.array([np.nan, 0.025]),
        ci_high=np.array([0.144, 0.031]),
    )


def test_interpretation_repr_aligned_table() -> None:
    r = repr(_interpretation())
    assert "PlattCalibrator" in r
    assert "a" in r and "-0.12" in r


def test_table_results_share_one_row_mechanism() -> None:
    # OFF-21: one row mechanism; repr and the HTML report print the same rows.
    rep = _metric_report()
    assert set(rep.as_dict()) == {"names", "values", "ci_low", "ci_high"}
    assert "log_loss" in repr(rep) and "0.131" in repr(rep)
    g = GroupedMetricReport(
        pooled=rep, groups=("a",), reports=(_metric_report(0.2),), counts=np.array([10])
    )
    assert repr(g._rows()[0]) == repr(("pooled", *rep._rows()[0]))
    assert g._headers() == ("group", "metric", "value", "ci_low", "ci_high")
    assert repr(g).startswith("GroupedMetricReport (1 groups)\ngroup")
    sel = SelectionReport(
        methods=("platt", "beta_abm"),
        score_mean=np.array([0.1, 0.2]),
        score_sd=np.array([0.01, 0.02]),
        guardrails_ok=np.array([True, True]),
        chosen=np.array([True, False]),
        criterion="brier",
        mcb=np.array([0.01, 0.02]),
        dsc=np.array([0.02, 0.03]),
        unc=0.1,
    )
    assert sel._headers()[-2:] == ("mcb", "dsc")
    assert sel._rows()[0][4] == "*" and sel._rows()[1][4] != "*"
    assert "brier" in repr(sel)


def test_array_results_compare_by_value_and_are_unhashable() -> None:
    # OFF-3: the generated __eq__ raised "truth value of an array is ambiguous".
    assert _metric_report() == _metric_report()  # nan == nan field-wise
    assert _metric_report() != _metric_report(0.2)
    assert _metric_report() != "MetricReport"
    with pytest.raises(TypeError):
        hash(_metric_report())
    # Results without arrays keep value hashing.
    assert hash(_interpretation()) == hash(_interpretation())
    # Mappings of arrays (BeltResult.bands) compare element-wise too.
    a = {0.8: (np.array([0.1, np.nan]), np.array([0.2, 0.3]))}
    b = {0.8: (np.array([0.1, np.nan]), np.array([0.2, 0.3]))}
    assert _values_equal(a, b)
    assert not _values_equal(a, {0.8: (a[0.8][0], np.array([0.2, 0.4]))})
    assert not _values_equal(a, {0.9: a[0.8]})


def test_belt_result_equality_read_only_bands_and_copies() -> None:
    d = make_pd_portfolio(n=2000, random_state=0)
    belt = calibration_belt(d.y, d.scores)
    assert belt == calibration_belt(d.y, d.scores)
    assert belt != calibration_belt(d.y, d.scores, confidence=(0.5, 0.9))
    assert set(belt.bands) == set(belt.levels)
    for obj in (belt, pickle.loads(pickle.dumps(belt)), copy.deepcopy(belt), copy.copy(belt)):
        assert obj == belt
        with pytest.raises(TypeError):
            obj.bands[0.5] = None  # type: ignore[index]
