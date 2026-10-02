"""Tests for probcal._results dataclasses."""

import dataclasses

import numpy as np
import pytest

from probcal._results import (
    BeltResult,
    Interpretation,
    MetricReport,
    ReliabilityCurve,
    SelectionReport,
)


def _interpretation() -> Interpretation:
    return Interpretation(
        method="PlattCalibrator",
        param_names=("a", "b"),
        param_values=(0.85, -0.12),
        messages=("slope a < 1: overconfident spread shrunk toward the base rate",),
    )


def test_interpretation_frozen() -> None:
    interp = _interpretation()
    with pytest.raises(dataclasses.FrozenInstanceError):
        interp.method = "other"  # type: ignore[misc]


def test_interpretation_as_dict() -> None:
    d = _interpretation().as_dict()
    assert d["method"] == "PlattCalibrator"
    assert d["param_names"] == ("a", "b")


def test_interpretation_repr_aligned_table() -> None:
    r = repr(_interpretation())
    assert "PlattCalibrator" in r
    assert "a" in r and "-0.12" in r


def test_metric_report_as_dict_and_repr() -> None:
    rep = MetricReport(
        names=("log_loss", "brier"),
        values=np.array([0.131, 0.028]),
        ci_low=np.array([0.120, 0.025]),
        ci_high=np.array([0.144, 0.031]),
    )
    d = rep.as_dict()
    assert set(d) == {"names", "values", "ci_low", "ci_high"}
    r = repr(rep)
    assert "log_loss" in r and "0.131" in r


def test_reliability_curve_roundtrip() -> None:
    curve = ReliabilityCurve(
        pred_mean=np.array([0.01, 0.05]),
        event_rate=np.array([0.012, 0.043]),
        count=np.array([500, 480]),
        ci_low=np.array([0.005, 0.03]),
        ci_high=np.array([0.02, 0.06]),
        pred_mean_logit=np.array([-4.59, -2.94]),
    )
    assert curve.as_dict()["count"].tolist() == [500, 480]


def test_selection_report_repr_marks_chosen() -> None:
    rep = SelectionReport(
        methods=("platt", "beta_abm"),
        score_mean=np.array([0.131, 0.129]),
        score_sd=np.array([0.004, 0.006]),
        guardrails_ok=np.array([True, True]),
        chosen=np.array([False, True]),
        criterion="log_loss",
    )
    r = repr(rep)
    assert "beta_abm" in r and "log_loss" in r


def test_belt_result_fields() -> None:
    belt = BeltResult(
        grid_p=np.array([0.01, 0.02]),
        grid_logit=np.array([-4.6, -3.9]),
        levels=(0.8, 0.95),
        bands={
            0.8: (np.array([0.005, 0.015]), np.array([0.02, 0.03])),
            0.95: (np.array([0.004, 0.012]), np.array([0.03, 0.04])),
        },
        degree=2,
        p_value=0.34,
    )
    assert belt.degree == 2
    assert belt.as_dict()["p_value"] == 0.34
    # The 0.3.x attribute names still work for one release, with a warning.
    with pytest.warns(DeprecationWarning, match="bands\\[0.95\\]"):
        np.testing.assert_array_equal(belt.upper_95, np.array([0.03, 0.04]))
    with pytest.warns(DeprecationWarning):
        np.testing.assert_array_equal(belt.lower_80, np.array([0.005, 0.015]))


# ---------------------------------------------------------------- 0.4.0 fixes


def _metric_report(v: float = 0.131) -> MetricReport:
    return MetricReport(
        names=("log_loss", "brier"),
        values=np.array([v, 0.028]),
        ci_low=np.array([np.nan, 0.025]),
        ci_high=np.array([0.144, 0.031]),
    )


def test_array_results_compare_by_value_and_are_unhashable() -> None:
    # OFF-3: the generated __eq__ raised "truth value of an array is ambiguous".
    assert _metric_report() == _metric_report()  # nan == nan field-wise
    assert _metric_report() != _metric_report(0.2)
    assert _metric_report() != "MetricReport"
    with pytest.raises(TypeError):
        hash(_metric_report())
    # Results without arrays keep value hashing.
    assert hash(_interpretation()) == hash(_interpretation())


def test_grouped_report_rows_reuse_metric_report_rows() -> None:
    # OFF-21: one row mechanism; repr and the HTML report print the same rows.
    from probcal._results import GroupedMetricReport

    g = GroupedMetricReport(
        pooled=_metric_report(),
        groups=("a",),
        reports=(_metric_report(0.2),),
        counts=np.array([10]),
    )
    assert repr(g._rows()[0]) == repr(("pooled", *_metric_report()._rows()[0]))
    assert g._headers() == ("group", "metric", "value", "ci_low", "ci_high")
    assert repr(g).startswith("GroupedMetricReport (1 groups)\ngroup")


def test_selection_report_headers_follow_corp_columns() -> None:
    rep = SelectionReport(
        methods=("platt",),
        score_mean=np.array([0.1]),
        score_sd=np.array([0.01]),
        guardrails_ok=np.array([True]),
        chosen=np.array([True]),
        criterion="brier",
        mcb=np.array([0.01]),
        dsc=np.array([0.02]),
        unc=0.1,
    )
    assert rep._headers()[-2:] == ("mcb", "dsc")
    assert rep._rows()[0][4] == "*"
