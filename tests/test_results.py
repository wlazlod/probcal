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
        lower_80=np.array([0.005, 0.015]),
        upper_80=np.array([0.02, 0.03]),
        lower_95=np.array([0.004, 0.012]),
        upper_95=np.array([0.03, 0.04]),
        degree=2,
        p_value=0.34,
    )
    assert belt.degree == 2
    assert belt.as_dict()["p_value"] == 0.34
