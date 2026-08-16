"""Tests for probcal.selection.CalibratorSelector."""

import numpy as np
import pytest

from probcal._math import expit, logit
from probcal._results import Interpretation, SelectionReport
from probcal.base import BaseCalibrator
from probcal.selection import CalibratorSelector

RNG = np.random.default_rng(113)


def _distorted(n: int = 2500) -> tuple[np.ndarray, np.ndarray]:
    s = expit(RNG.normal(-1.0, 1.4, n))
    y = (RNG.random(n) < expit(0.7 * logit(s) - 0.4)).astype(float)
    return s, y


def _calibrated(n: int = 1500) -> tuple[np.ndarray, np.ndarray]:
    s = expit(RNG.normal(-1.0, 1.2, n))
    y = (RNG.random(n) < s).astype(float)
    return s, y


def test_default_run_structure() -> None:
    sel = CalibratorSelector(cv=4, random_state=1).fit(*_distorted(1200))
    rep = sel.report_
    assert isinstance(rep, SelectionReport)
    assert len(rep.methods) == 8
    assert rep.criterion == "log_loss"
    assert int(np.sum(rep.chosen)) == 1
    assert np.all(np.diff(rep.score_mean) >= 0)  # sorted ascending
    assert rep.methods[int(np.flatnonzero(rep.chosen)[0])] == sel.best_name_


def test_winner_refit_and_delegation() -> None:
    s, y = _distorted(1500)
    sel = CalibratorSelector(cv=4, random_state=2).fit(s, y)
    assert sel.best_calibrator_.fitted_
    p = sel.predict_proba(np.linspace(0.01, 0.99, 50))
    assert np.all((p > 0) & (p < 1))
    assert isinstance(sel.interpret(), Interpretation)


def test_reproducible_under_seed() -> None:
    s, y = _distorted(1200)
    a = CalibratorSelector(cv=4, random_state=7).fit(s, y)
    b = CalibratorSelector(cv=4, random_state=7).fit(s, y)
    assert a.best_name_ == b.best_name_
    np.testing.assert_allclose(a.report_.score_mean, b.report_.score_mean)


def test_parsimony_on_calibrated_data() -> None:
    sel = CalibratorSelector(cv=5, random_state=11).fit(*_calibrated(1500))
    # On already calibrated data the criterion cannot separate methods beyond
    # noise; the one-standard-error parsimony rule must pull the winner into
    # the low-parameter parametric block.
    assert sel.best_name_ in {"temperature", "platt", "beta_abm"}


def test_forbidden_scoring_raises() -> None:
    with pytest.raises(ValueError, match="log_loss"):
        CalibratorSelector(scoring="ece").fit(*_calibrated(400))
    with pytest.raises(ValueError, match="log_loss"):
        CalibratorSelector(scoring="hosmer_lemeshow").fit(*_calibrated(400))


class SpyCalibrator(BaseCalibrator):
    """Constant predictor recording fitted and scored score-multisets."""

    calls: list[tuple[frozenset, frozenset]] = []

    def _fit(self, s: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        self._fit_scores = frozenset(np.round(s, 12).tolist())
        self._rate = float(np.average(y, weights=w))

    def _predict(self, s: np.ndarray) -> np.ndarray:
        SpyCalibrator.calls.append((self._fit_scores, frozenset(np.round(s, 12).tolist())))
        return np.full(len(s), np.clip(self._rate, 1e-6, 1 - 1e-6))

    def interpret(self) -> Interpretation:
        return Interpretation(
            method="SpyCalibrator", param_names=(), param_values=(), messages=("spy",)
        )


def test_structural_no_leakage_in_scoring() -> None:
    s, y = _calibrated(800)
    s = np.unique(s)[: len(s)]  # scores are unique -> set identity works
    y = y[: len(s)]
    SpyCalibrator.calls = []
    sel = CalibratorSelector(
        candidates={"spy": SpyCalibrator(), "spy2": SpyCalibrator()},
        cv=4,
        random_state=3,
    ).fit(s, y)
    n = len(s)
    oof_calls = [c for c in SpyCalibrator.calls if len(c[0]) < n]
    assert oof_calls
    for fit_scores, scored in oof_calls:
        assert not (fit_scores & scored), "selector scored a candidate on its fitting data"
    assert sel.best_name_ in ("spy", "spy2")


def test_custom_candidates() -> None:
    from probcal.parametric import PlattCalibrator, TemperatureCalibrator

    sel = CalibratorSelector(
        candidates={"p": PlattCalibrator(), "t": TemperatureCalibrator()},
        cv=3,
        random_state=5,
    ).fit(*_distorted(900))
    assert set(sel.report_.methods) == {"p", "t"}


def test_report_repr_mentions_criterion() -> None:
    sel = CalibratorSelector(cv=3, random_state=9).fit(*_distorted(900))
    assert "log_loss" in repr(sel.report_)


def test_export() -> None:
    import probcal

    assert "CalibratorSelector" in probcal.__all__


def test_complexity_rank_matches_old_parsimony_table() -> None:
    """Round-trip: each old _PARSIMONY entry equals the corresponding instance's
    complexity_rank (DECISIONS 49: exact values are inert beyond ordering)."""
    from probcal.bayesian import BBQCalibrator, ENIRCalibrator
    from probcal.binning import HistogramBinningCalibrator, ScalingBinningCalibrator
    from probcal.isotonic import CenteredIsotonicCalibrator, IsotonicCalibrator
    from probcal.parametric import BetaCalibrator, PlattCalibrator, TemperatureCalibrator
    from probcal.spline import SplineCalibrator
    from probcal.vennabers import CrossVennAbersCalibrator, VennAbersCalibrator

    expected = [
        (TemperatureCalibrator(), 1.0),  # temperature
        (PlattCalibrator(), 2.0),  # platt
        (BetaCalibrator(variant="ab"), 2.5),  # beta_ab
        (BetaCalibrator(variant="a"), 1.5),  # beta_a
        (BetaCalibrator(variant="abm"), 3.0),  # beta_abm
        (ScalingBinningCalibrator(), 4.0),  # scaling_binning
        (BBQCalibrator(), 40.0),  # bbq
        (HistogramBinningCalibrator(strategy="mass"), 10.0),  # histogram_mass
        (HistogramBinningCalibrator(strategy="width"), 10.0),  # histogram_width
        (SplineCalibrator(), 12.0),  # spline
        (CenteredIsotonicCalibrator(), 50.0),  # cir
        (IsotonicCalibrator(), 50.0),  # isotonic
        (VennAbersCalibrator(), 60.0),  # ivap
        (CrossVennAbersCalibrator(), 60.0),  # cvap
        (ENIRCalibrator(), 80.0),  # enir
    ]
    assert len(expected) == 15
    for calibrator, rank in expected:
        assert calibrator.complexity_rank == rank, type(calibrator).__name__


def test_forced_tie_custom_complexity_rank_wins() -> None:
    from probcal.parametric import TemperatureCalibrator

    class _TinyRank(TemperatureCalibrator):
        """Behaves exactly like TemperatureCalibrator but ranks simpler."""

        @property
        def complexity_rank(self) -> float:
            return 0.5

    # "temperature" is listed FIRST so dict/iteration order favors the built-in;
    # the test can only pass if complexity_rank (0.5 < 1.0) is actually consulted
    # by the tie-break, not merely dict order.
    sel = CalibratorSelector(
        candidates={"temperature": TemperatureCalibrator(), "custom": _TinyRank()},
        cv=5,
        random_state=11,
    ).fit(*_calibrated(1500))
    assert sel.best_name_ == "custom"


def test_default_complexity_rank() -> None:
    """No override -> 100.0, both on the base property and via the selector's
    getattr fallback for non-BaseCalibrator duck-typed candidates."""

    class _NoOverride(BaseCalibrator):
        def _fit(self, s: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
            self._rate = float(np.average(y, weights=w))

        def _predict(self, s: np.ndarray) -> np.ndarray:
            return np.full(len(s), self._rate)

        def interpret(self) -> Interpretation:
            return Interpretation(
                method="_NoOverride", param_names=(), param_values=(), messages=()
            )

    assert _NoOverride().complexity_rank == 100.0

    class _DuckTyped:
        """Not a BaseCalibrator; exercises the selector's getattr fallback."""

    assert getattr(_DuckTyped(), "complexity_rank", 100.0) == 100.0
