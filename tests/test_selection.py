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


def test_default_run_structure_and_reproducibility() -> None:
    data = _distorted(1200)
    sel = CalibratorSelector(cv=4, random_state=1).fit(*data)
    rep = sel.report_
    assert isinstance(rep, SelectionReport)
    assert len(rep.methods) == 8
    assert rep.criterion == "log_loss"
    assert int(np.sum(rep.chosen)) == 1
    assert np.all(np.diff(rep.score_mean) >= 0)  # sorted ascending
    assert rep.methods[int(np.flatnonzero(rep.chosen)[0])] == sel.best_name_
    again = CalibratorSelector(cv=4, random_state=1).fit(*data)
    assert again.best_name_ == sel.best_name_
    np.testing.assert_array_equal(again.report_.score_mean, rep.score_mean)


def test_parsimony_on_calibrated_data() -> None:
    sel = CalibratorSelector(cv=5, random_state=11).fit(*_calibrated(1500))
    # On already calibrated data the criterion cannot separate methods beyond
    # noise; the one-standard-error parsimony rule must pull the winner into
    # the low-parameter parametric block.
    assert sel.best_name_ in {"temperature", "platt", "beta_abm"}


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


def test_selection_report_has_mcb_dsc_columns() -> None:
    """report_ carries a CORP decomposition per candidate on its OOF vector,
    and it round-trips through to_dict/from_dict."""
    s, y = _distorted(1500)
    sel = CalibratorSelector(cv=3, random_state=3).fit(s, y)
    rep = sel.report_
    assert rep.mcb is not None and rep.mcb.shape == (len(rep.methods),)
    assert rep.dsc is not None and rep.dsc.shape == (len(rep.methods),)
    assert rep.unc is not None
    assert np.all(rep.mcb >= -1e-12) and np.all(rep.dsc >= -1e-12)

    sel2 = CalibratorSelector.from_dict(sel.to_dict())
    np.testing.assert_allclose(sel2.report_.mcb, rep.mcb)
    np.testing.assert_allclose(sel2.report_.dsc, rep.dsc)
    assert sel2.report_.unc == pytest.approx(rep.unc)


# ---------------------------------------------------------------- 0.3.4 regressions


def test_selector_delegates_the_inverse_protocol_to_its_winner() -> None:
    """CAL-7: 0.3.x delegated only predict/interpret, so a selector that picked
    Platt had no point_inverse and a bisection-only interval_inverse."""
    from probcal.isotonic import IsotonicCalibrator
    from probcal.parametric import PlattCalibrator

    s, y = _distorted(1200)
    sel = CalibratorSelector(candidates={"platt": PlattCalibrator()}, cv=3).fit(s, y)
    best = sel.best_calibrator_
    assert sel.affine_logit_coeffs_ == best.affine_logit_coeffs_
    assert sel.complexity_rank == best.complexity_rank == 2.0
    p = np.linspace(0.05, 0.9, 12)
    np.testing.assert_array_equal(sel.point_inverse(p), best.point_inverse(p))
    assert sel.interval_inverse(0.1, 0.3, space="logit") == best.interval_inverse(
        0.1, 0.3, space="logit"
    )

    iso = CalibratorSelector(candidates={"iso": IsotonicCalibrator()}, cv=3).fit(s, y)
    # Exact block-edge semantics, not the base-class bisection approximation.
    assert iso.interval_inverse(0.2, 0.4) == iso.best_calibrator_.interval_inverse(0.2, 0.4)
    assert iso.affine_logit_coeffs_ is None
    with pytest.raises(NotImplementedError):
        iso.point_inverse(np.array([0.3]))


def test_selector_is_monotone_follows_the_winner_through_json() -> None:
    from probcal.binning import HistogramBinningCalibrator

    rng = np.random.default_rng(3)
    s = rng.uniform(0.05, 0.95, 400)
    y = (rng.random(400) < 0.3).astype(float)  # flat truth: noisy, non-monotone bins
    sel = CalibratorSelector(
        candidates={"hist": HistogramBinningCalibrator(n_bins=12, shrinkage=None)}, cv=3
    ).fit(s, y)
    assert sel.best_calibrator_.is_monotone_ is False
    assert sel.is_monotone_ is False
    assert CalibratorSelector.from_json(sel.to_json()).is_monotone_ is False


def test_selector_unfitted_protocol() -> None:
    """CAL-24: interpret() refuses before fit; complexity_rank is the default."""
    sel = CalibratorSelector()
    with pytest.raises(RuntimeError, match="not fitted"):
        sel.interpret()
    assert sel.complexity_rank == 100.0
    assert sel.is_monotone_ is True


def test_selector_with_nested_non_json_candidate_round_trips() -> None:
    """Candidate params go through each prototype's own _params_for_dict."""
    from probcal import SegmentedCalibrator
    from probcal.parametric import PlattCalibrator

    cands = {"seg": SegmentedCalibrator(base=PlattCalibrator()), "platt": PlattCalibrator()}
    sel = CalibratorSelector(candidates=cands, cv=3).fit(*_distorted(900))
    again = CalibratorSelector.from_json(sel.to_json())
    grid = np.linspace(0.01, 0.99, 30)
    np.testing.assert_array_equal(again.predict_proba(grid), sel.predict_proba(grid))
    assert isinstance(again.candidates["seg"].base, PlattCalibrator)
    assert set(sel.report_.methods) == {"seg", "platt"}


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"scoring": "ece"}, "log_loss"),
        # CAL-14: cv=1 used to fit every candidate on an empty training fold.
        ({"cv": 1}, "cv"),
        ({"cv": 2.5}, "cv"),
        ({"cv": True}, "cv"),
    ],
)
def test_constructor_validation(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        CalibratorSelector(**kwargs).fit(*_calibrated(300))


def test_fewer_events_than_folds_raises_clearly() -> None:
    from probcal.parametric import PlattCalibrator

    rng = np.random.default_rng(1)
    y = np.zeros(200, int)
    y[:4] = 1
    s = rng.random(200)
    with pytest.raises(ValueError, match=r"cv=5 needs at least 5 .* 4 positives .* cv <= 4"):
        CalibratorSelector(cv=5).fit(s, y)
    CalibratorSelector(candidates={"platt": PlattCalibrator()}, cv=4).fit(s, y)
