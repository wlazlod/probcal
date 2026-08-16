"""CalibratorSelector: automatic method selection under nested validation.

The selector's scoring path only ever receives out-of-fold predictions —
selection on fitting data is an unrepresentable state, not a documented
misuse. Protocol, criteria, and report reading:
``docs/concepts/auto-selection.md``.
"""

import numpy as np

from ._results import Interpretation, SelectionReport
from ._validation import validate_binary_y, validate_scores, validate_weights
from .base import BaseCalibrator
from .binning import HistogramBinningCalibrator, ScalingBinningCalibrator
from .isotonic import CenteredIsotonicCalibrator, IsotonicCalibrator
from .metrics import brier_score as _brier
from .metrics import ece_sweep, ici, log_loss, smooth_ece
from .metrics.regression import calibration_guardrails
from .parametric import BetaCalibrator, PlattCalibrator, TemperatureCalibrator
from .vennabers import VennAbersCalibrator

_SCORERS = {
    "log_loss": log_loss,
    "brier": _brier,
    "ici": ici,
    "smooth_ece": smooth_ece,
    "ece_sweep": ece_sweep,
}


def _default_candidates() -> dict[str, BaseCalibrator]:
    return {
        "platt": PlattCalibrator(),
        "temperature": TemperatureCalibrator(),
        "beta_abm": BetaCalibrator(variant="abm"),
        "isotonic": IsotonicCalibrator(),
        "cir": CenteredIsotonicCalibrator(),
        "histogram_mass": HistogramBinningCalibrator(strategy="mass"),
        "scaling_binning": ScalingBinningCalibrator(),
        "ivap": VennAbersCalibrator(),
    }


class CalibratorSelector:
    """Choose a calibrator by inner cross-validation on the calibration data.

    Custom candidates declare their tie-break position by overriding
    ``complexity_rank`` (lower = simpler; default 100.0 ranks last).

    Parameters
    ----------
    candidates : dict[str, BaseCalibrator] or None
        Candidate instances (cloned per fold via ``get_params``). ``None``
        uses the spec's default menu: platt, temperature, beta_abm,
        isotonic, cir, histogram_mass, scaling_binning, ivap. The heavier
        methods (spline, BBQ, ENIR, CVAP) join by explicit opt-in.
    scoring : {"log_loss", "brier", "ici", "smooth_ece", "ece_sweep"}
        Out-of-fold selection criterion, lower is better. Plain ECE and
        Hosmer–Lemeshow are refused — see the metrics chapter's table.
    cv : int
        Inner stratified fold count.
    random_state : int
        Seed for the fold assignment.

    Attributes
    ----------
    best_name_ : str
        Winning candidate's name.
    best_calibrator_ : BaseCalibrator
        The winner refitted on the full calibration set.
    report_ : SelectionReport
        Ranked table: mean ± sd of the criterion, guardrail flags, chosen
        marker.
    """

    def __init__(
        self,
        candidates: dict[str, BaseCalibrator] | None = None,
        scoring: str = "log_loss",
        cv: int = 5,
        random_state: int = 42,
    ) -> None:
        self.candidates = candidates
        self.scoring = scoring
        self.cv = cv
        self.random_state = random_state

    def fit(self, s: object, y: object, sample_weight: object = None) -> "CalibratorSelector":
        """Run the nested selection and refit the winner on all data.

        Parameters
        ----------
        s : array_like
            Raw scores/probabilities in ``[0, 1]``.
        y : array_like
            Binary outcomes in ``{0, 1}``; both classes must be present.
        sample_weight : array_like or None
            Positive observation weights.

        Returns
        -------
        CalibratorSelector
            ``self``, with ``best_name_``, ``best_calibrator_``, and
            ``report_`` set.

        Raises
        ------
        ValueError
            If ``scoring`` is not one of the accepted criteria (plain ECE
            and Hosmer–Lemeshow are refused as selection criteria).
        """
        if self.scoring not in _SCORERS:
            raise ValueError(
                f"scoring must be one of {sorted(_SCORERS)} (proper scores and accepted "
                f"binning-free alternatives), got {self.scoring!r}; plain ECE and "
                "Hosmer-Lemeshow are not selection criteria"
            )
        scorer = _SCORERS[self.scoring]
        menu = self.candidates if self.candidates is not None else _default_candidates()
        s_arr = validate_scores(s)
        y_arr = validate_binary_y(y)
        w_arr = validate_weights(sample_weight, len(s_arr))

        rng = np.random.default_rng(self.random_state)
        folds = np.empty(len(y_arr), dtype=np.int64)
        for cls in (0.0, 1.0):
            idx = np.flatnonzero(y_arr == cls)
            perm = rng.permutation(idx)
            folds[perm] = np.arange(len(perm)) % self.cv

        names = list(menu)
        means = np.empty(len(names))
        sds = np.empty(len(names))
        guards = np.empty(len(names), dtype=bool)
        for i, name in enumerate(names):
            proto = menu[name]
            fold_scores = np.empty(self.cv)
            oof = np.empty(len(y_arr))
            for k in range(self.cv):
                train, held = folds != k, folds == k
                cal = type(proto)(**proto.get_params())
                cal.fit(s_arr[train], y_arr[train], sample_weight=w_arr[train])
                pred_held = cal.predict_proba(s_arr[held])
                oof[held] = pred_held
                fold_scores[k] = scorer(y_arr[held], pred_held, sample_weight=w_arr[held])
            means[i] = fold_scores.mean()
            sds[i] = fold_scores.std(ddof=1)
            guards[i] = calibration_guardrails(y_arr, oof, sample_weight=w_arr).all_ok

        # Parsimony tie-break within one standard error of the best mean.
        best_idx = int(np.argmin(means))
        se_best = sds[best_idx] / np.sqrt(self.cv)
        tied = [i for i in range(len(names)) if means[i] <= means[best_idx] + se_best]
        winner = min(
            tied,
            key=lambda i: (getattr(menu[names[i]], "complexity_rank", 100.0), means[i]),
        )

        order = np.argsort(means, kind="stable")
        chosen = np.zeros(len(names), dtype=bool)
        chosen[winner] = True
        self.report_ = SelectionReport(
            methods=tuple(names[i] for i in order),
            score_mean=means[order],
            score_sd=sds[order],
            guardrails_ok=guards[order],
            chosen=chosen[order],
            criterion=self.scoring,
        )
        self.best_name_ = names[winner]
        proto = menu[self.best_name_]
        self.best_calibrator_ = type(proto)(**proto.get_params())
        self.best_calibrator_.fit(s_arr, y_arr, sample_weight=w_arr)
        return self

    def predict_proba(self, s: object) -> np.ndarray:
        """Delegate to the refitted winner."""
        return self.best_calibrator_.predict_proba(s)

    def interpret(self) -> Interpretation:
        """Delegate to the refitted winner."""
        return self.best_calibrator_.interpret()
