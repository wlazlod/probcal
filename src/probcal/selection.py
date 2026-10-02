"""CalibratorSelector: automatic method selection under nested validation.

The selector's scoring path only ever receives out-of-fold predictions —
selection on fitting data is an unrepresentable state, not a documented
misuse. Protocol, criteria, and report reading:
``docs/concepts/auto-selection.md``.
"""

from typing import cast

import numpy as np

from ._corp import corp_fit, decompose
from ._registry import SERIALIZABLE, register
from ._results import Interpretation, SelectionReport
from ._serialize import decode_value, encode_value
from ._validation import stratified_folds, validate_cv
from .base import BaseCalibrator, clone_unfitted
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


@register
class CalibratorSelector(BaseCalibrator):
    """Choose a calibrator by inner cross-validation on the calibration data.

    Custom candidates declare their tie-break position by overriding
    ``complexity_rank`` (lower = simpler; default 100.0 ranks last).

    Once fitted, the selector *is* its winner for every protocol question:
    ``predict_proba``, ``interpret``, ``is_monotone_``,
    ``affine_logit_coeffs_``, ``complexity_rank``, ``point_inverse`` and
    ``interval_inverse`` all delegate to ``best_calibrator_``. ``fit`` raises
    ``ValueError`` if ``scoring`` is not one of the accepted criteria (plain
    ECE and Hosmer–Lemeshow are refused as selection criteria) or ``cv`` is
    not an integer ``>= 2``.

    Parameters
    ----------
    candidates : dict[str, BaseCalibrator] or None
        Candidate instances (cloned per fold via ``get_params``). ``None``
        uses the default menu: platt, temperature, beta_abm,
        isotonic, cir, histogram_mass, scaling_binning, ivap. The heavier
        methods (spline, BBQ, ENIR, CVAP) join by explicit opt-in.
    scoring : {"log_loss", "brier", "ici", "smooth_ece", "ece_sweep"}
        Out-of-fold selection criterion, lower is better. Plain ECE and
        Hosmer–Lemeshow are refused — see the metrics chapter's table.
    cv : int
        Inner stratified fold count; an integer ``>= 2``.
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

    _STATE_ATTRS = ("best_name_", "best_calibrator_")

    def _state(self) -> dict[str, object]:
        base = super()._state()
        r = self.report_
        base["report"] = {
            "methods": list(r.methods),
            "score_mean": encode_value(r.score_mean),
            "score_sd": encode_value(r.score_sd),
            "guardrails_ok": encode_value(r.guardrails_ok),
            "chosen": encode_value(r.chosen),
            "criterion": r.criterion,
            "mcb": encode_value(r.mcb),
            "dsc": encode_value(r.dsc),
            "unc": encode_value(r.unc),
        }
        return base

    def _set_state(self, state: dict[str, object]) -> None:
        state = dict(state)
        state.pop("is_monotone_", None)  # pre-0.4 payloads; now read from the winner
        rep = state.pop("report")
        super()._set_state(state)
        self.report_ = SelectionReport(
            methods=tuple(rep["methods"]),  # type: ignore[index]
            score_mean=decode_value(rep["score_mean"]),  # type: ignore[index, arg-type]
            score_sd=decode_value(rep["score_sd"]),  # type: ignore[index, arg-type]
            guardrails_ok=decode_value(rep["guardrails_ok"]),  # type: ignore[index, arg-type]
            chosen=decode_value(rep["chosen"]),  # type: ignore[index, arg-type]
            criterion=rep["criterion"],  # type: ignore[index]
            # .get(..., None): the 0.2.0 golden predates these columns.
            mcb=decode_value(rep.get("mcb")),  # type: ignore[attr-defined, arg-type]
            dsc=decode_value(rep.get("dsc")),  # type: ignore[attr-defined, arg-type]
            unc=decode_value(rep.get("unc")),  # type: ignore[attr-defined, arg-type]
        )

    def _params_for_dict(self) -> dict[str, object]:
        """Encode candidate prototypes by registry name.

        Custom candidate classes outside the registry cannot be rebuilt on
        load and raise ``ValueError`` naming the class.
        """
        params = dict(self.get_params())
        cands = params.get("candidates")
        if cands is not None:
            enc = {}
            for name, proto in cands.items():  # type: ignore[attr-defined]
                cls_name = type(proto).__name__
                if cls_name not in SERIALIZABLE:
                    raise ValueError(
                        f"cannot serialize candidate {name!r}: {cls_name} is not a "
                        "registered probcal class"
                    )
                to_params = getattr(proto, "_params_for_dict", proto.get_params)
                enc[name] = {"class": cls_name, "params": to_params()}
            params["candidates"] = {"__candidates__": enc}
        return params

    @classmethod
    def _params_from_dict(cls, params: dict[str, object]) -> dict[str, object]:
        params = dict(params)
        cands = params.get("candidates")
        if isinstance(cands, dict) and "__candidates__" in cands:
            built = {}
            for name, spec in cands["__candidates__"].items():  # type: ignore[attr-defined]
                proto_cls = SERIALIZABLE[spec["class"]]
                from_params = getattr(proto_cls, "_params_from_dict", dict)
                built[name] = proto_cls(**from_params(dict(spec["params"])))
            params["candidates"] = built
        return params

    def _fit(self, s_arr: np.ndarray, y_arr: np.ndarray, w_arr: np.ndarray) -> None:
        if self.scoring not in _SCORERS:
            raise ValueError(
                f"scoring must be one of {sorted(_SCORERS)} (proper scores and accepted "
                f"binning-free alternatives), got {self.scoring!r}; plain ECE and "
                "Hosmer-Lemeshow are not selection criteria"
            )
        scorer = _SCORERS[self.scoring]
        cv = validate_cv(self.cv, y_arr)
        menu = self.candidates if self.candidates is not None else _default_candidates()
        # CORP decomposition to attach to the report: Brier when the selection
        # criterion itself is Brier, log loss otherwise (matches the scorer
        # actually driving the ranking as closely as the two-way CORP split allows).
        corp_score = "brier" if self.scoring == "brier" else "log_loss"

        folds = stratified_folds(y_arr, cv, self.random_state)

        names = list(menu)
        means = np.empty(len(names))
        sds = np.empty(len(names))
        guards = np.empty(len(names), dtype=bool)
        mcbs = np.empty(len(names))
        dscs = np.empty(len(names))
        unc = 0.0
        for i, name in enumerate(names):
            proto = menu[name]
            fold_scores = np.empty(cv)
            oof = np.empty(len(y_arr))
            for k in range(cv):
                train, held = folds != k, folds == k
                cal = cast(BaseCalibrator, clone_unfitted(proto))
                cal.fit(s_arr[train], y_arr[train], sample_weight=w_arr[train])
                pred_held = cal.predict_proba(s_arr[held])
                oof[held] = pred_held
                fold_scores[k] = scorer(y_arr[held], pred_held, sample_weight=w_arr[held])
            means[i] = fold_scores.mean()
            sds[i] = fold_scores.std(ddof=1)
            guards[i] = calibration_guardrails(y_arr, oof, sample_weight=w_arr).all_ok
            # One corp_fit/decompose per candidate on its own OOF vector — cheap
            # relative to the cv-fold refits above.
            _, _, _, _, pav = corp_fit(y_arr, oof, w_arr)
            _, mcbs[i], dscs[i], unc = decompose(y_arr, oof, pav, w_arr, corp_score)

        # Parsimony tie-break within one standard error of the best mean.
        best_idx = int(np.argmin(means))
        se_best = sds[best_idx] / np.sqrt(cv)
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
            mcb=mcbs[order],
            dsc=dscs[order],
            unc=float(unc),
        )
        self.best_name_ = names[winner]
        self.best_calibrator_: BaseCalibrator = cast(
            BaseCalibrator, clone_unfitted(menu[self.best_name_])
        )
        self.best_calibrator_.fit(s_arr, y_arr, sample_weight=w_arr)

    # ------------------------------------------------------------- delegation

    def _winner(self) -> BaseCalibrator | None:
        return self.__dict__.get("best_calibrator_")

    def _predict(self, s: np.ndarray) -> np.ndarray:
        """Delegate to the refitted winner."""
        return self.best_calibrator_.predict_proba(s)

    def interpret(self) -> Interpretation:
        """Delegate to the refitted winner."""
        self._check_fitted()
        return self.best_calibrator_.interpret()

    @property  # type: ignore[override]
    def is_monotone_(self) -> bool:  # type: ignore[override]
        """The winner's ``is_monotone_`` (``True`` before fitting)."""
        best = self._winner()
        return True if best is None else bool(getattr(best, "is_monotone_", True))

    @property
    def affine_logit_coeffs_(self) -> tuple[float, float] | None:
        """The winner's affine-logit coefficients, or ``None``."""
        self._check_fitted()
        return self.best_calibrator_.affine_logit_coeffs_

    @property
    def complexity_rank(self) -> float:
        """The winner's parsimony rank once fitted; 100.0 before."""
        best = self._winner()
        return 100.0 if best is None else float(getattr(best, "complexity_rank", 100.0))

    def interval_inverse(
        self,
        lo: float,
        hi: float,
        *,
        space: str = "probability",
        buffer_logit: float = 0.0,
    ) -> tuple[float, float]:
        """Delegate to the winner's exact ``interval_inverse``."""
        self._check_fitted()
        return self.best_calibrator_.interval_inverse(
            lo, hi, space=space, buffer_logit=buffer_logit
        )

    def point_inverse(self, p: object, *, space: str = "probability") -> np.ndarray:
        """Delegate to the winner's ``point_inverse``."""
        self._check_fitted()
        return self.best_calibrator_.point_inverse(p, space=space)

    def _point_inverse_logit(self, p: np.ndarray) -> np.ndarray:
        return self.best_calibrator_._point_inverse_logit(p)

    def _output_range(self) -> tuple[float, float]:
        return self.best_calibrator_._output_range()

    def _inverse_left(self, t: float) -> float:
        return self.best_calibrator_._inverse_left(t)

    def _inverse_right(self, t: float) -> float:
        return self.best_calibrator_._inverse_right(t)
