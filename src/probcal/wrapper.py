"""CalibratedModel: model-level wrapper with prefit and cross-validation flows.

Theory of the flows (why prefit is the credit-risk canon, why the pooled cv
variant is the recommended default): ``docs/concepts/data-splitting.md``.
"""

import copy
import json
import os
from datetime import UTC, datetime
from typing import Any, Self

import numpy as np

from ._math import expit, logit
from ._registry import register
from ._results import Interpretation
from ._serialize import SCHEMA_VERSION, check_schema, data_fingerprint, fingerprint_of_dict
from ._validation import validate_binary_y, validate_weights
from .base import BaseCalibrator, UnattainableTargetError
from .offset import LogitOffset


def _model_scores(model: Any, X: np.ndarray) -> np.ndarray:
    """Duck-typed score extraction: predict_proba column 1, else expit(margin)."""
    if hasattr(model, "predict_proba"):
        out = np.asarray(model.predict_proba(X), dtype=np.float64)
        if out.ndim == 2:
            return out[:, 1]
        return out
    if hasattr(model, "decision_function"):
        return expit(np.asarray(model.decision_function(X), dtype=np.float64))
    raise TypeError(
        "model must expose predict_proba(X) or decision_function(X); " f"got {type(model).__name__}"
    )


def _clone(model: Any) -> Any:
    """sklearn.base.clone when sklearn is installed, deepcopy otherwise (DECISIONS 48)."""
    try:
        from sklearn.base import clone  # runtime-optional; never a module-level import

        return clone(model)
    except Exception:
        return copy.deepcopy(model)


@register
class CalibratedModel:
    """Wrap any scoring model with a probcal calibrator (and optional offsets).

    Parameters
    ----------
    model : object
        Duck-typed model with ``predict_proba(X)`` or ``decision_function(X)``.
        For ``flow="cv"`` it must also have ``fit(X, y)`` and be clonable.
    calibrator : BaseCalibrator
        Unfitted calibrator instance (its parameters are cloned per fold in
        the cv flow via ``get_params``).
    flow : {"prefit", "cv"}
        ``"prefit"``: the model is already trained; ``fit(X_cal, y_cal)``
        scores the calibration set and fits the calibrator — the canonical
        credit-risk flow. ``"cv"``: the model is cloned and retrained per
        fold; every observation is scored by a model that did not train on
        it.
    cv : int
        Fold count for the cv flow (stratified, seeded).
    ensemble : bool
        ``False`` (recommended default): one calibrator on pooled
        out-of-fold scores, final model refit on all data — a single
        auditable mapping. ``True``: keep the per-fold (model, calibrator)
        pairs and average their predictions.
    random_state : int
        Seed for the fold assignment.

    Attributes
    ----------
    model_ : object
        The deployed model (the input model for prefit; the full-data refit
        for pooled cv).
    calibrator_ : BaseCalibrator
        The fitted calibrator (pooled/prefit flows).
    ensemble_ : list[tuple[model, BaseCalibrator]]
        The fold pairs (ensemble flow only).
    offsets_ : list[LogitOffset]
        Appended offset stages, each separately inspectable.
    """

    def __init__(
        self,
        model: Any,
        calibrator: BaseCalibrator,
        flow: str = "prefit",
        cv: int = 5,
        ensemble: bool = False,
        random_state: int = 42,
        *,
        model_id: str | None = None,
    ) -> None:
        self.model = model
        self.calibrator = calibrator
        self.flow = flow
        self.cv = cv
        self.ensemble = ensemble
        self.random_state = random_state
        self.model_id = model_id

    # ------------------------------------------------------------------ fitting

    def fit(self, X: object, y: object, sample_weight: object = None) -> Self:
        """Fit the calibration stage per the configured flow.

        Parameters
        ----------
        X : array_like
            Calibration-set inputs, passed to the model (``flow="cv"``) or
            scored directly by the already-trained model (``flow="prefit"``).
        y : array_like
            Binary outcomes in ``{0, 1}``; both classes must be present.
        sample_weight : array_like or None
            Positive observation weights.

        Returns
        -------
        Self
            The fitted wrapper.

        Raises
        ------
        ValueError
            If ``flow`` is not ``"prefit"`` or ``"cv"``.
        TypeError
            If ``flow="cv"`` and the model has no ``fit(X, y)`` method.
        """
        if self.flow not in ("prefit", "cv"):
            raise ValueError(f"flow must be 'prefit' or 'cv', got {self.flow!r}")
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = validate_binary_y(y)
        w_arr = validate_weights(sample_weight, len(y_arr))
        self.offsets_: list[LogitOffset] = []
        self.ensemble_: list[tuple[Any, BaseCalibrator]] = []
        if self.flow == "prefit":
            self.model_ = self.model
            s = _model_scores(self.model_, X_arr)
            self.calibrator_ = self._fresh_calibrator().fit(s, y_arr, sample_weight=w_arr)
            self._cal_scores = s
        else:
            self._fit_cv(X_arr, y_arr, w_arr)
        self.fit_meta_ = {
            "n_obs": int(len(y_arr)),
            "n_events": float(np.sum(w_arr * y_arr)),
            "weight_sum": float(w_arr.sum()),
            "fitted_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "data_fingerprint": data_fingerprint(self._cal_scores, y_arr, w_arr),
        }
        self.fitted_ = True
        return self

    def _fresh_calibrator(self) -> BaseCalibrator:
        return type(self.calibrator)(**self.calibrator.get_params())

    def _fit_cv(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        if not hasattr(self.model, "fit"):
            raise TypeError("flow='cv' requires a model with fit(X, y)")
        rng = np.random.default_rng(self.random_state)
        folds = np.empty(len(y), dtype=np.int64)
        for cls in (0.0, 1.0):
            idx = np.flatnonzero(y == cls)
            perm = rng.permutation(idx)
            folds[perm] = np.arange(len(perm)) % self.cv
        oof_scores = np.empty(len(y))
        for k in range(self.cv):
            train, held = folds != k, folds == k
            fold_model = _clone(self.model)
            fold_model.fit(X[train], y[train])
            s_held = _model_scores(fold_model, X[held])
            oof_scores[held] = s_held
            if self.ensemble:
                fold_cal = self._fresh_calibrator()
                fold_cal.fit(s_held, y[held], sample_weight=w[held])
                self.ensemble_.append((fold_model, fold_cal))
        if self.ensemble:
            self.model_ = None
            self.calibrator_ = None  # type: ignore[assignment]
            self._cal_scores = oof_scores
        else:
            self.calibrator_ = self._fresh_calibrator().fit(oof_scores, y, sample_weight=w)
            self.model_ = _clone(self.model)
            self.model_.fit(X, y)
            self._cal_scores = oof_scores

    # ------------------------------------------------------------------ prediction

    def _check_fitted(self) -> None:
        if not getattr(self, "fitted_", False):
            raise RuntimeError("CalibratedModel is not fitted; call fit() first")

    def _base_predict(self, X: np.ndarray) -> np.ndarray:
        if self.ensemble_:
            preds = [cal.predict_proba(_model_scores(model, X)) for model, cal in self.ensemble_]
            return np.mean(preds, axis=0)
        return self.calibrator_.predict_proba(_model_scores(self.model_, X))

    def predict_proba(self, X: object) -> np.ndarray:
        """Calibrated (and offset) probabilities ``P(y=1)`` for new inputs.

        Parameters
        ----------
        X : array_like
            New inputs, passed to the deployed model (or every ensemble
            fold's model, averaged).

        Returns
        -------
        numpy.ndarray of shape (n,)
            Calibrated probabilities, after any appended offset stages.
        """
        self._check_fitted()
        p = self._base_predict(np.asarray(X, dtype=np.float64))
        for off in self.offsets_:
            p = off.transform(p)
        return p

    def predict_proba_2d(self, X: object) -> np.ndarray:
        """Sklearn-style ``(n, 2)`` probability matrix."""
        p = self.predict_proba(X)
        return np.column_stack([1.0 - p, p])

    # ------------------------------------------------------------------ offsets

    def offset_to(
        self,
        target_mean: float | None = None,
        delta: float | None = None,
        X: object = None,
    ) -> Self:
        """Append an inspectable :class:`LogitOffset` stage.

        Mode B (``target_mean``) anchors the portfolio mean of the current
        pipeline output — computed on ``X`` when given, else on the stored
        calibration scores (DECISIONS 48). The offset is never folded into
        the calibrator's parameters.

        Parameters
        ----------
        target_mean : float or None
            Mode B: desired post-shift portfolio mean; mutually exclusive
            with ``delta`` (enforced by :class:`LogitOffset`).
        delta : float or None
            Mode A: the log-odds shift to apply directly; mutually exclusive
            with ``target_mean``.
        X : array_like or None
            Inputs to compute the current pipeline output on; ``None`` uses
            the stored calibration scores instead.

        Returns
        -------
        Self
            The wrapper, with the new offset appended to ``offsets_``.
        """
        self._check_fitted()
        if X is not None:
            p_now = self.predict_proba(X)
        else:
            if self._cal_scores is None:
                raise RuntimeError(
                    "calibration scores are not serialized; after from_dict, pass X so "
                    "the offset stage can record its audit means on current output"
                )
            p_now = self._base_predict_from_scores(self._cal_scores)
        off = LogitOffset(delta=delta, target_mean=target_mean)
        off.fit(p_now)
        self.offsets_.append(off)
        return self

    def _base_predict_from_scores(self, s: np.ndarray) -> np.ndarray:
        if self.ensemble_:
            preds = [cal.predict_proba(s) for _, cal in self.ensemble_]
            p = np.mean(preds, axis=0)
        else:
            p = self.calibrator_.predict_proba(s)
        for off in self.offsets_:
            p = off.transform(p)
        return p

    # ------------------------------------------------------------------ protocol

    @property
    def is_monotone_(self) -> bool:
        """Monotone iff the calibrator stage is (offsets always are)."""
        if self.ensemble_:
            return all(cal.is_monotone_ for _, cal in self.ensemble_)
        return self.calibrator_.is_monotone_

    @property
    def affine_logit_coeffs_(self) -> tuple[float, float] | None:
        """Composed ``(a, b + sum(deltas))`` when the calibrator stage is affine."""
        self._check_fitted()
        if self.ensemble_:
            return None
        coeffs = self.calibrator_.affine_logit_coeffs_
        if coeffs is None:
            return None
        a, b = coeffs
        return (a, b + sum(off.delta_ for off in self.offsets_))

    def interval_inverse(
        self,
        lo: float,
        hi: float,
        *,
        space: str = "probability",
        buffer_logit: float = 0.0,
    ) -> tuple[float, float]:
        """Preimage of a calibrated interval through the full pipeline.

        Composes right-to-left: the buffer shrinks the final interval, each
        offset subtracts its delta on the logit scale, and the calibrator's
        own inverse finishes the job. Returns bounds on the model's
        probability output (``space="probability"``) or their logits.

        Parameters
        ----------
        lo, hi : float
            Calibrated-probability bounds; ``lo=0``/``hi=1`` map to the full
            raw range (−inf/+inf on the logit scale).
        space : {"probability", "logit"}, keyword-only
            Scale of the returned raw bounds.
        buffer_logit : float, keyword-only
            Shrink the calibrated interval by this margin in logit space
            before inverting.

        Returns
        -------
        tuple of float
            ``(raw_lo, raw_hi)`` preimage bounds, on the scale requested by
            ``space``.

        Raises
        ------
        NotImplementedError
            If the wrapper was fitted with ``ensemble=True`` (K distinct
            maps have no single preimage).
        UnattainableTargetError
            If the (buffered) interval does not intersect the pipeline's
            output range.
        ValueError
            If ``lo``, ``hi`` are not ordered in ``[0, 1]``.
        """
        self._check_fitted()
        if self.ensemble_:
            raise NotImplementedError(
                "interval_inverse is not defined for the ensemble flow (K distinct maps); "
                "use ensemble=False for threshold translation"
            )
        if not 0.0 <= lo <= hi <= 1.0:
            raise ValueError(f"need 0 <= lo <= hi <= 1, got lo={lo}, hi={hi}")
        lo_b, hi_b = float(lo), float(hi)
        if buffer_logit > 0.0:
            if lo > 0.0:
                lo_b = float(expit(np.array([logit(np.array([lo]))[0] + buffer_logit]))[0])
            if hi < 1.0:
                hi_b = float(expit(np.array([logit(np.array([hi]))[0] - buffer_logit]))[0])
            if lo_b > hi_b:
                raise UnattainableTargetError(
                    f"buffer_logit={buffer_logit} empties the calibrated interval [{lo}, {hi}]"
                )
        total_delta = sum(off.delta_ for off in self.offsets_)
        if total_delta != 0.0:
            if lo_b > 0.0:
                lo_b = float(expit(np.array([logit(np.array([lo_b]))[0] - total_delta]))[0])
            if hi_b < 1.0:
                hi_b = float(expit(np.array([logit(np.array([hi_b]))[0] - total_delta]))[0])
        return self.calibrator_.interval_inverse(lo_b, hi_b, space=space, buffer_logit=0.0)

    def interpret(self) -> Interpretation:
        """Concatenated interpretation of the calibrator and every offset stage.

        Returns
        -------
        Interpretation
            Parameters and messages concatenated across the calibrator
            stage(s) and every appended offset, in application order.
        """
        self._check_fitted()
        if self.ensemble_:
            parts = [cal.interpret() for _, cal in self.ensemble_]
        else:
            parts = [self.calibrator_.interpret()]
        parts += [off.interpret() for off in self.offsets_]
        names: tuple[str, ...] = ()
        values: tuple[float, ...] = ()
        messages: tuple[str, ...] = ()
        for part in parts:
            names += part.param_names
            values += part.param_values
            messages += part.messages
        return Interpretation(
            method=f"CalibratedModel[{', '.join(p.method for p in parts)}]",
            param_names=names,
            param_values=values,
            messages=messages,
        )

    @property
    def chain_(self) -> "object":
        """The equivalent model-free :class:`probcal.Chain` (calibrator + offsets).

        Hand this to a recourse engine when the base model stays behind:
        the chain calibrates on the model *probability*, so its
        ``space="logit"`` bounds are bounds on the raw margin.
        """
        self._check_fitted()
        if self.ensemble_:
            raise NotImplementedError("the ensemble flow has no single equivalent chain")
        from .chain import Chain

        return Chain([self.calibrator_, *self.offsets_])

    # ------------------------------------------------------------------ serialization

    def to_dict(self) -> dict[str, object]:
        """Versioned snapshot: nested calibrator, offsets, and a model *reference*.

        The base model is never serialized — only a reference (class name,
        the user-supplied ``model_id``, and ``get_params()`` when available
        and JSON-encodable); reattach it on load via
        ``CalibratedModel.from_dict(d, model=...)``. The stored calibration
        scores are not serialized either: after a reload,
        ``offset_to`` needs an explicit ``X``.

        Raises
        ------
        RuntimeError
            If not yet fitted.
        NotImplementedError
            For the ensemble flow: K fold models cannot be referenced.
        """
        self._check_fitted()
        if self.ensemble_:
            raise NotImplementedError(
                "the ensemble flow holds K fold models that cannot be referenced; "
                "serialize a pooled (ensemble=False) or prefit wrapper instead"
            )
        from . import __version__

        ref_model = self.model_ if self.model_ is not None else self.model
        model_params: object = None
        if hasattr(ref_model, "get_params"):
            try:
                candidate = ref_model.get_params()
                json.dumps(candidate)
                model_params = candidate
            except (TypeError, ValueError):
                model_params = None
        return {
            "probcal_schema": SCHEMA_VERSION,
            "probcal_version": __version__,
            "class": type(self).__name__,
            "params": {
                "flow": self.flow,
                "cv": self.cv,
                "ensemble": self.ensemble,
                "random_state": self.random_state,
                "model_id": self.model_id,
            },
            "state": {
                "calibrator": self.calibrator_.to_dict(),
                "offsets": [off.to_dict() for off in self.offsets_],
                "model_ref": {
                    "class_name": type(ref_model).__name__,
                    "model_id": self.model_id,
                    "params": model_params,
                },
            },
            "fit_meta": dict(getattr(self, "fit_meta_", {})),
        }

    @classmethod
    def from_dict(cls, d: dict, model: Any = None) -> "CalibratedModel":
        """Rebuild a fitted wrapper, reattaching the base model.

        Parameters
        ----------
        d : dict
            Output of :meth:`to_dict`.
        model : object or None
            The base model to reattach (matched against the stored reference
            is the caller's responsibility). With ``None``, the loaded
            wrapper can serialize and introspect but ``predict_proba``
            raises until a model is assigned to ``model_``.

        Raises
        ------
        ValueError
            If the schema version is unknown or the payload class differs.
        """
        check_schema(d)
        if d.get("class") != cls.__name__:
            raise ValueError(f"payload was written by {d.get('class')!r}, not {cls.__name__}")
        from ._registry import load

        params = d.get("params", {})
        state = d.get("state", {})
        calibrator = load(state["calibrator"])
        obj = cls(
            model,
            calibrator,  # type: ignore[arg-type]
            flow=params.get("flow", "prefit"),
            cv=params.get("cv", 5),
            ensemble=params.get("ensemble", False),
            random_state=params.get("random_state", 42),
            model_id=params.get("model_id"),
        )
        obj.calibrator_ = calibrator  # type: ignore[assignment]
        obj.offsets_ = [LogitOffset.from_dict(o) for o in state.get("offsets", [])]
        obj.ensemble_ = []
        obj.model_ = model
        obj._cal_scores = None  # type: ignore[assignment]
        obj.fit_meta_ = dict(d.get("fit_meta", {}))
        obj.fitted_ = True
        return obj

    def to_json(
        self, path: "str | os.PathLike[str] | None" = None, *, indent: int = 2
    ) -> str | None:
        """Serialize to JSON text, or to ``path`` when given (returns None then)."""
        text = json.dumps(self.to_dict(), indent=indent)
        if path is None:
            return text
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return None

    @classmethod
    def from_json(cls, path_or_str: object, model: Any = None) -> "CalibratedModel":
        """Load from a JSON string or a filesystem path (see :meth:`from_dict`)."""
        text = str(path_or_str)
        if not text.lstrip().startswith("{"):
            with open(text, encoding="utf-8") as fh:
                text = fh.read()
        return cls.from_dict(json.loads(text), model=model)

    def fingerprint(self) -> str:
        """SHA-256 of the canonical serialized form, version- and timestamp-blind."""
        return fingerprint_of_dict(self.to_dict())
