"""CalibratedModel: model-level wrapper with prefit and cross-validation flows.

Theory of the flows (why prefit is the credit-risk canon, why the pooled cv
variant is the recommended default): ``docs/concepts/data-splitting.md``.
"""

import copy
import inspect
import json
import warnings
from datetime import UTC, datetime
from typing import Any, Self

import numpy as np

from ._math import expit
from ._registry import register
from ._results import Interpretation
from ._serialize import JsonIO, check_payload, data_fingerprint, encode_value, envelope
from ._validation import stratified_folds, validate_binary_y, validate_cv, validate_weights
from .base import BaseCalibrator, clone_unfitted
from .chain import Chain, _concat_interpretations
from .offset import LogitOffset


def _model_scores(model: Any, X: object) -> np.ndarray:
    """Duck-typed score extraction: predict_proba column 1, else expit(margin).

    ``X`` is handed to the model untouched (a DataFrame stays a DataFrame).
    """
    if hasattr(model, "predict_proba"):
        out = np.asarray(model.predict_proba(X), dtype=np.float64)
        if out.ndim == 2:
            return out[:, 1]
        return out
    if hasattr(model, "decision_function"):
        return expit(np.asarray(model.decision_function(X), dtype=np.float64))
    raise TypeError(
        f"model must expose predict_proba(X) or decision_function(X); got {type(model).__name__}"
    )


def _n_rows(X: object) -> int:
    """Row count of ``X`` without converting it (``shape[0]``, else ``len``)."""
    shape = getattr(X, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[0])
    return len(X)  # type: ignore[arg-type]


def _take_rows(X: object, mask: np.ndarray) -> object:
    """Rows of ``X`` selected by a boolean mask, keeping its type where possible.

    pandas objects go through ``.iloc``; anything array-like with a
    ``shape`` (ndarrays, scipy sparse) is indexed directly; a plain sequence
    becomes an ndarray with numpy's own dtype inference first (strings stay
    strings — no float coercion).
    """
    idx = np.flatnonzero(mask)
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    if hasattr(X, "shape") and hasattr(X, "__getitem__"):
        return X[idx]
    return np.asarray(X)[idx]


def _accepts_sample_weight(fit: Any) -> bool:
    """Whether ``fit`` takes a ``sample_weight`` keyword (named or via ``**kwargs``)."""
    try:
        params = inspect.signature(fit).parameters
    except (TypeError, ValueError):
        return False
    return "sample_weight" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


def _clone(model: Any) -> Any:
    """sklearn.base.clone when sklearn is installed, deepcopy otherwise."""
    try:
        from sklearn.base import clone  # runtime-optional; never a module-level import

        return clone(model)
    except Exception:
        return copy.deepcopy(model)


@register
class CalibratedModel(JsonIO):
    """Wrap any scoring model with a probcal calibrator (and optional offsets).

    Parameters
    ----------
    model : object
        Duck-typed model with ``predict_proba(X)`` or ``decision_function(X)``.
        For ``flow="cv"`` it must also have ``fit(X, y)`` and be clonable.
        ``X`` is always passed to the model as given (never converted), so
        DataFrames, column transformers, and string features work.
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
        Fold count for the cv flow (stratified, seeded); an integer ``>= 2``.
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

    _weight_warned: bool = False

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
        X : array_like, DataFrame, or any model input
            Calibration-set inputs, passed untouched to the model
            (``flow="cv"``: row subsets via ``.iloc`` or numpy indexing) or
            scored directly by the already-trained model (``flow="prefit"``).
        y : array_like
            Binary outcomes in ``{0, 1}``; both classes must be present.
        sample_weight : array_like or None
            Positive observation weights. Always used by the calibrator; in
            the cv flow also forwarded to ``model.fit`` when its signature
            accepts ``sample_weight`` (otherwise a ``UserWarning`` says the
            model is trained unweighted).

        Returns
        -------
        Self
            The fitted wrapper.

        Raises
        ------
        ValueError
            If ``flow`` is not ``"prefit"`` or ``"cv"``, ``cv`` is not an
            integer ``>= 2`` (or a class has fewer than 2 members), or ``X``
            and ``y`` differ in length.
        TypeError
            If ``flow="cv"`` and the model has no ``fit(X, y)`` method.
        """
        if self.flow not in ("prefit", "cv"):
            raise ValueError(f"flow must be 'prefit' or 'cv', got {self.flow!r}")
        y_arr = validate_binary_y(y)
        if _n_rows(X) != len(y_arr):
            raise ValueError(f"X has {_n_rows(X)} rows but y has {len(y_arr)}")
        w_arr = validate_weights(sample_weight, len(y_arr))
        self.offsets_: list[LogitOffset] = []
        self.ensemble_: list[tuple[Any, BaseCalibrator]] = []
        if self.flow == "prefit":
            self.model_ = self.model
            s = _model_scores(self.model_, X)
            self.calibrator_ = self._fresh_calibrator().fit(s, y_arr, sample_weight=w_arr)
            self._cal_scores = s
        else:
            cv = validate_cv(self.cv, y_arr)
            self._fit_cv(X, y_arr, w_arr if sample_weight is not None else None, cv)
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
        return clone_unfitted(self.calibrator)  # type: ignore[return-value]

    def _fit_model(self, model: Any, X: object, y: np.ndarray, w: np.ndarray | None) -> None:
        """``model.fit(X, y[, sample_weight=w])``; warns once when weights are dropped."""
        if w is None:
            model.fit(X, y)
        elif _accepts_sample_weight(model.fit):
            model.fit(X, y, sample_weight=w)
        else:
            if not self._weight_warned:
                warnings.warn(
                    f"{type(model).__name__}.fit does not accept sample_weight; the model is "
                    "trained unweighted (the calibrator still uses the weights)",
                    UserWarning,
                    stacklevel=4,
                )
                self._weight_warned = True
            model.fit(X, y)

    def _fit_cv(self, X: object, y: np.ndarray, w: np.ndarray | None, cv: int) -> None:
        if not hasattr(self.model, "fit"):
            raise TypeError("flow='cv' requires a model with fit(X, y)")
        self._weight_warned = False
        w_cal = w if w is not None else np.ones(len(y))
        folds = stratified_folds(y, cv, self.random_state)
        oof_scores = np.empty(len(y))
        for k in range(cv):
            train, held = folds != k, folds == k
            fold_model = _clone(self.model)
            self._fit_model(
                fold_model, _take_rows(X, train), y[train], None if w is None else w[train]
            )
            s_held = _model_scores(fold_model, _take_rows(X, held))
            oof_scores[held] = s_held
            if self.ensemble:
                fold_cal = self._fresh_calibrator()
                fold_cal.fit(s_held, y[held], sample_weight=w_cal[held])
                self.ensemble_.append((fold_model, fold_cal))
        self._cal_scores = oof_scores
        if self.ensemble:
            self.model_ = None
            self.calibrator_ = None  # type: ignore[assignment]
        else:
            self.calibrator_ = self._fresh_calibrator().fit(oof_scores, y, sample_weight=w_cal)
            self.model_ = _clone(self.model)
            self._fit_model(self.model_, X, y, w)

    # ------------------------------------------------------------------ prediction

    def _check_fitted(self) -> None:
        if not getattr(self, "fitted_", False):
            raise RuntimeError("CalibratedModel is not fitted; call fit() first")

    def __sklearn_is_fitted__(self) -> bool:
        """Fitted state for sklearn >= 1.6 (model and calibrator both fitted)."""
        return bool(getattr(self, "fitted_", False))

    def _base_predict(self, X: object) -> np.ndarray:
        if self.ensemble_:
            preds = [cal.predict_proba(_model_scores(model, X)) for model, cal in self.ensemble_]
            return np.mean(preds, axis=0)
        return self.calibrator_.predict_proba(_model_scores(self.model_, X))

    def predict_proba(self, X: object) -> np.ndarray:
        """Calibrated (and offset) probabilities ``P(y=1)`` for new inputs.

        Parameters
        ----------
        X : array_like, DataFrame, or any model input
            New inputs, passed untouched to the deployed model (or every
            ensemble fold's model, averaged).

        Returns
        -------
        numpy.ndarray of shape (n,)
            Calibrated probabilities, after any appended offset stages.
        """
        self._check_fitted()
        p = self._base_predict(X)
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
        *args: object,
        target_mean: float | None = None,
        delta: float | None = None,
        X: object = None,
    ) -> Self:
        """Append an inspectable :class:`LogitOffset` stage.

        Mode B (``target_mean``) anchors the portfolio mean of the current
        pipeline output — computed on ``X`` when given, else on the stored
        calibration scores. The offset is never folded into
        the calibrator's parameters.

        Positional use (``offset_to(0.03)``, the 0.3 signature
        ``(target_mean, delta, X)``) still works but emits a
        ``DeprecationWarning``; it will be removed in 0.5.0.

        Parameters
        ----------
        target_mean : float or None, keyword-only
            Mode B: desired post-shift portfolio mean; mutually exclusive
            with ``delta`` (enforced by :class:`LogitOffset`).
        delta : float or None, keyword-only
            Mode A: the log-odds shift to apply directly; mutually exclusive
            with ``target_mean``.
        X : array_like or None, keyword-only
            Inputs to compute the current pipeline output on; ``None`` uses
            the stored calibration scores instead.

        Returns
        -------
        Self
            The wrapper, with the new offset appended to ``offsets_``.
        """
        if args:
            if len(args) > 3:
                raise TypeError("offset_to takes at most 3 positional arguments (deprecated)")
            warnings.warn(
                "positional arguments to CalibratedModel.offset_to are deprecated and will "
                "be removed in 0.5.0; use offset_to(target_mean=..., delta=..., X=...)",
                DeprecationWarning,
                stacklevel=2,
            )
            given = dict(zip(("target_mean", "delta", "X"), args, strict=False))
            target_mean = given.get("target_mean", target_mean)  # type: ignore[assignment]
            delta = given.get("delta", delta)  # type: ignore[assignment]
            X = given.get("X", X)
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

    def with_offset(self, offset: LogitOffset) -> "CalibratedModel":
        """A deep copy of this wrapper with an already-fitted offset appended.

        Needs no data: unlike :meth:`offset_to`, nothing is fitted — the
        given offset (e.g. from :func:`probcal.offset_from_estimate` or a
        monitor recommendation) is copied and appended as the last stage.
        ``self`` is left untouched.

        Parameters
        ----------
        offset : LogitOffset
            A fitted offset.

        Returns
        -------
        CalibratedModel
            The new wrapper (``offsets_ == [*self.offsets_, offset]``, copied).

        Raises
        ------
        TypeError
            If ``offset`` is not a :class:`LogitOffset`.
        RuntimeError
            If this wrapper or the offset is not fitted.
        """
        self._check_fitted()
        if not isinstance(offset, LogitOffset):
            raise TypeError(f"offset must be a LogitOffset, got {type(offset).__name__}")
        offset._check_fitted()
        new = copy.deepcopy(self)
        new.offsets_.append(copy.deepcopy(offset))
        return new

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
    # Everything below delegates to the equivalent Chain (calibrator + offsets);
    # the ensemble flow (K distinct maps) has no single chain.

    def _require_chain(self, what: str) -> Chain:
        self._check_fitted()
        if self.ensemble_:
            raise NotImplementedError(
                f"{what} is not defined for the ensemble flow (K distinct maps); "
                "use ensemble=False for threshold translation"
            )
        return self.chain_

    @property
    def is_monotone_(self) -> bool:
        """Monotone iff the calibrator stage is (offsets always are)."""
        self._check_fitted()
        if self.ensemble_:
            return all(cal.is_monotone_ for _, cal in self.ensemble_)
        return self.chain_.is_monotone_

    @property
    def affine_logit_coeffs_(self) -> tuple[float, float] | None:
        """Composed ``(a, b + sum(deltas))`` when the calibrator stage is affine."""
        self._check_fitted()
        if self.ensemble_:
            return None
        return self.chain_.affine_logit_coeffs_

    def interval_inverse(
        self,
        lo: float,
        hi: float,
        *,
        space: str = "probability",
        buffer_logit: float = 0.0,
    ) -> tuple[float, float]:
        """Preimage of a calibrated interval through the full pipeline.

        Delegates to :meth:`Chain.interval_inverse` on :attr:`chain_`: the
        buffer shrinks the final interval, each offset subtracts its delta
        on the logit scale, and the calibrator's own inverse finishes the
        job. Returns bounds on the model's probability output
        (``space="probability"``) or their logits.

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
        chain = self._require_chain("interval_inverse")
        return chain.interval_inverse(lo, hi, space=space, buffer_logit=buffer_logit)

    def point_inverse(self, p: object, *, space: str = "probability") -> np.ndarray:
        """Exact preimage of pipeline probabilities (see :meth:`Chain.point_inverse`).

        Raises
        ------
        NotImplementedError
            For the ensemble flow, or a calibrator without an exact inverse.
        UnattainableTargetError
            If a target lies outside ``(0, 1)`` or is not representable.
        """
        chain = self._require_chain("point_inverse")
        return chain.point_inverse(p, space=space)

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
            parts += [off.interpret() for off in self.offsets_]
        else:
            parts = [stage.interpret() for stage in self.chain_.stages]  # type: ignore[attr-defined]
        return _concat_interpretations("CalibratedModel", parts, qualify=False)

    @property
    def chain_(self) -> Chain:
        """The equivalent model-free :class:`probcal.Chain` (calibrator + offsets).

        Hand this to a recourse engine when the base model stays behind:
        the chain calibrates on the model *probability*, so its
        ``space="logit"`` bounds are bounds on the raw margin.

        The returned chain aliases this wrapper's own fitted calibrator and
        offsets rather than copying them, so calling ``fit`` on the chain
        refits this :class:`CalibratedModel`'s calibrator and offsets in
        place.
        """
        self._check_fitted()
        if self.ensemble_:
            raise NotImplementedError("the ensemble flow has no single equivalent chain")
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
        ref_model = self.model_ if self.model_ is not None else self.model
        model_params: object = None
        if hasattr(ref_model, "get_params"):
            try:
                candidate = ref_model.get_params()
                json.dumps(candidate, allow_nan=False)
                model_params = candidate
            except (TypeError, ValueError):
                model_params = None
        return envelope(
            self,
            params={
                "flow": self.flow,
                "cv": self.cv,
                "ensemble": self.ensemble,
                "random_state": self.random_state,
                "model_id": self.model_id,
            },
            state={
                "calibrator": self.calibrator_.to_dict(),
                "offsets": [off.to_dict() for off in self.offsets_],
                "model_ref": {
                    "class_name": type(ref_model).__name__,
                    "model_id": self.model_id,
                    "params": model_params,
                },
            },
            fit_meta=encode_value(dict(getattr(self, "fit_meta_", {}))),
        )

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
        check_payload(cls, d)
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

    @classmethod
    def from_json(cls, path_or_str: object, model: Any = None) -> Self:  # type: ignore[override]
        """Load from a JSON string or a filesystem path (see :meth:`from_dict`)."""
        return super().from_json(path_or_str, model=model)
