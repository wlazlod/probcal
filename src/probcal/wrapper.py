"""CalibratedModel: model-level wrapper with prefit and cross-validation flows.

Theory of the flows (why prefit is the credit-risk canon, why the pooled cv
variant is the recommended default): ``docs/concepts/data-splitting.md``.
"""

import copy
import inspect
import json
import sys
import warnings
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Self

import numpy as np

from ._math import expit
from ._protocol import CalibratorProtocolMixin
from ._registry import load, register
from ._results import Interpretation
from ._serialize import JsonIO, check_payload, data_fingerprint, encode_value, envelope
from ._validation import stratified_folds, validate_binary_y, validate_cv, validate_weights
from .base import BaseCalibrator, clone_unfitted
from .chain import Chain, _concat_interpretations
from .offset import LogitOffset


def _check_model_classes(model: Any) -> None:
    """A model exposing ``classes_`` must have been trained on the labels ``{0, 1}``.

    Column 1 of ``predict_proba`` (and the sign of a binary
    ``decision_function``) refers to ``classes_[1]``; :class:`CalibratedModel`
    calibrates against a 0/1 target, so any other labelling (e.g.
    ``["bad", "good"]``, where column 1 is ``P("good")``) would silently
    calibrate the wrong event.
    """
    classes = getattr(model, "classes_", None)
    if classes is None:
        return
    arr = np.asarray(classes)
    try:
        ok = arr.shape == (2,) and np.array_equal(arr.astype(np.float64), [0.0, 1.0])
    except (TypeError, ValueError):
        ok = False
    if not ok:
        raise ValueError(
            f"the model was trained on classes {list(arr.tolist())}, but CalibratedModel "
            "calibrates P(y=1) against a 0/1 target and reads the model's score for "
            "classes_[1]; retrain the model on 0/1 labels (1 = the event, e.g. default), "
            "or use probcal.sklearn.CalibratedClassifier, which keeps the original labels"
        )


def _model_scores(model: Any, X: object) -> np.ndarray:
    """Duck-typed score extraction: predict_proba column 1, else expit(margin).

    ``X`` is handed to the model untouched (a DataFrame stays a DataFrame).
    A model exposing ``classes_`` must have been trained on ``{0, 1}``
    labels, so that column 1 is ``P(y=1)``.
    """
    _check_model_classes(model)
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


def _accepts_sample_weight(estimator: Any) -> bool:
    """Whether ``estimator.fit`` can consume ``sample_weight``.

    A named ``sample_weight`` parameter counts (sklearn's
    ``has_fit_parameter`` test); a bare ``**kwargs`` does not — a
    ``Pipeline`` takes ``**params`` but rejects ``sample_weight`` unless
    metadata routing is enabled. Under sklearn's metadata routing a router
    (``Pipeline``, ``GridSearchCV``, ...) counts as capable, and sklearn's
    routing machinery has the final word: an undeclared request raises
    ``UnsetMetadataPassedError`` instead of dropping the weights. sklearn is
    consulted only when it is already imported (a router implies it is), so
    this never imports it.
    """
    try:
        params = inspect.signature(estimator.fit).parameters
    except (AttributeError, TypeError, ValueError):
        return False
    if "sample_weight" in params:
        return True
    if "sklearn" not in sys.modules:
        return False
    from sklearn import get_config  # already imported; see the docstring
    from sklearn.utils.metadata_routing import MetadataRouter, get_routing_for_object

    if get_config().get("enable_metadata_routing", False):
        return isinstance(get_routing_for_object(estimator), MetadataRouter)
    return False


def _clone(model: Any) -> Any:
    """sklearn.base.clone when sklearn is installed, deepcopy otherwise."""
    try:
        from sklearn.base import clone  # runtime-optional; never a module-level import

        return clone(model)
    except Exception:
        return copy.deepcopy(model)


@register
class CalibratedModel(CalibratorProtocolMixin, JsonIO):
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

    Notes
    -----
    ``interval_inverse`` and ``point_inverse`` invert the full pipeline
    (calibrator, then every offset) through :attr:`chain_` — see
    :meth:`Chain.interval_inverse` and :meth:`Chain.point_inverse`; the
    bounds are on the model's probability output (or its logit). The
    ensemble flow (K distinct maps) has no single preimage, so both raise
    ``NotImplementedError`` there.
    """

    _cal_scores: np.ndarray | None = None
    _cal_weights: np.ndarray | None = None
    _model_ref: dict[str, object] | None = None

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
            names ``sample_weight`` (or, with sklearn metadata routing on,
            the model is a router); otherwise a ``UserWarning`` says the
            model is trained unweighted. Also kept for
            :meth:`offset_to` without ``X``.

        Returns
        -------
        Self
            The fitted wrapper.

        Raises
        ------
        ValueError
            If ``flow`` is not ``"prefit"`` or ``"cv"``, ``cv`` is not an
            integer ``>= 2`` (or a class has fewer than 2 members; with
            ``ensemble=True``, fewer than ``cv`` members, since every fold's
            calibrator needs both classes), ``X`` and ``y`` differ in
            length, or the model exposes ``classes_`` other than ``{0, 1}``.
        TypeError
            If ``flow="cv"`` and the model has no ``fit(X, y)`` method.
        """
        if self.flow not in ("prefit", "cv"):
            raise ValueError(f"flow must be 'prefit' or 'cv', got {self.flow!r}")
        y_arr = validate_binary_y(y)
        if _n_rows(X) != len(y_arr):
            raise ValueError(f"X has {_n_rows(X)} rows but y has {len(y_arr)}")
        w_arr = validate_weights(sample_weight, len(y_arr))
        weighted = sample_weight is not None
        self.offsets_: list[LogitOffset] = []
        self.ensemble_: list[tuple[Any, BaseCalibrator]] = []
        if self.flow == "prefit":
            self.model_ = self.model
            s = _model_scores(self.model_, X)
            self.calibrator_ = self._fresh_calibrator().fit(s, y_arr, sample_weight=w_arr)
        else:
            cv = validate_cv(self.cv, y_arr, held_out_both_classes=self.ensemble)
            s = self._fit_cv(X, y_arr, w_arr, cv, weighted=weighted)
        self._cal_scores = s
        self._cal_weights = w_arr if weighted else None
        self.fit_meta_ = {
            "n_obs": int(len(y_arr)),
            "n_events": float(np.sum(w_arr * y_arr)),
            "weight_sum": float(w_arr.sum()),
            "fitted_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "data_fingerprint": data_fingerprint(s, y_arr, w_arr),
        }
        self.fitted_ = True
        return self

    def _fresh_calibrator(self) -> BaseCalibrator:
        return clone_unfitted(self.calibrator)  # type: ignore[return-value]

    def _fit_cv(
        self, X: object, y: np.ndarray, w: np.ndarray, cv: int, *, weighted: bool
    ) -> np.ndarray:
        """Cross-fitted flow; returns the pooled out-of-fold scores."""
        if not hasattr(self.model, "fit"):
            raise TypeError("flow='cv' requires a model with fit(X, y)")
        # Whether the model fits take the weights is decided once, up front.
        model_weighted = weighted and _accepts_sample_weight(self.model)
        if weighted and not model_weighted:
            warnings.warn(
                f"{type(self.model).__name__}.fit does not accept sample_weight; the model is "
                "trained unweighted (the calibrator still uses the weights)",
                UserWarning,
                stacklevel=3,
            )

        def fit_clone(rows: np.ndarray | None) -> Any:
            model = _clone(self.model)
            X_fit, y_fit, w_fit = (
                (X, y, w) if rows is None else (_take_rows(X, rows), y[rows], w[rows])
            )
            if model_weighted:
                model.fit(X_fit, y_fit, sample_weight=w_fit)
            else:
                model.fit(X_fit, y_fit)
            return model

        folds = stratified_folds(y, cv, self.random_state)
        oof_scores = np.empty(len(y))
        for k in range(cv):
            fold_model = fit_clone(folds != k)
            held = folds == k
            s_held = _model_scores(fold_model, _take_rows(X, held))
            oof_scores[held] = s_held
            if self.ensemble:
                fold_cal = self._fresh_calibrator()
                fold_cal.fit(s_held, y[held], sample_weight=w[held])
                self.ensemble_.append((fold_model, fold_cal))
        if self.ensemble:
            self.model_ = None
            self.calibrator_ = None  # type: ignore[assignment]
        else:
            self.calibrator_ = self._fresh_calibrator().fit(oof_scores, y, sample_weight=w)
            self.model_ = fit_clone(None)
        return oof_scores

    # ------------------------------------------------------------------ prediction

    def _check_fitted(self) -> None:
        if not getattr(self, "fitted_", False):
            raise RuntimeError("CalibratedModel is not fitted; call fit() first")

    def __sklearn_is_fitted__(self) -> bool:
        """Fitted state for sklearn >= 1.6 (model and calibrator both fitted)."""
        return bool(getattr(self, "fitted_", False))

    def _pipeline(self, scores: "Callable[[Any], np.ndarray]") -> np.ndarray:
        """Calibrated, offset output; ``scores(model)`` gives each calibrator's input."""
        if self.ensemble_:
            p = np.mean([cal.predict_proba(scores(model)) for model, cal in self.ensemble_], axis=0)
        else:
            p = self.calibrator_.predict_proba(scores(self.model_))
        for off in self.offsets_:
            p = off.transform(p)
        return p

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
        return self._pipeline(lambda model: _model_scores(model, X))

    # ------------------------------------------------------------------ offsets

    def offset_to(
        self,
        *,
        target_mean: float | None = None,
        delta: float | None = None,
        X: object = None,
        sample_weight: object = None,
    ) -> Self:
        """Append an inspectable :class:`LogitOffset` stage.

        Mode B (``target_mean``) anchors the portfolio mean of the current
        pipeline output — computed on ``X`` when given (weighted by
        ``sample_weight``), else on the stored calibration scores, weighted
        by the ``sample_weight`` given to :meth:`fit`. The offset is never
        folded into the calibrator's parameters.

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
            the stored calibration scores (and calibration weights) instead.
        sample_weight : array_like or None, keyword-only
            Positive weights for the portfolio mean on ``X``; only valid
            together with ``X``.

        Returns
        -------
        Self
            The wrapper, with the new offset appended to ``offsets_``.
        """
        self._check_fitted()
        if X is not None:
            p_now = self.predict_proba(X)
            weights = sample_weight
        else:
            if sample_weight is not None:
                raise ValueError(
                    "sample_weight is only accepted together with X; without X the "
                    "calibration weights given to fit() are used"
                )
            cal_scores = self._cal_scores
            if cal_scores is None:
                raise RuntimeError(
                    "calibration scores are not serialized; after from_dict, pass X so "
                    "the offset stage can record its audit means on current output"
                )
            p_now = self._pipeline(lambda _: cal_scores)
            weights = self._cal_weights
        off = LogitOffset(delta=delta, target_mean=target_mean)
        off.fit(p_now, sample_weight=weights)
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

    # ------------------------------------------------------------------ protocol
    # interval_inverse / point_inverse come from CalibratorProtocolMixin,
    # through the equivalent Chain (calibrator + offsets); the ensemble flow
    # (K distinct maps) has no single chain.

    def _fitted_calibrator(self) -> Chain:  # type: ignore[override]
        return self.chain_

    def _calibrators(self) -> list[BaseCalibrator]:
        """The fitted calibrator stage(s): one, or one per ensemble fold."""
        self._check_fitted()
        return [cal for _, cal in self.ensemble_] or [self.calibrator_]

    @property
    def is_monotone_(self) -> bool:
        """Monotone iff every calibrator stage is (offsets always are)."""
        return all(cal.is_monotone_ for cal in self._calibrators())

    @property
    def affine_logit_coeffs_(self) -> tuple[float, float] | None:
        """Composed ``(a, b + sum(deltas))`` when the calibrator stage is affine."""
        self._check_fitted()
        return None if self.ensemble_ else self.chain_.affine_logit_coeffs_

    def interpret(self) -> Interpretation:
        """Concatenated interpretation of the calibrator and every offset stage.

        Returns
        -------
        Interpretation
            Parameters and messages concatenated across the calibrator
            stage(s) and every appended offset, in application order.
        """
        stages: list[BaseCalibrator | LogitOffset] = [*self._calibrators(), *self.offsets_]
        parts = [stage.interpret() for stage in stages]
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

        Raises
        ------
        NotImplementedError
            For the ensemble flow.
        """
        self._check_fitted()
        if self.ensemble_:
            raise NotImplementedError(
                "the ensemble flow (K distinct maps) has no single equivalent chain; "
                "use ensemble=False for threshold translation"
            )
        return Chain([self.calibrator_, *self.offsets_])

    # ------------------------------------------------------------------ serialization

    def to_dict(self) -> dict[str, object]:
        """Versioned snapshot: nested calibrator, offsets, and a model *reference*.

        The base model is never serialized — only a reference (class name,
        the user-supplied ``model_id``, and ``get_params()`` when available
        and JSON-encodable); reattach it on load via
        ``CalibratedModel.from_dict(d, model=...)``. A wrapper loaded without
        a model writes back the reference it was loaded with, so its
        fingerprint is unchanged. The stored calibration
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
        if ref_model is None and self._model_ref is not None:
            # Loaded without a model: keep the reference it was saved with.
            model_ref = self._model_ref
        else:
            model_ref = self._model_ref_of(ref_model)
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
                "model_ref": model_ref,
            },
            fit_meta=encode_value(self.fit_meta_),
        )

    def _model_ref_of(self, ref_model: Any) -> dict[str, object]:
        """Class name, ``model_id`` and JSON-encodable ``get_params()`` of a model."""
        model_params: object = None
        if hasattr(ref_model, "get_params"):
            try:
                candidate = ref_model.get_params()
                json.dumps(candidate, allow_nan=False)
                model_params = candidate
            except (TypeError, ValueError):
                model_params = None
        return {
            "class_name": type(ref_model).__name__,
            "model_id": self.model_id,
            "params": model_params,
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
        check_payload(cls, d)
        state = d["state"]
        calibrator = load(state["calibrator"])
        obj = cls(model, calibrator, **d["params"])  # type: ignore[arg-type]
        obj.calibrator_ = calibrator  # type: ignore[assignment]
        obj.offsets_ = [LogitOffset.from_dict(o) for o in state["offsets"]]
        obj.ensemble_ = []
        obj.model_ = model
        obj._model_ref = dict(state["model_ref"])
        obj.fit_meta_ = dict(d["fit_meta"])
        obj.fitted_ = True
        return obj

    @classmethod
    def from_json(cls, path_or_str: object, model: Any = None) -> Self:  # type: ignore[override]
        """Load from a JSON string or a filesystem path (see :meth:`from_dict`)."""
        return super().from_json(path_or_str, model=model)
