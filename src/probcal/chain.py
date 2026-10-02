"""Chain: model-free composition of a calibrator with logit-offset stages.

The object a recourse engine inverts after a macro re-offset: recourse must
run through ``offset ∘ calibrator`` exactly, and every stage stays separately
inspectable. ``CalibratedModel.chain_`` builds the equivalent chain for users
who fitted through the wrapper and want to hand it on without the model.
"""

from collections.abc import Sequence

import numpy as np

from ._math import expit, expit1, logit, logit1
from ._registry import load, register
from ._results import Interpretation
from ._serialize import JsonIO, check_payload, envelope
from ._validation import validate_space
from .base import (
    BaseCalibrator,
    _check_representable,
    _validate_point_targets,
    shrink_interval,
    validate_interval,
)
from .offset import LogitOffset


def _concat_interpretations(
    method: str, parts: "list[Interpretation]", *, qualify: bool
) -> Interpretation:
    """One :class:`Interpretation` from several stages, in application order.

    ``qualify=True`` prefixes each parameter name with its stage's method.
    """
    names: tuple[str, ...] = ()
    values: tuple[float, ...] = ()
    messages: tuple[str, ...] = ()
    for part in parts:
        names += (
            tuple(f"{part.method}.{n}" for n in part.param_names) if qualify else part.param_names
        )
        values += part.param_values
        messages += part.messages
    return Interpretation(
        method=f"{method}[{', '.join(p.method for p in parts)}]",
        param_names=names,
        param_values=values,
        messages=messages,
    )


@register
class Chain(JsonIO):
    """A calibrator followed by zero or more ``LogitOffset`` stages.

    Exposes the full calibrator protocol — forward map, exact inverse maps,
    monotonicity, affine coefficients, interpretation, serialization — for
    the composed map ``sigma(logit(g(s)) + delta_1 + ... + delta_m)``.

    Stages may be given fitted (the chain is then immediately usable) or
    unfitted (the chain must be fitted with :meth:`fit` before any reading
    method is called). ``fit`` always refits every stage in place,
    sequentially: the calibrator on ``(s, y, sample_weight)``, then each
    offset in turn on the running calibrated probabilities.

    Parameters
    ----------
    stages : Sequence
        A :class:`~probcal.base.BaseCalibrator` first, then zero or more
        :class:`~probcal.offset.LogitOffset` stages, in application order.
        Stored verbatim (as ``self.stages``) when a list is given.

    Attributes
    ----------
    stages : list[object]
        The stages, in application order, stored verbatim.
    calibrator_ : BaseCalibrator
        Read-only view of ``stages[0]``.
    offsets_ : tuple[LogitOffset, ...]
        Read-only view of ``stages[1:]``.
    fitted_ : bool
        ``True`` iff every stage is fitted.
    is_monotone_ : bool
        True iff every stage is monotone (offsets always are).
    """

    def __init__(self, stages: "Sequence[object]") -> None:
        listed = stages if isinstance(stages, list) else list(stages)
        if not listed:
            raise ValueError("Chain needs at least a calibrator stage")
        head, tail = listed[0], listed[1:]
        if not isinstance(head, BaseCalibrator):
            raise ValueError(f"the first stage must be a calibrator, got {type(head).__name__}")
        for off in tail:
            if not isinstance(off, LogitOffset):
                raise ValueError(
                    f"every stage after the first must be a LogitOffset, got {type(off).__name__}"
                )
        # Stored verbatim when a list is given: sklearn's clone() constructs
        # with the params it just cloned and then checks identity via get_params.
        self.stages: list[object] = listed
        self.fitted_: bool = all(getattr(st, "fitted_", False) for st in listed)

    @property
    def calibrator_(self) -> BaseCalibrator:
        """Read-only view of the first stage."""
        return self.stages[0]  # type: ignore[return-value]

    @property
    def offsets_(self) -> "tuple[LogitOffset, ...]":
        """Read-only view of the offset stages, in application order."""
        return tuple(self.stages[1:])  # type: ignore[arg-type]

    def _check_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("Chain is not fitted; call fit() first")

    def fit(self, s: object, y: object, sample_weight: object = None) -> "Chain":
        """Fit every stage sequentially on the same calibration data.

        The head calibrator is fitted on ``(s, y, sample_weight)``; each
        offset is then fitted on the running calibrated probabilities, so
        the offset anchors the calibrator's in-sample output — exactly what
        ``CalibratedModel.offset_to`` does. There is no cross-fitting inside
        a chain and no automatic MLE offset (``estimate_offset`` remains an
        explicit choice). ``fit`` always refits every stage, including
        stages that were already fitted at construction; to keep a stage
        frozen, compose fitted objects and skip ``fit``, as before.
        """
        self.calibrator_.fit(s, y, sample_weight=sample_weight)
        p = self.calibrator_.predict_proba(s)
        for off in self.offsets_:
            off.fit(p, y=y, sample_weight=sample_weight)
            p = off.transform(p)
        self.fitted_ = True
        return self

    def get_params(self, deep: bool = True) -> dict[str, object]:
        """Constructor parameters, with ``stages__i[__param]`` nesting when ``deep``."""
        params: dict[str, object] = {"stages": self.stages}
        if deep:
            for i, stage in enumerate(self.stages):
                params[f"stages__{i}"] = stage
                for key, value in stage.get_params(deep=True).items():  # type: ignore[attr-defined]
                    params[f"stages__{i}__{key}"] = value
        return params

    def set_params(self, **params: object) -> "Chain":
        """Set ``stages`` wholesale, one stage (``stages__i``), or a nested stage param.

        Stage replacements (``stages__i``) validate the whole candidate
        list before anything on the chain changes, so a rejected
        replacement leaves the chain as it was; a wholesale ``stages=``
        key is applied first and independently of the indexed keys in the
        same call. Like the stages' own ``set_params``, setting a nested
        parameter (``stages__i__param``) does not clear ``fitted_``; call
        ``fit`` again for the new value to take effect.
        """
        if "stages" in params:
            self.__init__(params.pop("stages"))  # type: ignore[misc, arg-type]
        nested: dict[int, dict[str, object]] = {}
        replacements: dict[int, object] = {}
        for key, value in params.items():
            prefix, _, rest = key.partition("__")
            idx_text, _, sub = rest.partition("__")
            if prefix != "stages" or not idx_text.isdigit():
                raise ValueError(f"invalid parameter {key!r} for Chain")
            idx = int(idx_text)
            if idx >= len(self.stages):
                raise ValueError(
                    f"invalid parameter {key!r} for Chain: stage index {idx} is out "
                    f"of range for {len(self.stages)} stages"
                )
            if not sub:
                replacements[idx] = value
            else:
                nested.setdefault(idx, {})[sub] = value
        if replacements:
            candidate = list(self.stages)
            for idx, stage in replacements.items():
                candidate[idx] = stage
            self.__init__(candidate)  # type: ignore[misc] # validates before mutating
        for idx, sub_params in nested.items():
            self.stages[idx].set_params(**sub_params)  # type: ignore[attr-defined]
        return self

    # ------------------------------------------------------------------ forward

    @property
    def delta_(self) -> float:
        """Total log-odds shift of the offset stages."""
        self._check_fitted()
        return float(sum(off.delta_ for off in self.offsets_))

    @property
    def is_monotone_(self) -> bool:
        """True iff the calibrator stage is monotone (offsets always are)."""
        self._check_fitted()
        return bool(self.calibrator_.is_monotone_)

    def predict_proba(self, s: object) -> np.ndarray:
        """The composed calibrated probability, applied stage by stage."""
        self._check_fitted()
        p = self.calibrator_.predict_proba(s)
        for off in self.offsets_:
            p = off.transform(p)
        return p

    def __sklearn_is_fitted__(self) -> bool:
        """Fitted state for sklearn >= 1.6 (``True`` iff every stage is fitted)."""
        return bool(self.fitted_)

    # ------------------------------------------------------------------ protocol

    @property
    def affine_logit_coeffs_(self) -> tuple[float, float] | None:
        """``(a, b + sum(delta))`` when the calibrator is affine on the logit scale."""
        self._check_fitted()
        coeffs = self.calibrator_.affine_logit_coeffs_
        if coeffs is None:
            return None
        a, b = coeffs
        return (a, b + self.delta_)

    def _shift_bound(self, value: float, *, is_lower: bool) -> float:
        """Move one calibrated bound back through the offsets (0/1 are fixed points)."""
        if is_lower and value <= 0.0:
            return 0.0
        if not is_lower and value >= 1.0:
            return 1.0
        return expit1(logit1(value) - self.delta_)

    def interval_inverse(
        self,
        lo: float,
        hi: float,
        *,
        space: str = "probability",
        buffer_logit: float = 0.0,
    ) -> tuple[float, float]:
        """Preimage of a calibrated interval through every stage.

        The buffer applies to the *final* calibrated scale, then the bounds
        travel back through the offsets on the logit scale, then the
        calibrator's own generalized inverse finishes the job — every
        refusal (empty buffered interval, unattainable target) is raised by
        the same doctrine as the underlying stages.

        Parameters
        ----------
        lo, hi : float
            Calibrated-probability bounds on the chain's output scale.
        space : {"probability", "logit"}
            Scale of the returned raw bounds.
        buffer_logit : float
            Logit-space shrinkage applied before inverting.

        Returns
        -------
        tuple of float
            ``(raw_lo, raw_hi)`` on the requested scale.

        Raises
        ------
        UnattainableTargetError
            If the buffered interval is empty or does not intersect the
            chain's output range.
        """
        self._check_fitted()
        validate_interval(lo, hi)
        validate_space(space)
        lo_b, hi_b = shrink_interval(lo, hi, buffer_logit)
        lo_c = self._shift_bound(lo_b, is_lower=True)
        hi_c = self._shift_bound(hi_b, is_lower=False)
        return self.calibrator_.interval_inverse(lo_c, hi_c, space=space, buffer_logit=0.0)

    def point_inverse(self, p: object, *, space: str = "probability") -> np.ndarray:
        """Exact preimage of composed calibrated probabilities.

        Shifts the targets back through the offsets on the logit scale, then
        the calibrator's own exact point inverse finishes; the boundary
        doctrine (strict ``(0, 1)`` targets, representable probability-space
        results) is inherited from the stages.

        Raises
        ------
        UnattainableTargetError
            If a target lies outside ``(0, 1)``, is unattainable for the
            calibrator, or the probability-space result is not
            representable.
        """
        self._check_fitted()
        validate_space(space)
        arr = _validate_point_targets(p)
        shifted_z = logit(arr) - self.delta_
        _check_representable(shifted_z, "probability")  # the intermediate must round-trip
        z = self.calibrator_.point_inverse(expit(shifted_z), space="logit")
        _check_representable(np.asarray(z), space)
        return np.asarray(z) if space == "logit" else expit(np.asarray(z))

    def interpret(self) -> Interpretation:
        """Concatenated interpretation of every stage."""
        self._check_fitted()
        parts = [self.calibrator_.interpret()] + [off.interpret() for off in self.offsets_]
        return _concat_interpretations("Chain", parts, qualify=True)

    # ------------------------------------------------------------------ serialization

    def to_dict(self) -> dict[str, object]:
        """Versioned snapshot: the stages' own envelopes, in order."""
        self._check_fitted()
        return envelope(
            self,
            params={},
            state={
                "stages": [self.calibrator_.to_dict()] + [off.to_dict() for off in self.offsets_],
            },
            fit_meta={},
        )

    @classmethod
    def from_dict(cls, d: dict) -> "Chain":
        """Rebuild the chain by loading every stage through the registry."""
        check_payload(cls, d)
        stages = [load(sd) for sd in d["state"]["stages"]]
        return cls(stages)
