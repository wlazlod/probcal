"""Chain: model-free composition of a calibrator with logit-offset stages.

The object a recourse engine inverts after a macro re-offset: recourse must
run through ``offset ∘ calibrator`` exactly, and every stage stays separately
inspectable. ``CalibratedModel.chain_`` builds the equivalent chain for users
who fitted through the wrapper and want to hand it on without the model.
"""

import json
import os
from collections.abc import Sequence

import numpy as np

from ._math import expit, logit
from ._registry import load, register
from ._results import Interpretation
from ._serialize import SCHEMA_VERSION, check_schema, fingerprint_of_dict
from .base import BaseCalibrator, _check_representable, _validate_point_targets
from .offset import LogitOffset


@register
class Chain:
    """A fitted calibrator followed by zero or more fitted ``LogitOffset`` stages.

    Exposes the full calibrator protocol — forward map, exact inverse maps,
    monotonicity, affine coefficients, interpretation, serialization — for
    the composed map ``sigma(logit(g(s)) + delta_1 + ... + delta_m)``.

    Parameters
    ----------
    stages : Sequence
        A fitted :class:`~probcal.base.BaseCalibrator` first, then zero or
        more fitted :class:`~probcal.offset.LogitOffset` stages, in
        application order.

    Attributes
    ----------
    calibrator_ : BaseCalibrator
        The first stage.
    offsets_ : tuple[LogitOffset, ...]
        The offset stages, in application order.
    is_monotone_ : bool
        True iff every stage is monotone (offsets always are).
    """

    def __init__(self, stages: "Sequence[object]") -> None:
        stages = list(stages)
        if not stages:
            raise ValueError("Chain needs at least a calibrator stage")
        head, tail = stages[0], stages[1:]
        if not isinstance(head, BaseCalibrator):
            raise ValueError(
                f"the first stage must be a fitted calibrator, got {type(head).__name__}"
            )
        for off in tail:
            if not isinstance(off, LogitOffset):
                raise ValueError(
                    "every stage after the first must be a LogitOffset, got "
                    f"{type(off).__name__}"
                )
        for stage in stages:
            if not getattr(stage, "fitted_", False):
                raise RuntimeError(
                    f"Chain stages must already be fitted; {type(stage).__name__} is not"
                )
        self.calibrator_: BaseCalibrator = head
        self.offsets_: tuple[LogitOffset, ...] = tuple(tail)  # type: ignore[arg-type]
        self.fitted_ = True

    # ------------------------------------------------------------------ forward

    @property
    def delta_(self) -> float:
        """Total log-odds shift of the offset stages."""
        return float(sum(off.delta_ for off in self.offsets_))

    @property
    def is_monotone_(self) -> bool:
        """True iff the calibrator stage is monotone (offsets always are)."""
        return bool(self.calibrator_.is_monotone_)

    def predict_proba(self, s: object) -> np.ndarray:
        """The composed calibrated probability, applied stage by stage."""
        p = self.calibrator_.predict_proba(s)
        for off in self.offsets_:
            p = off.transform(p)
        return p

    def __sklearn_is_fitted__(self) -> bool:
        """Fitted state for sklearn >= 1.6; a chain is built from fitted stages."""
        return bool(self.fitted_)

    # ------------------------------------------------------------------ protocol

    @property
    def affine_logit_coeffs_(self) -> tuple[float, float] | None:
        """``(a, b + sum(delta))`` when the calibrator is affine on the logit scale."""
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
        return float(expit(np.array([logit(np.array([value]))[0] - self.delta_]))[0])

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
        from .base import UnattainableTargetError

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
        arr = _validate_point_targets(p)
        shifted_z = logit(arr) - self.delta_
        _check_representable(shifted_z, "probability")  # the intermediate must round-trip
        z = self.calibrator_.point_inverse(expit(shifted_z), space="logit")
        _check_representable(np.asarray(z), space)
        return np.asarray(z) if space == "logit" else expit(np.asarray(z))

    def interpret(self) -> Interpretation:
        """Concatenated interpretation of every stage."""
        parts = [self.calibrator_.interpret()] + [off.interpret() for off in self.offsets_]
        names: tuple[str, ...] = ()
        values: tuple[float, ...] = ()
        messages: tuple[str, ...] = ()
        for part in parts:
            names += tuple(f"{part.method}.{n}" for n in part.param_names)
            values += part.param_values
            messages += part.messages
        return Interpretation(
            method=f"Chain[{', '.join(p.method for p in parts)}]",
            param_names=names,
            param_values=values,
            messages=messages,
        )

    # ------------------------------------------------------------------ serialization

    def to_dict(self) -> dict[str, object]:
        """Versioned snapshot: the stages' own envelopes, in order."""
        from . import __version__

        return {
            "probcal_schema": SCHEMA_VERSION,
            "probcal_version": __version__,
            "class": type(self).__name__,
            "params": {},
            "state": {
                "stages": [self.calibrator_.to_dict()] + [off.to_dict() for off in self.offsets_],
            },
            "fit_meta": {},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chain":
        """Rebuild the chain by loading every stage through the registry."""
        check_schema(d)
        if d.get("class") != cls.__name__:
            raise ValueError(f"payload was written by {d.get('class')!r}, not {cls.__name__}")
        stages = [load(sd) for sd in d["state"]["stages"]]
        return cls(stages)

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
    def from_json(cls, path_or_str: object) -> "Chain":
        """Load from a JSON string or a filesystem path."""
        text = str(path_or_str)
        if not text.lstrip().startswith("{"):
            with open(text, encoding="utf-8") as fh:
                text = fh.read()
        return cls.from_dict(json.loads(text))

    def fingerprint(self) -> str:
        """SHA-256 of the canonical serialized form (stages included)."""
        return fingerprint_of_dict(self.to_dict())
