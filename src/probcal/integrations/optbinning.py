"""Calibrated optbinning scorecards.

Requires the ``probcal[optbinning]`` extra (optbinning >= 0.21). The
scorecard stays the deployed artifact — points are untouched; probcal adds a
calibrated PD layer on top plus exact translation between calibrated PD
bands and point cut-offs, possible because ``Scorecard.score`` is affine in
the fitted logistic regression's log-odds unless ``rounding=True``.
"""

import hashlib
import json
import os
import warnings

import numpy as np

try:
    import optbinning as _optbinning  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "probcal.integrations.optbinning requires optbinning >= 0.21; install the "
        "extra: pip install 'probcal[optbinning]'"
    ) from exc

from .._math import logit
from .._serialize import SCHEMA_VERSION, fingerprint_of_dict
from ..base import BaseCalibrator
from ..parametric import BetaCalibrator
from ..thresholds import calibrated_bands_to_raw

_AFFINE_ATOL = 1e-6  # measured residual is ~1e-13; rounding=True measures ~1.4


class CalibratedScorecard:
    """An optbinning ``Scorecard`` with a probcal calibration layer on top.

    Built by :func:`calibrate_scorecard`. Points (``score``) are unchanged —
    the scorecard remains the deployed artifact; ``predict_proba`` returns
    the calibrated PD, and the calibrator protocol (``interval_inverse``,
    ``point_inverse``, ...) operates on the scorecard's model-probability
    scale, so calibrated policies translate to raw probabilities and — via
    ``points_affine_coeffs_`` — exactly to the points scale.

    Attributes
    ----------
    scorecard_ : optbinning.Scorecard
        The wrapped, fitted scorecard.
    calibrator_ : BaseCalibrator
        The fitted probcal calibrator over the scorecard's probabilities.
    points_affine_coeffs_ : tuple(A, B) or None
        ``score = A + B * logit(p_model)``, recovered from the calibration
        data and verified to machine precision; ``None`` (with a warning at
        build time) when the relation is not affine (``rounding=True``).
    """

    def __init__(
        self,
        scorecard: object,
        calibrator: BaseCalibrator,
        points_affine_coeffs: tuple[float, float] | None,
    ) -> None:
        self.scorecard_ = scorecard
        self.calibrator_ = calibrator
        self.points_affine_coeffs_ = points_affine_coeffs

    # ------------------------------------------------------------------ forward

    def _model_proba(self, X: object) -> np.ndarray:
        return np.asarray(self.scorecard_.predict_proba(X))[:, 1]  # type: ignore[attr-defined]

    def predict_proba(self, X: object) -> np.ndarray:
        """Calibrated PD for scorecard inputs (1-D, probcal convention)."""
        return self.calibrator_.predict_proba(self._model_proba(X))

    def score(self, X: object) -> np.ndarray:
        """Unchanged scorecard points — the deployed artifact is untouched."""
        return np.asarray(self.scorecard_.score(X))  # type: ignore[attr-defined]

    def interpret(self):  # noqa: ANN201 - probcal Interpretation
        """The calibration layer's plain-language reading."""
        return self.calibrator_.interpret()

    # ------------------------------------------------------------------ protocol

    @property
    def is_monotone_(self) -> bool:
        """Whether the calibration layer preserves the scorecard's ranking."""
        return bool(self.calibrator_.is_monotone_)

    @property
    def affine_logit_coeffs_(self) -> tuple[float, float] | None:
        """The calibration layer's affine-logit coefficients, if any."""
        return self.calibrator_.affine_logit_coeffs_

    def interval_inverse(
        self, lo: float, hi: float, *, space: str = "probability", buffer_logit: float = 0.0
    ) -> tuple[float, float]:
        """Preimage of a calibrated PD interval on the model-probability scale."""
        return self.calibrator_.interval_inverse(lo, hi, space=space, buffer_logit=buffer_logit)

    def point_inverse(self, p: object, *, space: str = "probability") -> np.ndarray:
        """Exact preimage of calibrated PDs on the model-probability scale."""
        return self.calibrator_.point_inverse(p, space=space)

    def masterscale(self, bands: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
        """Calibrated PD bands -> scorecard point cut-offs, exactly.

        Composes :func:`probcal.thresholds.calibrated_bands_to_raw` (bands on
        the calibrated scale to raw log-odds intervals) with the verified
        affine points map. Point intervals are returned as ``(lo, hi)`` with
        ``lo <= hi`` (the affine slope is negative for the usual
        higher-points-safer scaling), and the cut-offs are checked for
        monotone ordering across bands.

        Parameters
        ----------
        bands : dict[str, tuple(lo, hi)]
            Calibrated PD bands, e.g. ``{"A": (0.0, 0.01), "B": (0.01, 0.05)}``.

        Returns
        -------
        dict[str, tuple(points_lo, points_hi)]
            Point cut-offs per band.

        Raises
        ------
        RuntimeError
            If the scorecard is not affine in log-odds (``rounding=True``) —
            use :meth:`interval_inverse` on the raw probability instead —
            or if the resulting cut-offs are not monotone across bands.
        """
        if self.points_affine_coeffs_ is None:
            raise RuntimeError(
                "this scorecard is not affine in log-odds (rounding=True); the exact "
                "masterscale is unavailable — use interval_inverse on the raw "
                "probability instead"
            )
        a_pts, b_pts = self.points_affine_coeffs_
        raw = calibrated_bands_to_raw(self.calibrator_, bands, space="logit")
        out: dict[str, tuple[float, float]] = {}
        for name, (z_lo, z_hi) in raw.items():
            p_lo = a_pts + b_pts * z_lo if np.isfinite(z_lo) else np.inf * -np.sign(b_pts)
            p_hi = a_pts + b_pts * z_hi if np.isfinite(z_hi) else np.inf * np.sign(b_pts)
            out[name] = (min(p_lo, p_hi), max(p_lo, p_hi))
        # Bands ordered by rising calibrated PD must map to monotone point
        # ranges (falling when B < 0, i.e. higher points = safer).
        order = sorted(out, key=lambda k: bands[k][0])
        cuts = [out[k] for k in order]
        if b_pts < 0:
            mono = all(cuts[i][1] >= cuts[i + 1][1] - 1e-9 for i in range(len(cuts) - 1))
        else:
            mono = all(cuts[i][0] <= cuts[i + 1][0] + 1e-9 for i in range(len(cuts) - 1))
        if not mono:
            raise RuntimeError("masterscale cut-offs are not monotone across bands")
        return out

    # ------------------------------------------------------------------ provenance

    def to_dict(self) -> dict[str, object]:
        """Calibrator envelope plus a fingerprint of the scorecard table.

        The scorecard object itself is not serialized (it is optbinning's
        artifact); rebuild with ``CalibratedScorecard.from_dict(d,
        scorecard=...)`` after loading the scorecard through optbinning's
        own ``save``/``load``.
        """
        from .. import __version__

        return {
            "probcal_schema": SCHEMA_VERSION,
            "probcal_version": __version__,
            "class": type(self).__name__,
            "params": {},
            "state": {
                "calibrator": self.calibrator_.to_dict(),
                "points_affine_coeffs": (
                    list(self.points_affine_coeffs_)
                    if self.points_affine_coeffs_ is not None
                    else None
                ),
                "scorecard_fingerprint": self.scorecard_fingerprint(),
            },
            "fit_meta": {},
        }

    @classmethod
    def from_dict(cls, d: dict, scorecard: object) -> "CalibratedScorecard":
        """Rebuild around a scorecard loaded through optbinning's own tooling.

        Raises
        ------
        ValueError
            If the payload class differs, or the supplied scorecard's table
            fingerprint does not match the stored one.
        """
        from .._registry import load
        from .._serialize import check_schema

        check_schema(d)
        if d.get("class") != cls.__name__:
            raise ValueError(f"payload was written by {d.get('class')!r}, not {cls.__name__}")
        state = d["state"]
        coeffs = state.get("points_affine_coeffs")
        obj = cls(
            scorecard,
            load(state["calibrator"]),  # type: ignore[arg-type]
            tuple(coeffs) if coeffs is not None else None,  # type: ignore[arg-type]
        )
        stored = state.get("scorecard_fingerprint")
        if stored is not None and obj.scorecard_fingerprint() != stored:
            raise ValueError(
                "the supplied scorecard's table fingerprint does not match the stored "
                "one — this calibration layer was fitted against a different scorecard"
            )
        return obj

    def scorecard_fingerprint(self) -> str:
        """SHA-256 of the scorecard table (CSV form) — names the deployed artifact."""
        table = self.scorecard_.table(style="detailed")  # type: ignore[attr-defined]
        return hashlib.sha256(table.to_csv(index=False).encode("utf-8")).hexdigest()

    def fingerprint(self) -> str:
        """SHA-256 over the calibration layer and the scorecard-table fingerprint."""
        return fingerprint_of_dict(self.to_dict())

    def to_json(
        self, path: "str | os.PathLike[str] | None" = None, *, indent: int = 2
    ) -> str | None:
        """Serialize the calibration layer (see :meth:`to_dict`)."""
        text = json.dumps(self.to_dict(), indent=indent)
        if path is None:
            return text
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return None


def calibrate_scorecard(
    scorecard: object,
    X_cal: object,
    y_cal: object,
    *,
    calibrator: BaseCalibrator | None = None,
    sample_weight: object = None,
) -> CalibratedScorecard:
    """Fit a probcal calibration layer on a fitted optbinning scorecard.

    Parameters
    ----------
    scorecard : optbinning.Scorecard
        Fitted scorecard with ``predict_proba``, ``score``, and ``table``.
    X_cal, y_cal : array_like
        Held-out calibration data (never the scorecard's training data —
        see the data-splitting chapter).
    calibrator : BaseCalibrator or None, keyword-only
        Unfitted probcal prototype; ``None`` uses ``BetaCalibrator()``.
    sample_weight : array_like or None, keyword-only
        Positive observation weights for the calibration fit.

    Returns
    -------
    CalibratedScorecard
        Calibrated PD layer over the unchanged scorecard, with the affine
        points map recovered and verified (or refused with a warning when
        ``rounding=True`` breaks affinity).
    """
    proto = calibrator if calibrator is not None else BetaCalibrator()
    cal = type(proto)(**proto.get_params())
    p_model = np.asarray(scorecard.predict_proba(X_cal))[:, 1]  # type: ignore[attr-defined]
    cal.fit(p_model, y_cal, sample_weight=sample_weight)

    points = np.asarray(scorecard.score(X_cal), dtype=np.float64)  # type: ignore[attr-defined]
    z = logit(p_model)
    b_pts, a_pts = np.polyfit(z, points, 1)
    resid = float(np.max(np.abs(points - (a_pts + b_pts * z))))
    coeffs: tuple[float, float] | None = (float(a_pts), float(b_pts))
    if resid > _AFFINE_ATOL:
        warnings.warn(
            f"scorecard points are not affine in log-odds (max residual {resid:.3g}, "
            "e.g. rounding=True); masterscale is unavailable — falling back to "
            "interval_inverse on the raw probability",
            UserWarning,
            stacklevel=2,
        )
        coeffs = None
    return CalibratedScorecard(scorecard, cal, coeffs)
