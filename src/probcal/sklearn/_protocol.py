"""Delegation of the probcal calibrator protocol to a fitted inner calibrator.

sklearn-free on purpose: shared by the sklearn adapters and by
``probcal.integrations.optbinning.CalibratedScorecard``.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .._results import Interpretation
    from ..base import BaseCalibrator


class CalibratorProtocolMixin:
    """``is_monotone_``, ``affine_logit_coeffs_``, the inverse maps, and
    ``interpret`` — each read from the fitted calibrator returned by
    :meth:`_fitted_calibrator`, so a wrapper can be handed to any consumer of
    the calibrator protocol (e.g. treecf's ``Target.calibrated``).
    """

    def _fitted_calibrator(self) -> "BaseCalibrator":
        """The fitted calibrator; overridden where fittedness must be checked."""
        return self.calibrator_  # type: ignore[attr-defined,no-any-return]

    @property
    def is_monotone_(self) -> bool:
        """Whether the fitted calibration map is non-decreasing (delegated)."""
        return bool(self._fitted_calibrator().is_monotone_)

    @property
    def affine_logit_coeffs_(self) -> tuple[float, float] | None:
        """Affine-logit coefficients of the calibration map, if any (delegated)."""
        return self._fitted_calibrator().affine_logit_coeffs_

    def interval_inverse(
        self, lo: float, hi: float, *, space: str = "probability", buffer_logit: float = 0.0
    ) -> tuple[float, float]:
        """Preimage of a calibrated interval on the calibrator's input scale (delegated)."""
        return self._fitted_calibrator().interval_inverse(
            lo, hi, space=space, buffer_logit=buffer_logit
        )

    def point_inverse(self, p: object, *, space: str = "probability") -> np.ndarray:
        """Exact preimage of calibrated probabilities (delegated)."""
        return self._fitted_calibrator().point_inverse(p, space=space)

    def interpret(self) -> "Interpretation":
        """The fitted calibrator's plain-language reading (delegated)."""
        return self._fitted_calibrator().interpret()
