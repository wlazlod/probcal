"""Parametric calibrators: Platt, temperature, and beta calibration.

Theory, derivations, and parameter interpretation: ``docs/concepts/methods-parametric.md``.

References
----------
Platt (1999); Lin, Lin & Weng (2007); Guo et al. (2017); Kull, Silva Filho &
Flach (2017, AISTATS and EJS) — full records in the documentation.
"""

import warnings

import numpy as np

from ._math import bisect, expit, irls_logistic, logit
from ._results import Interpretation
from .base import BaseCalibrator

_U_LO = 1e-6
_U_HI = 1e6


class PlattCalibrator(BaseCalibrator):
    """Logistic recalibration on the logit scale (Platt scaling).

    Fits ``logit g(s) = a * logit(s) + b`` by IRLS with Lin–Lin–Weng smoothed
    targets ``(N+ + 1)/(N+ + 2)`` and ``1/(N- + 2)`` for stability on small
    samples. The identity map is ``(a, b) = (1, 0)``.

    Attributes
    ----------
    a_ : float
        Fitted slope — spread correction: ``a < 1`` shrinks overconfident
        scores toward the base rate, ``a > 1`` sharpens underconfident ones.
    b_ : float
        Fitted intercept — calibration-in-the-large shift in log-odds.

    References
    ----------
    Platt (1999); Lin, Lin & Weng (2007). The logistic family fitted on raw
    SVM outputs (Platt's original setting) does not contain the identity; on
    logits it does — see the parametric-methods chapter.
    """

    def _fit(self, s: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        z = logit(s)
        n_pos = float(np.sum(y == 1.0))
        n_neg = float(np.sum(y == 0.0))
        targets = np.where(y == 1.0, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))
        X = np.column_stack([np.ones_like(z), z])
        res = irls_logistic(X, targets, w=w)
        self.b_ = float(res.beta[0])
        self.a_ = float(res.beta[1])

    def _predict(self, s: np.ndarray) -> np.ndarray:
        return expit(self.a_ * logit(s) + self.b_)

    @property
    def affine_logit_coeffs_(self) -> tuple[float, float] | None:
        """``(a, b)``: Platt scaling is affine on the logit scale."""
        self._check_fitted()
        return (self.a_, self.b_)

    def interpret(self) -> Interpretation:
        """Read the fitted slope and intercept against the identity ``(1, 0)``."""
        self._check_fitted()
        if self.a_ < 1.0:
            slope_msg = (
                f"slope a = {self.a_:.3f} < 1: scores were overconfident (too spread out); "
                "predictions are shrunk toward the base rate"
            )
        else:
            slope_msg = (
                f"slope a = {self.a_:.3f} >= 1: scores were underconfident (too flat); "
                "predictions are sharpened"
            )
        int_msg = (
            f"intercept b = {self.b_:.3f}: base-rate (calibration-in-the-large) shift of "
            f"{self.b_:+.3f} log-odds, odds factor {np.exp(self.b_):.3f}"
        )
        return Interpretation(
            method=type(self).__name__,
            param_names=("a", "b"),
            param_values=(self.a_, self.b_),
            messages=(slope_msg, int_msg, "identity map corresponds to (a, b) = (1, 0)"),
        )


class TemperatureCalibrator(BaseCalibrator):
    """Temperature scaling: ``g(s) = sigma(logit(s) / T)``.

    ``T`` minimizes the calibration-set negative log-likelihood via a
    safeguarded 1-D Newton iteration (bisection fallback) on ``u = 1/T``.

    Attributes
    ----------
    T_ : float
        Fitted temperature. ``T > 1``: the model was overconfident and is
        softened; ``T < 1``: underconfident and sharpened. Temperature cannot
        fix base-rate error — ``s = 0.5`` is a fixed point for every ``T``;
        use Platt scaling or ``LogitOffset`` for level shifts.

    References
    ----------
    Guo, Pleiss, Sun & Weinberger (2017).
    """

    def _fit(self, s: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        z = logit(s)

        def score(u: float) -> float:
            return float(np.sum(w * z * (expit(u * z) - y)))

        f_lo, f_hi = score(_U_LO), score(_U_HI)
        if f_lo * f_hi > 0.0:
            u = _U_LO if abs(f_lo) <= abs(f_hi) else _U_HI
            warnings.warn(
                "TemperatureCalibrator: NLL has no interior minimum in "
                f"1/T ∈ [{_U_LO:g}, {_U_HI:g}]; clamping to the boundary",
                UserWarning,
                stacklevel=2,
            )
        else:
            u = bisect(score, _U_LO, _U_HI, tol=1e-12)
        self.T_ = float(1.0 / u)

    def _predict(self, s: np.ndarray) -> np.ndarray:
        return expit(logit(s) / self.T_)

    @property
    def affine_logit_coeffs_(self) -> tuple[float, float] | None:
        """``(1/T, 0)``: temperature scaling is affine on the logit scale."""
        self._check_fitted()
        return (1.0 / self.T_, 0.0)

    def interpret(self) -> Interpretation:
        """Read the fitted temperature against the identity ``T = 1``."""
        self._check_fitted()
        if self.T_ > 1.0:
            msg = (
                f"T = {self.T_:.3f} > 1: the model was overconfident; logits are divided "
                "by T (softening toward 1/2)"
            )
        else:
            msg = (
                f"T = {self.T_:.3f} <= 1: the model was underconfident; logits are divided "
                "by T (sharpening away from 1/2)"
            )
        return Interpretation(
            method=type(self).__name__,
            param_names=("T",),
            param_values=(self.T_,),
            messages=(
                msg,
                "temperature cannot fix base-rate error: s = 0.5 maps to 0.5 for every T "
                "(use PlattCalibrator or LogitOffset for level shifts)",
            ),
        )


class BetaCalibrator(BaseCalibrator):
    """Beta calibration: ``logit g(s) = a·ln s − b·ln(1 − s) + c``.

    Variants (spec §6; DECISIONS entry 27): ``"abm"`` fits ``(a, b, c)``;
    ``"ab"`` ties ``a = b`` (equivalent to Platt scaling on logits); ``"a"``
    additionally fixes ``c = 0`` (a single-parameter map, the temperature
    family in a different parameterization). The monotonicity constraint
    ``a, b >= 0`` is enforced by the betacal refit strategy: a negative
    exponent drops its feature and refits.

    Attributes
    ----------
    a_ : float
        Sensitivity near ``s -> 0`` — governs the low-probability tail
        (critical for low-PD credit portfolios).
    b_ : float
        Sensitivity near ``s -> 1``.
    c_ : float
        Base-rate shift in log-odds.
    constraint_active_ : bool
        Whether the ``a, b >= 0`` constraint forced a refit.

    References
    ----------
    Kull, Silva Filho & Flach (2017), AISTATS 54 and EJS 11(2). The identity
    is ``(a, b, c) = (1, 1, 0)``: beta calibration cannot un-calibrate an
    already calibrated model. ``a != b`` captures asymmetric tail distortion;
    temperature is the special case ``a = b = 1/T, c = 0``.
    """

    def __init__(self, variant: str = "abm") -> None:
        self.variant = variant

    def _fit(self, s: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        if self.variant not in ("abm", "ab", "a"):
            raise ValueError(f"variant must be 'abm', 'ab', or 'a', got {self.variant!r}")
        ln_s = np.log(s)
        ln_1ms = -np.log1p(-s)  # -ln(1 - s), non-negative
        self.constraint_active_ = False

        if self.variant == "abm":
            self._fit_abm(ln_s, ln_1ms, y, w)
        elif self.variant == "ab":
            z = ln_s + ln_1ms  # logit(s)
            X = np.column_stack([np.ones_like(z), z])
            beta = irls_logistic(X, y, w=w).beta
            a = float(beta[1])
            if a < 0.0:
                self.constraint_active_ = True
                a = 0.0
                beta0 = self._intercept_only(y, w)
                self.c_ = beta0
            else:
                self.c_ = float(beta[0])
            self.a_ = self.b_ = a
        else:  # "a": a = b, c = 0
            z = ln_s + ln_1ms

            def score(u: float) -> float:
                return float(np.sum(w * z * (expit(u * z) - y)))

            f_lo, f_hi = score(0.0), score(_U_HI)
            if f_lo * f_hi > 0.0:
                a = 0.0 if abs(f_lo) <= abs(f_hi) else _U_HI
                if a == 0.0:
                    self.constraint_active_ = True
            else:
                a = bisect(score, 0.0, _U_HI, tol=1e-12)
            self.a_ = self.b_ = float(a)
            self.c_ = 0.0

    def _fit_abm(self, ln_s: np.ndarray, ln_1ms: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        ones = np.ones_like(ln_s)
        beta = irls_logistic(np.column_stack([ones, ln_s, ln_1ms]), y, w=w).beta
        a, b, c = float(beta[1]), float(beta[2]), float(beta[0])
        if a < 0.0:
            self.constraint_active_ = True
            beta = irls_logistic(np.column_stack([ones, ln_1ms]), y, w=w).beta
            a, b, c = 0.0, float(beta[1]), float(beta[0])
            if b < 0.0:
                b, c = 0.0, self._intercept_only(y, w)
        elif b < 0.0:
            self.constraint_active_ = True
            beta = irls_logistic(np.column_stack([ones, ln_s]), y, w=w).beta
            a, b, c = float(beta[1]), 0.0, float(beta[0])
            if a < 0.0:
                a, c = 0.0, self._intercept_only(y, w)
        self.a_, self.b_, self.c_ = a, b, c

    @staticmethod
    def _intercept_only(y: np.ndarray, w: np.ndarray) -> float:
        p = float(np.average(y, weights=w))
        return float(logit(np.array([p]))[0])

    def _predict(self, s: np.ndarray) -> np.ndarray:
        return expit(self.a_ * np.log(s) + self.b_ * (-np.log1p(-s)) + self.c_)

    @property
    def affine_logit_coeffs_(self) -> tuple[float, float] | None:
        """``(a, c)`` for the tied variants; ``None`` for ``"abm"``.

        With ``a = b`` the map reduces to ``logit g = a * logit(s) + c``,
        which is affine on the logit scale; the full three-parameter map
        is not (see the shap-calibration chapter).
        """
        self._check_fitted()
        if self.variant in ("ab", "a"):
            return (self.a_, self.c_)
        return None

    def interpret(self) -> Interpretation:
        """Read the fitted exponents and intercept against the identity (1, 1, 0)."""
        self._check_fitted()
        messages = [
            (
                f"a = {self.a_:.3f}: sensitivity near s -> 0; a < 1 raises the smallest "
                "probabilities (model was overconfident in the low tail), a > 1 deepens them"
            ),
            (
                f"b = {self.b_:.3f}: sensitivity near s -> 1; the mirrored reading for the "
                "high tail"
            ),
            (
                f"c = {self.c_:.3f}: base-rate shift of {self.c_:+.3f} log-odds, odds factor "
                f"{np.exp(self.c_):.3f}"
            ),
            "identity map corresponds to (a, b, c) = (1, 1, 0)",
        ]
        if abs(self.a_ - self.b_) > 0.1:
            messages.append(
                f"a != b (gap {self.a_ - self.b_:+.3f}): asymmetric tail distortion that no "
                "symmetric (Platt/temperature) map could express"
            )
        if self.constraint_active_:
            messages.append(
                "monotonicity constraint a, b >= 0 was active: a negative exponent was "
                "dropped and the model refitted (betacal strategy)"
            )
        return Interpretation(
            method=type(self).__name__,
            param_names=("a", "b", "c"),
            param_values=(self.a_, self.b_, self.c_),
            messages=tuple(messages),
        )
