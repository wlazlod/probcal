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
from ._registry import register
from ._results import Interpretation
from .base import (
    BaseCalibrator,
    UnattainableTargetError,
    _check_representable,
    _validate_point_targets,
)

_U_LO = 1e-6
_U_HI = 1e6

_IRLS_NOT_CONVERGED = (
    "IRLS did not converge; coefficients may be unreliable — inspect interpret() "
    "and consider a nonparametric calibrator"
)

_BETA_INVERSE_KAPPA = 1.524  # minimax hyperbola parameter (max deviation 0.076)


def _beta_point_inverse_z(
    K: np.ndarray,
    a: float,
    b: float,
    kappa: float = _BETA_INVERSE_KAPPA,
    max_steps: int = 4,
    rtol: float = 1e-13,
    rtol_final: float = 1e-10,
) -> np.ndarray:
    """Exact-to-machine-precision root of ``a*z + (b-a)*softplus(z) = K``.

    Layer 1 seeds with the minimax-hyperbola approximation to softplus
    (``kappa=1.524``, max deviation 0.076): the admissible root of the
    resulting quadratic is exact at ``a=b`` and in both tails, with error
    bounded by ``0.076*|b-a|/min(a,b)`` elsewhere. Layer 2 refines with up to
    ``max_steps`` (default 4) Halley corrections, exiting early once the
    residual certificate ``|f(z)| <= rtol * max(1, |K|)`` is met; the
    certificate bounds the coordinate error by ``|f(z)| / min(a, b)``. The
    initial design was a fixed 2 steps (machine precision up to
    asymmetry ratio 3, ~1e-10 at ratio 10, ~1e-4 at ratio 50 without a 3rd
    step); the certified cap is a finite, bounded expression — not
    open-ended iteration — that reaches machine precision at ratio 50 in
    <= 3 steps without paying a 4th step in the common (low-asymmetry) case.

    Machine precision is verified numerically over
    ``a, b in (0, 5]`` and asymmetry ratio ``<= 50`` — the realistic fit
    domain. Outside it (extreme exponent ratios, e.g. ``a=1e-4, b=5``), the
    seed can be far off and 4 Halley steps may not recover full precision;
    rather than silently return an uncertified value (the package's
    no-silent-clamp doctrine — see ``UnattainableTargetError``), a final
    residual is recomputed after the loop and checked against a looser
    tolerance ``rtol_final * max(1, |K|)``. A failure raises ``RuntimeError``
    naming the worst residual: the method either returns a certified-exact
    root, or it raises — never a silently uncertified one.

    Parameters
    ----------
    K : numpy.ndarray
        Target values ``logit(p) - c``.
    a, b : float
        Beta-calibration exponents, both strictly positive (degenerate
        ``a=0``/``b=0``/``a=b`` cases are handled by the caller).
    kappa : float
        Minimax hyperbola parameter for the Layer-1 seed.
    max_steps : int
        Fixed Halley iteration cap.
    rtol : float
        Early-exit residual certificate tolerance, relative to
        ``max(1, |K|)``.
    rtol_final : float
        Post-loop certificate tolerance, relative to ``max(1, |K|)``; looser
        than ``rtol`` (the error bound ``residual / min(a, b)`` stays
        negligible at ``1e-10``) but still enforced unconditionally.

    Returns
    -------
    numpy.ndarray
        ``z`` solving the equation, elementwise, certified to
        ``|h(z) - K| <= rtol_final * max(1, |K|)``.

    Raises
    ------
    RuntimeError
        If the post-loop residual certificate is not met anywhere in ``K``
        (only reachable at exponent ratios far outside the verified
        domain — see above); names the worst residual and suggests
        ``interval_inverse``.
    """
    z = ((a + b) * K - (b - a) * np.sqrt(K**2 + kappa * a * b)) / (2.0 * a * b)
    for _ in range(max_steps):
        s = expit(z)
        f = a * z + (b - a) * np.logaddexp(0.0, z) - K
        if np.all(np.abs(f) <= rtol * np.maximum(1.0, np.abs(K))):
            break
        f1 = a + (b - a) * s
        f2 = (b - a) * s * (1.0 - s)
        z = z - 2.0 * f * f1 / (2.0 * f1**2 - f * f2)
    f_final = a * z + (b - a) * np.logaddexp(0.0, z) - K
    tol_final = rtol_final * np.maximum(1.0, np.abs(K))
    if np.any(np.abs(f_final) > tol_final):
        raise RuntimeError(
            f"beta point inverse (a={a:g}, b={b:g}) failed to certify to residual <= "
            f"{rtol_final:g}*max(1,|K|) after {max_steps} Halley steps (max observed "
            f"residual {float(np.max(np.abs(f_final))):.3g}); this exponent ratio is "
            "outside the verified domain — use interval_inverse"
        )
    return z


@register
class PlattCalibrator(BaseCalibrator):
    """Logistic recalibration on the logit scale (Platt scaling).

    Fits ``logit g(s) = a * logit(s) + b`` by IRLS with Lin–Lin–Weng smoothed
    targets ``(N+ + 1)/(N+ + 2)`` and ``1/(N- + 2)`` for stability on small
    samples, where ``N+``/``N-`` are the weighted class masses (row counts
    under unit weights), so that integer weights match row duplication. The
    identity map is ``(a, b) = (1, 0)``.

    Attributes
    ----------
    a_ : float
        Fitted slope — spread correction: ``a < 1`` shrinks overconfident
        scores toward the base rate, ``a > 1`` sharpens underconfident ones.
    b_ : float
        Fitted intercept — calibration-in-the-large shift in log-odds.
    converged_ : bool
        Whether IRLS converged; if ``False`` a warning was raised at fit time
        and ``interpret()`` records it.

    References
    ----------
    Platt (1999); Lin, Lin & Weng (2007). The logistic family fitted on raw
    SVM outputs (Platt's original setting) does not contain the identity; on
    logits it does — see the parametric-methods chapter.
    """

    _STATE_ATTRS = ("a_", "b_", "is_monotone_", "converged_")

    def _fit(self, s: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        z = logit(s)
        n_pos = float(np.sum(w * (y == 1.0)))
        n_neg = float(np.sum(w * (y == 0.0)))
        targets = np.where(y == 1.0, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))
        X = np.column_stack([np.ones_like(z), z])
        res = irls_logistic(X, targets, w=w)
        self.b_ = float(res.beta[0])
        self.a_ = float(res.beta[1])
        self.is_monotone_ = self.a_ > 0.0
        self.converged_ = bool(res.converged)
        if not self.converged_:
            warnings.warn(_IRLS_NOT_CONVERGED, UserWarning, stacklevel=2)

    def _predict(self, s: np.ndarray) -> np.ndarray:
        return expit(self.a_ * logit(s) + self.b_)

    @property
    def affine_logit_coeffs_(self) -> tuple[float, float] | None:
        """``(a, b)``: Platt scaling is affine on the logit scale."""
        self._check_fitted()
        return (self.a_, self.b_)

    @property
    def complexity_rank(self) -> float:
        """Parsimony rank 2.0: a two-parameter map, simpler than the nonparametric methods."""
        return 2.0

    def _closed_inverse(self, t: float) -> float:
        return float(expit(np.array([(logit(np.array([t]))[0] - self.b_) / self.a_]))[0])

    def _inverse_left(self, t: float) -> float:
        return self._closed_inverse(t)

    def _inverse_right(self, t: float) -> float:
        return self._closed_inverse(t)

    def interpret(self) -> Interpretation:
        """Read the fitted slope and intercept against the identity ``(1, 0)``.

        If IRLS did not converge at fit time (a ``UserWarning`` was raised),
        the messages include a note not to trust the coefficients.
        """
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
        messages = [slope_msg, int_msg, "identity map corresponds to (a, b) = (1, 0)"]
        if not self.converged_:
            messages.append("IRLS did not converge; coefficients may be unreliable")
        return Interpretation(
            method=type(self).__name__,
            param_names=("a", "b"),
            param_values=(self.a_, self.b_),
            messages=tuple(messages),
        )


@register
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

    _STATE_ATTRS = ("T_",)

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

    @property
    def complexity_rank(self) -> float:
        """Parsimony rank 1.0: the simplest map, a single parameter."""
        return 1.0

    def _closed_inverse(self, t: float) -> float:
        return float(expit(np.array([self.T_ * logit(np.array([t]))[0]]))[0])

    def _inverse_left(self, t: float) -> float:
        return self._closed_inverse(t)

    def _inverse_right(self, t: float) -> float:
        return self._closed_inverse(t)

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


@register
class BetaCalibrator(BaseCalibrator):
    """Beta calibration: ``logit g(s) = a·ln s − b·ln(1 − s) + c``.

    Variants: ``"abm"`` fits ``(a, b, c)``;
    ``"ab"`` ties ``a = b`` (equivalent to Platt scaling on logits); ``"a"``
    additionally fixes ``c = 0`` (a single-parameter map, the temperature
    family in a different parameterization). The monotonicity constraint
    ``a, b >= 0`` is enforced by the betacal refit strategy: a negative
    exponent drops its feature and refits.

    Parameters
    ----------
    variant : {"abm", "ab", "a"}
        ``"abm"`` (default) fits the full ``(a, b, c)``; ``"ab"`` ties
        ``a = b``; ``"a"`` additionally fixes ``c = 0``.

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
    converged_ : bool
        Whether the fit whose coefficients survive converged (``True`` for
        closed-form paths); if ``False`` a warning was raised at fit time and
        ``interpret()`` records it.
    separation_fallback_ : bool
        Whether any IRLS call during fitting detected separation and fell
        back to the ridge-regularized fit; recorded by ``interpret()``.

    References
    ----------
    Kull, Silva Filho & Flach (2017), AISTATS 54 and EJS 11(2). The identity
    is ``(a, b, c) = (1, 1, 0)``: beta calibration cannot un-calibrate an
    already calibrated model. ``a != b`` captures asymmetric tail distortion;
    temperature is the special case ``a = b = 1/T, c = 0``.
    """

    _STATE_ATTRS = (
        "a_",
        "b_",
        "c_",
        "constraint_active_",
        "converged_",
        "separation_fallback_",
    )

    def __init__(self, variant: str = "abm") -> None:
        self.variant = variant

    def _fit(self, s: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        if self.variant not in ("abm", "ab", "a"):
            raise ValueError(f"variant must be 'abm', 'ab', or 'a', got {self.variant!r}")
        ln_s = np.log(s)
        ln_1ms = -np.log1p(-s)  # -ln(1 - s), non-negative
        self.constraint_active_ = False
        self.converged_ = True
        self.separation_fallback_ = False

        if self.variant == "abm":
            self._fit_abm(ln_s, ln_1ms, y, w)
        elif self.variant == "ab":
            z = ln_s + ln_1ms  # logit(s)
            X = np.column_stack([np.ones_like(z), z])
            beta = self._irls(X, y, w)
            a = float(beta[1])
            if a < 0.0:
                self.constraint_active_ = True
                a = 0.0
                beta0 = self._intercept_only(y, w)
                self.c_ = beta0
                self.converged_ = True  # surviving coefficient is closed-form
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

        if not self.converged_:
            warnings.warn(_IRLS_NOT_CONVERGED, UserWarning, stacklevel=2)

    def _irls(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        # Last call wins for converged_ (its coefficients survive the cascade);
        # separation_fallback_ is sticky so interpret() records any fallback.
        res = irls_logistic(X, y, w=w)
        self.converged_ = bool(res.converged)
        self.separation_fallback_ = self.separation_fallback_ or bool(res.separation)
        return res.beta

    def _fit_abm(self, ln_s: np.ndarray, ln_1ms: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        ones = np.ones_like(ln_s)
        beta = self._irls(np.column_stack([ones, ln_s, ln_1ms]), y, w)
        a, b, c = float(beta[1]), float(beta[2]), float(beta[0])
        if a < 0.0:
            self.constraint_active_ = True
            beta = self._irls(np.column_stack([ones, ln_1ms]), y, w)
            a, b, c = 0.0, float(beta[1]), float(beta[0])
            if b < 0.0:
                b, c = 0.0, self._intercept_only(y, w)
                self.converged_ = True  # surviving coefficient is closed-form
        elif b < 0.0:
            self.constraint_active_ = True
            beta = self._irls(np.column_stack([ones, ln_s]), y, w)
            a, b, c = float(beta[1]), 0.0, float(beta[0])
            if a < 0.0:
                a, c = 0.0, self._intercept_only(y, w)
                self.converged_ = True  # surviving coefficient is closed-form
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

    def point_inverse(self, p: object, *, space: str = "probability") -> np.ndarray:
        """Raw scores whose calibrated probabilities equal ``p`` (exact preimage).

        Overrides :meth:`BaseCalibrator.point_inverse` with the beta
        family's own exact construction, so the ``"abm"``
        variant — not affine on the logit scale — still gets a closed-form
        inverse instead of falling back to :meth:`interval_inverse`'s
        bisection. With ``z = logit(s)`` and ``K = logit(p) - c``, the
        forward map is ``a*z + (b-a)*softplus(z) = K``, solved by a
        minimax-hyperbola seed refined by up to 4 certified Halley steps
        (:func:`_beta_point_inverse_z`). Degenerate exponents are handled by
        dedicated closed forms: ``a == b`` collapses to the affine formula
        ``z = K/a``; ``a == 0`` (``h`` ranges over ``(0, inf)``, attainable
        probability range ``(sigma(c), 1)``) gives ``z = ln(expm1(K/b))``;
        ``b == 0`` (range ``(-inf, 0)``, attainable range ``(0, sigma(c))``)
        gives ``z = -ln(expm1(-K/a))``; ``a == b == 0`` is a constant map
        with no point inverse.

        Parameters
        ----------
        p : array_like
            Calibrated probabilities strictly inside ``(0, 1)``; boundary
            and out-of-range values raise ``UnattainableTargetError``
            (all-or-nothing, no silent clamp).
        space : {"probability", "logit"}, keyword-only
            Scale of the returned raw values.

        Returns
        -------
        numpy.ndarray
            Raw scores (or logits, if ``space="logit"``) whose calibrated
            probability equals ``p``.

        Raises
        ------
        RuntimeError
            If not yet fitted; or if the general (``a != b``, both nonzero)
            case fails to certify to machine precision after 4 Halley steps
            (only reachable at exponent ratios far outside the numerically
            verified domain, ``a, b in (0, 5]`` and ratio ``<= 50`` — see
            :func:`_beta_point_inverse_z`).
        ValueError
            If ``space`` is not ``"probability"`` or ``"logit"``.
        NotImplementedError
            If the calibrator is not monotone, or the fit collapsed to a
            constant map (``a == b == 0``): a constant map has no point
            inverse.
        UnattainableTargetError
            If any element of ``p`` lies outside the open interval
            ``(0, 1)``, or outside the attainable probability range of a
            degenerate (``a == 0`` or ``b == 0``) fit — ``p`` is validated
            all-or-nothing: if any element is outside the range (named in
            the error message), the whole call raises and no element is
            silently clamped. Also raised when ``space="probability"`` and
            the raw logit of any result exceeds ``logit(1 - 1e-12)`` in
            magnitude — the probability representation would round to
            0.0/1.0 and silently fail to round-trip; ``space="logit"`` is
            exact there.
        """
        self._check_fitted()
        if not self.is_monotone_:
            raise NotImplementedError(
                f"{type(self).__name__} is not monotone (is_monotone_=False); its preimage "
                "may be a union of intervals. Use a monotone calibrator for thresholding "
                "and recourse."
            )
        if space not in ("probability", "logit"):
            raise ValueError(f"space must be 'probability' or 'logit', got {space!r}")
        arr = _validate_point_targets(p)
        a, b = self.a_, self.b_
        K = logit(arr) - self.c_
        if a == 0.0 and b == 0.0:
            raise NotImplementedError(
                f"{type(self).__name__} fitted a constant map (a=b=0); it has no exact point "
                "inverse; use interval_inverse"
            )
        if a == b:
            z = K / a
        elif a == 0.0:
            lo = float(expit(np.array([self.c_]))[0])
            if np.any(K <= 0.0):
                raise UnattainableTargetError(
                    f"calibrated target is outside the attainable probability range "
                    f"({lo:.6g}, 1) of this degenerate (a=0) beta fit"
                )
            z = np.log(np.expm1(K / b))
        elif b == 0.0:
            hi = float(expit(np.array([self.c_]))[0])
            if np.any(K >= 0.0):
                raise UnattainableTargetError(
                    f"calibrated target is outside the attainable probability range "
                    f"(0, {hi:.6g}) of this degenerate (b=0) beta fit"
                )
            z = -np.log(np.expm1(-K / a))
        else:
            z = _beta_point_inverse_z(K, a, b)
        _check_representable(z, space)
        return z if space == "logit" else expit(z)

    @property
    def complexity_rank(self) -> float:
        """Parsimony rank by variant: 1.5 ("a"), 2.5 ("ab"), 3.0 ("abm").

        ``.get`` with a fallback because ``variant`` is validated only in
        ``_fit``; the property must not raise pre-fit.
        """
        return {"a": 1.5, "ab": 2.5, "abm": 3.0}.get(self.variant, 100.0)

    def interpret(self) -> Interpretation:
        """Read the fitted exponents and intercept against the identity (1, 1, 0).

        If IRLS did not converge at fit time (a ``UserWarning`` was raised),
        the messages include a note not to trust the coefficients.
        """
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
        if not self.converged_:
            messages.append("IRLS did not converge; coefficients may be unreliable")
        if self.separation_fallback_:
            messages.append(
                "separation was detected during fitting; at least one fit fell back to "
                "the ridge-regularized solution (ridge=1e-6)"
            )
        return Interpretation(
            method=type(self).__name__,
            param_names=("a", "b", "c"),
            param_values=(self.a_, self.b_, self.c_),
            messages=tuple(messages),
        )
