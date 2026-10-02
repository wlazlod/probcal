"""Spline calibrator: penalized natural cubic splines on the logit scale.

Theory: ``docs/concepts/methods-nonparametric.md``.

References
----------
Lucena (2018); Hastie, Tibshirani & Friedman (2009), §5.2.1 — full records in
the documentation.
"""

import warnings

import numpy as np

from ._math import expit, irls_logistic, logit, natural_cubic_basis
from ._registry import register
from ._results import Interpretation
from ._steps import migrate_state
from ._validation import EPS, stratified_folds, validate_cv, validate_positive_int
from .base import BaseCalibrator

_MONOTONE_RTOL = 1e-12
"""Relative slack (of the largest |slope|) when checking the logit-scale slope."""


def _second_difference_penalty(k: int) -> np.ndarray:
    """P = D'D for the (k-2, k) second-difference matrix on the coefficients."""
    if k < 3:
        return np.zeros((k, k))
    d = np.zeros((k - 2, k))
    for i in range(k - 2):
        d[i, i : i + 3] = (1.0, -2.0, 1.0)
    return d.T @ d


def _log_loss_sum(y: np.ndarray, w: np.ndarray, eta: np.ndarray) -> float:
    p = np.clip(expit(eta), EPS, 1.0 - EPS)
    return float(-np.sum(w * (y * np.log(p) + (1.0 - y) * np.log1p(-p))))


@register
class SplineCalibrator(BaseCalibrator):
    """Natural cubic spline calibration on the logit scale.

    Models ``logit g(s) = sum_k theta_k N_k(logit s)`` with the natural cubic
    basis (linear beyond the boundary knots), fitted by penalized IRLS with a
    second-difference roughness penalty — the package's safeguarded Newton
    solver (:func:`probcal._math.irls_logistic` with a penalty matrix:
    step-halving, full-step convergence test, separation fallback). The
    penalty weight is chosen by K-fold cross-validated log loss within the
    calibration set.

    Parameters
    ----------
    n_knots : int or None
        Number of knots (placed at equally spaced quantiles of the logit
        scores); defaults to ``clip(ceil(n^(1/3)), 4, 12)``.
    lambdas : array_like or None
        Candidate penalty weights (finite, non-negative); defaults to
        ``logspace(-4, 4, 17)``.
    cv : int
        Inner fold count for the lambda search; an integer ``>= 2``.
    random_state : int
        Seed for the stratified fold assignment.

    Attributes
    ----------
    lambda_ : float
        Selected penalty weight.
    edof_ : float
        Effective degrees of freedom — trace of the smoother matrix at the
        fitted solution; the honest complexity measure.
    n_knots_ : int
        Number of knots actually used.
    knots_ : numpy.ndarray
        Knot locations on the logit scale.
    theta_ : numpy.ndarray
        Spline coefficients on the natural cubic basis.
    converged_ : bool
        Whether the final penalized IRLS fit converged; if ``False`` a
        warning was raised at fit time and ``interpret()`` records it.
    is_monotone_ : bool
        Checked exactly after fitting: the logit-scale slope is evaluated at
        every knot (which also gives both linear tails) and at each interior
        extremum of the piecewise-quadratic slope. The penalty does not
        enforce monotonicity, and a non-monotone fit is flagged with a
        warning.

    References
    ----------
    Lucena (2018); Hastie, Tibshirani & Friedman (2009), §5.2.1.
    """

    _STATE_ATTRS = (
        "knots_",
        "n_knots_",
        "lambdas_grid_",
        "lambda_",
        "theta_",
        "edof_",
        "is_monotone_",
        "converged_",
    )

    def __init__(
        self,
        n_knots: int | None = None,
        lambdas: object = None,
        cv: int = 5,
        random_state: int = 42,
    ) -> None:
        self.n_knots = n_knots
        self.lambdas = lambdas
        self.cv = cv
        self.random_state = random_state

    def _set_state(self, state: dict[str, object]) -> None:
        super()._set_state(migrate_state(state, {"_knots": "knots_", "_theta": "theta_"}))

    def _fit(self, s: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        cv = validate_cv(self.cv, y)
        n = len(s)
        z = logit(s)
        if self.n_knots is None:
            k = int(np.clip(np.ceil(n ** (1 / 3)), 4, 12))
        else:
            k = validate_positive_int(self.n_knots, "n_knots")
        knots = np.unique(np.quantile(z, np.linspace(0.0, 1.0, k)))
        if len(knots) < 3:
            raise ValueError("SplineCalibrator: fewer than 3 distinct knots; scores too tied")
        grid = (
            np.logspace(-4.0, 4.0, 17)
            if self.lambdas is None
            else np.asarray(self.lambdas, dtype=np.float64)
        )
        if grid.ndim != 1 or grid.size == 0 or not np.all(np.isfinite(grid)) or np.any(grid < 0):
            raise ValueError(
                "lambdas must be a non-empty 1-D sequence of finite values >= 0, "
                f"got {self.lambdas!r}"
            )
        self.knots_ = knots
        self.n_knots_ = int(len(knots))
        self.lambdas_grid_ = grid

        basis = natural_cubic_basis(z, knots)
        penalty = _second_difference_penalty(basis.shape[1])

        folds = stratified_folds(y, cv, self.random_state)
        cv_loss = np.zeros(len(grid))
        with warnings.catch_warnings():
            # A separating training fold falls back to its ridge refit; the
            # warning belongs to the final fit, not to the inner search.
            warnings.simplefilter("ignore", UserWarning)
            for j, lam in enumerate(grid):
                for f in range(cv):
                    tr, va = folds != f, folds == f
                    res = irls_logistic(basis[tr], y[tr], w=w[tr], penalty=float(lam) * penalty)
                    cv_loss[j] += _log_loss_sum(y[va], w[va], basis[va] @ res.beta)
        self.lambda_ = float(grid[int(np.argmin(cv_loss))])

        res = irls_logistic(basis, y, w=w, penalty=self.lambda_ * penalty)
        self.theta_ = res.beta
        self.converged_ = bool(res.converged)
        # Smoother trace at the *final* coefficients (not the last iterate's weights).
        mu = expit(basis @ self.theta_)
        bwb = (basis * (w * mu * (1.0 - mu))[:, None]).T @ basis
        hess = bwb + self.lambda_ * penalty
        if res.separation:
            hess = hess + 1e-6 * np.eye(basis.shape[1])  # the ridge the fallback fit used
        self.edof_ = float(np.trace(np.linalg.solve(hess, bwb)))
        if not self.converged_:
            warnings.warn(
                "SplineCalibrator: penalized IRLS did not converge; the fitted curve may be "
                "unreliable — inspect interpret() or use a larger penalty",
                UserWarning,
                stacklevel=2,
            )

        self.is_monotone_ = self._check_monotone_exact()
        if not self.is_monotone_:
            warnings.warn(
                "SplineCalibrator: fitted curve is not monotone; consider a larger "
                "penalty or a monotone calibrator for ranking-sensitive use",
                UserWarning,
                stacklevel=2,
            )

    def _check_monotone_exact(self) -> bool:
        """Logit-scale slope ``f'`` is non-negative over the whole real line.

        ``f`` is cubic between knots and linear beyond the boundary knots, so
        ``f'`` is piecewise quadratic with constant tails equal to its values
        at the boundary knots; its minimum is attained at a knot or where the
        piecewise-linear ``f''`` crosses zero inside a knot interval.
        """
        kn, theta = self.knots_, self.theta_
        d1 = natural_cubic_basis(kn, kn, deriv=1) @ theta
        d2 = natural_cubic_basis(kn, kn, deriv=2) @ theta
        left, right = d2[:-1], d2[1:]
        cross = (left * right) < 0.0
        x_star = kn[:-1][cross] + left[cross] / (left[cross] - right[cross]) * np.diff(kn)[cross]
        slopes = np.concatenate([d1, natural_cubic_basis(x_star, kn, deriv=1) @ theta])
        scale = max(1.0, float(np.max(np.abs(slopes))))
        return bool(np.min(slopes) >= -_MONOTONE_RTOL * scale)

    @property
    def complexity_rank(self) -> float:
        """Parsimony rank 12.0: a penalized basis expansion, more flexible than binning."""
        return 12.0

    def _predict(self, s: np.ndarray) -> np.ndarray:
        return expit(natural_cubic_basis(logit(s), self.knots_) @ self.theta_)

    def interpret(self) -> Interpretation:
        """Read effective degrees of freedom as the honest complexity measure."""
        self._check_fitted()
        messages = [
            (
                f"effective degrees of freedom {self.edof_:.2f} (trace of the smoother): "
                "values near 2 mean a parametric family would have sufficed; larger values "
                "mean the curvature is real"
            ),
            (
                f"penalty lambda = {self.lambda_:.4g} chosen by {self.cv}-fold "
                f"cross-validated log loss over {len(self.lambdas_grid_)} candidates; "
                f"{self.n_knots_} knots at logit-score quantiles"
            ),
            (
                "regions where the fitted curve runs steeper than the identity are locally "
                "underconfident score regions; shallower, locally overconfident"
            ),
        ]
        if not self.is_monotone_:
            messages.append("fitted curve is NOT monotone (warned at fit)")
        if getattr(self, "converged_", True) is False:
            messages.append("penalized IRLS did not converge; the curve may be unreliable")
        return Interpretation(
            method=type(self).__name__,
            param_names=("edof", "lambda", "n_knots"),
            param_values=(self.edof_, self.lambda_, float(self.n_knots_)),
            messages=tuple(messages),
        )
