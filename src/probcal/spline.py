"""Spline calibrator: penalized natural cubic splines on the logit scale.

Theory: ``docs/concepts/methods-nonparametric.md``.

References
----------
Lucena (2018); Hastie, Tibshirani & Friedman (2009), §5.2.1 — full records in
the documentation.
"""

import warnings

import numpy as np

from ._math import expit, logit, natural_cubic_basis
from ._registry import register
from ._results import Interpretation
from .base import BaseCalibrator


def _second_difference_penalty(k: int) -> np.ndarray:
    """P = D'D for the (k-2, k) second-difference matrix on the coefficients."""
    if k < 3:
        return np.zeros((k, k))
    d = np.zeros((k - 2, k))
    for i in range(k - 2):
        d[i, i : i + 3] = (1.0, -2.0, 1.0)
    return d.T @ d


# Independent of _math.irls_logistic: the lam * penalty term already regularizes,
# so the separation handling of IRLS_SPEC / DECISIONS 57 is not needed here.
def _penalized_irls(
    basis: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    penalty: np.ndarray,
    lam: float,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Newton/IRLS for penalized logistic regression; returns (theta, B'WB)."""
    k = basis.shape[1]
    theta = np.zeros(k)
    bwb = np.zeros((k, k))
    for _ in range(max_iter):
        eta = np.clip(basis @ theta, -30.0, 30.0)
        mu = expit(eta)
        wt = w * mu * (1.0 - mu)
        grad = basis.T @ (w * (y - mu)) - lam * (penalty @ theta)
        bwb = (basis * wt[:, None]).T @ basis
        hess = bwb + lam * penalty + 1e-10 * np.eye(k)
        step = np.linalg.solve(hess, grad)
        theta = theta + step
        if np.max(np.abs(step)) < tol * (1.0 + np.max(np.abs(theta))):
            break
    return theta, bwb


@register
class SplineCalibrator(BaseCalibrator):
    """Natural cubic spline calibration on the logit scale.

    Models ``logit g(s) = sum_k theta_k N_k(logit s)`` with the natural cubic
    basis (linear beyond the boundary knots), fitted by penalized IRLS with a
    second-difference roughness penalty. The penalty weight is chosen by
    K-fold cross-validated log loss within the calibration set.

    Parameters
    ----------
    n_knots : int or None
        Number of knots (placed at equally spaced quantiles of the logit
        scores); defaults to ``clip(ceil(n^(1/3)), 4, 12)`` (DECISIONS entry).
    lambdas : array_like or None
        Candidate penalty weights; defaults to ``logspace(-4, 4, 17)``.
    cv : int
        Inner fold count for the lambda search.
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
    is_monotone_ : bool
        Checked on a dense grid after fitting; the penalty does not enforce
        monotonicity, and a rare non-monotone fit is flagged with a warning.

    References
    ----------
    Lucena (2018); Hastie, Tibshirani & Friedman (2009), §5.2.1.
    """

    _STATE_ATTRS = (
        "_knots",
        "n_knots_",
        "lambdas_grid_",
        "lambda_",
        "_theta",
        "edof_",
        "is_monotone_",
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

    def _fit(self, s: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        n = len(s)
        z = logit(s)
        k = self.n_knots if self.n_knots is not None else int(np.clip(np.ceil(n ** (1 / 3)), 4, 12))
        qs = np.linspace(0.0, 1.0, k)
        knots = np.unique(np.quantile(z, qs))
        if len(knots) < 3:
            raise ValueError("SplineCalibrator: fewer than 3 distinct knots; scores too tied")
        self._knots = knots
        self.n_knots_ = int(len(knots))
        grid = (
            np.logspace(-4.0, 4.0, 17)
            if self.lambdas is None
            else np.asarray(self.lambdas, dtype=np.float64)
        )
        self.lambdas_grid_ = grid

        basis = natural_cubic_basis(z, knots)
        penalty = _second_difference_penalty(basis.shape[1])

        folds = self._stratified_folds(y)
        cv_loss = np.zeros(len(grid))
        for j, lam in enumerate(grid):
            total = 0.0
            for f in range(self.cv):
                tr, va = folds != f, folds == f
                theta, _ = _penalized_irls(basis[tr], y[tr], w[tr], penalty, float(lam))
                p = expit(np.clip(basis[va] @ theta, -30.0, 30.0))
                p = np.clip(p, 1e-12, 1.0 - 1e-12)
                total += float(
                    -np.sum(w[va] * (y[va] * np.log(p) + (1.0 - y[va]) * np.log(1.0 - p)))
                )
            cv_loss[j] = total
        best = int(np.argmin(cv_loss))
        self.lambda_ = float(grid[best])

        theta, bwb = _penalized_irls(basis, y, w, penalty, self.lambda_)
        self._theta = theta
        hess = bwb + self.lambda_ * penalty + 1e-10 * np.eye(basis.shape[1])
        self.edof_ = float(np.trace(np.linalg.solve(hess, bwb)))

        probe = np.linspace(0.002, 0.998, 399)
        p_probe = self._predict(probe)
        self.is_monotone_ = bool(np.all(np.diff(p_probe) >= -1e-10))
        if not self.is_monotone_:
            warnings.warn(
                "SplineCalibrator: fitted curve is not monotone; consider a larger "
                "penalty or a monotone calibrator for ranking-sensitive use",
                UserWarning,
                stacklevel=2,
            )

    @property
    def complexity_rank(self) -> float:
        """Parsimony rank 12.0: a penalized basis expansion, more flexible than binning."""
        return 12.0

    def _stratified_folds(self, y: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        folds = np.empty(len(y), dtype=np.int64)
        for cls in (0.0, 1.0):
            idx = np.flatnonzero(y == cls)
            perm = rng.permutation(idx)
            folds[perm] = np.arange(len(perm)) % self.cv
        return folds

    def _predict(self, s: np.ndarray) -> np.ndarray:
        basis = natural_cubic_basis(logit(s), self._knots)
        return expit(np.clip(basis @ self._theta, -30.0, 30.0))

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
            messages.append("fitted curve is NOT monotone on the probe grid (warned at fit)")
        return Interpretation(
            method=type(self).__name__,
            param_names=("edof", "lambda", "n_knots"),
            param_values=(self.edof_, self.lambda_, float(self.n_knots_)),
            messages=tuple(messages),
        )
