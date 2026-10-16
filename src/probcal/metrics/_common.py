"""Shared input preparation and the sample-weight convention of ``probcal.metrics``.

Sample-weight convention
------------------------
Every ``sample_weight`` in ``probcal.metrics`` (and ``probcal.curves``) must be
strictly positive and finite. Weights are **relative frequency weights**:
every statistic is invariant to multiplying all weights by a constant, and
unit weights reproduce the unweighted numbers exactly.

* Point estimates (scores, binned/smooth calibration errors, fitted
  intercept/slope) are weighted means or weighted-likelihood estimates.
* Wherever a *sample size* enters -- a test statistic's variance, a
  finite-sample bias correction, a null envelope -- the variance is the one
  the weighted statistic actually has when ``y ~ Bernoulli(p)`` row by row:
  ``Var(sum(w * (y - p))) = sum(w**2 * p * (1 - p))``. Concretely,
  ``spiegelhalter_z`` and the ``ecce_curve`` null envelope use that variance
  directly; ``hosmer_lemeshow`` uses it per risk group (in the classical
  ``n pbar (1 - pbar)`` form); ``ece_debiased`` and
  ``murphy_decomposition(bias_corrected=True)`` use each bin's effective
  count ``sum_b(w)**2 / sum_b(w**2)``; and the likelihood-ratio tests of
  ``calibration_test`` and ``calibration_belt`` are taken on the weighted
  log-likelihood and referred to a second-order Rao-Scott corrected
  chi-square (:func:`rao_scott`), with sandwich covariances for the belt.
  A single global effective sample size (Kish's ``sum(w)**2 / sum(w**2)``)
  is *not* used: it misstates these variances whenever the
  weights vary with ``p``.

One count-form function reads weights literally: ``pluto_tasche_from_arrays``
sums them as obligor/default counts (integer weights equal row duplication).
``hl_e_test`` raises each likelihood-ratio factor to ``min(1, w / mean(w))``
within its grade, as ``CalibrationMonitor`` does (also scale-invariant). The
per-grade binomial/Jeffreys tests use raw integer counts and ignore weights.
"""

import numpy as np

from .._validation import validate_binary_y, validate_scores, validate_weights


def _prep(y: object, p: object, sample_weight: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validated ``(y, p, w)``: two-class binary ``y``, clipped ``p``, positive ``w``.

    ``w`` is all ones for ``sample_weight=None``.
    """
    y_arr = validate_binary_y(y)
    p_arr = validate_scores(p, name="p")
    if len(y_arr) != len(p_arr):
        raise ValueError("y and p must have equal length")
    w_arr = validate_weights(sample_weight, len(p_arr))
    return y_arr, p_arr, w_arr


def gram(X: np.ndarray, d: np.ndarray) -> np.ndarray:
    """``X' diag(d) X``."""
    return (X * d[:, None]).T @ X


def rao_scott(lr: float, X: np.ndarray, mu: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Second-order Rao-Scott correction of a weighted likelihood-ratio statistic.

    ``lr`` is twice the weighted log-likelihood gain of a logistic model with
    design ``X`` over a null that fixes all ``X.shape[1]`` coefficients, and
    ``mu`` the null (or fitted) probabilities. Under the null ``lr`` is
    distributed as ``sum(lambda_j * chi2_1)``, the ``lambda_j`` being the
    eigenvalues of ``I**-1 J`` with ``I = X' diag(w mu (1 - mu)) X`` (the
    information the weighted fit assumes) and ``J = X' diag(w**2 mu (1 -
    mu)) X`` (the actual variance of the weighted score). The returned
    ``(stat, df) = (lr / (lbar (1 + a2)), k / (1 + a2))``, ``lbar`` and
    ``a2`` the mean and squared coefficient of variation of the
    ``lambda_j``, is referred to ``chi2(df)`` (Rao & Scott, 1984). It is
    invariant to rescaling ``w``; for equal weights every ``lambda_j`` is
    the common weight and ``(lr / w[0], k)`` is returned exactly, so unit
    weights give the plain likelihood-ratio test.
    """
    k = X.shape[1]
    if is_uniform(w):
        return lr / float(w[0]), float(k)
    v = mu * (1.0 - mu)
    info = gram(X, w * v)
    meat = gram(X, w * w * v)
    try:
        lam = np.linalg.eigvals(np.linalg.solve(info, meat)).real
    except np.linalg.LinAlgError:
        lam = np.linalg.eigvals(np.linalg.pinv(info) @ meat).real
    lbar = float(lam.mean())
    if not lbar > 0.0:
        return lr, float(k)
    a2 = float(np.mean((lam - lbar) ** 2)) / lbar**2
    return lr / (lbar * (1.0 + a2)), k / (1.0 + a2)


def is_uniform(w: np.ndarray) -> bool:
    """``True`` for all-equal weights."""
    return bool(np.all(w == w[0]))
