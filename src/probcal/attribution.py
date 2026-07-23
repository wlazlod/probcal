"""SHAP / additive-attribution adjustment to calibrated outputs.

Post-hoc calibration breaks SHAP local accuracy: ``base + sum(phi)``
reconstructs the raw score, not the calibrated probability. This module
restores additivity on the calibrated scale — exactly for calibrators affine
on the logit scale, and by the Aumann–Shapley secant rescaling in general.
Theory, identifiability limits, and invariance properties:
``docs/concepts/shap-calibration.md``.

No shap import: plain arrays are accepted, and ``shap.Explanation`` objects
are duck-typed via their ``.values`` / ``.base_values`` attributes.

References
----------
Lundberg & Lee (2017); Lundberg et al. (2020); Sundararajan, Taly & Yan
(2017); Aumann & Shapley (1974) — full records in the documentation.
"""

from dataclasses import dataclass

import numpy as np

from ._math import expit, logit

_DEGENERATE_EPS = 1e-8
_CENTRAL_DIFF_H = 1e-4


@dataclass(frozen=True)
class AdjustedAttribution:
    """Attributions rescaled to the calibrated output scale.

    Attributes
    ----------
    phi_adj : numpy.ndarray of shape (n, d)
        Adjusted per-feature attributions.
    base_adj : numpy.ndarray of shape (n,)
        Adjusted base values.
    target : numpy.ndarray of shape (n,)
        The calibrated output each row reconstructs
        (``base_adj + phi_adj.sum(axis=1)``).
    method_used : str
        ``"affine-exact"`` (exact Shapley values by linearity) or
        ``"aumann-shapley"`` (exact additivity; nonlinearity distributed
        proportionally to phi).
    max_reconstruction_error : float
        ``max |base_adj + sum(phi_adj) - target|`` over rows.
    """

    phi_adj: np.ndarray
    base_adj: np.ndarray
    target: np.ndarray
    method_used: str
    max_reconstruction_error: float


def _extract(phi: object, base_value: object) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(phi, "values") and hasattr(phi, "base_values"):
        base_value = np.asarray(phi.base_values, dtype=np.float64)
        phi = np.asarray(phi.values, dtype=np.float64)
    phi_arr = np.asarray(phi, dtype=np.float64)
    if phi_arr.ndim != 2:
        raise ValueError(f"phi must be 2-D (n, d), got shape {phi_arr.shape}")
    n = phi_arr.shape[0]
    base_arr = np.asarray(base_value, dtype=np.float64)
    if base_arr.ndim == 0:
        base_arr = np.full(n, float(base_arr))
    if base_arr.shape != (n,):
        raise ValueError(f"base_value must be scalar or shape ({n},), got {base_arr.shape}")
    return phi_arr, base_arr


def adjust_attributions(
    phi: object,
    base_value: object,
    calibrator: object,
    *,
    scale: str = "logit",
    method: str = "auto",
) -> AdjustedAttribution:
    """Rescale additive attributions so they sum to the calibrated output.

    Parameters
    ----------
    phi : array_like of shape (n, d) or shap.Explanation-like
        Raw attributions on the model's score scale (log-odds margins for
        ``scale="logit"``, probabilities for ``scale="probability"``).
        Objects exposing ``.values`` and ``.base_values`` are duck-typed;
        ``base_value`` is then ignored.
    base_value : float or array_like of shape (n,)
        SHAP base value(s) on the same scale as ``phi``.
    calibrator : fitted calibrator
        Any object with ``predict_proba``; ``affine_logit_coeffs_`` (when not
        None) enables the exact affine path.
    scale : {"logit", "probability"}
        Working scale of the attributions. Affine-exactness exists only on
        the logit scale.
    method : {"auto", "affine", "aumann-shapley"}
        ``"auto"`` uses affine-exact when available, else Aumann–Shapley.

    Returns
    -------
    AdjustedAttribution

    Raises
    ------
    ValueError
        If ``method="affine"`` is forced for a calibrator that is not affine
        on the logit scale (or on the probability scale, where no calibrator
        is affine).
    """
    if scale not in ("logit", "probability"):
        raise ValueError(f"scale must be 'logit' or 'probability', got {scale!r}")
    if method not in ("auto", "affine", "aumann-shapley"):
        raise ValueError(f"unknown method {method!r}")
    phi_arr, base_arr = _extract(phi, base_value)
    s = base_arr + phi_arr.sum(axis=1)

    coeffs = getattr(calibrator, "affine_logit_coeffs_", None)
    affine_available = scale == "logit" and coeffs is not None
    if method == "affine" and not affine_available:
        raise ValueError(
            "method='affine' requires a calibrator affine on the logit scale "
            "(affine_logit_coeffs_ is None, or scale='probability' was requested)"
        )
    use_affine = affine_available and method in ("auto", "affine")

    if scale == "logit":

        def g_work(t: np.ndarray) -> np.ndarray:
            return logit(calibrator.predict_proba(expit(t)))

    else:

        def g_work(t: np.ndarray) -> np.ndarray:
            return calibrator.predict_proba(np.clip(t, 1e-12, 1.0 - 1e-12))

    target = g_work(s)

    if use_affine:
        a, b = coeffs
        phi_adj = a * phi_arr
        base_adj = a * base_arr + b
        method_used = "affine-exact"
    else:
        g_s0 = g_work(base_arr)
        diff = s - base_arr
        multiplier = np.empty(len(s))
        regular = np.abs(diff) >= _DEGENERATE_EPS
        multiplier[regular] = (target[regular] - g_s0[regular]) / diff[regular]
        if np.any(~regular):
            b0 = base_arr[~regular]
            h = _CENTRAL_DIFF_H
            multiplier[~regular] = (g_work(b0 + h) - g_work(b0 - h)) / (2.0 * h)
        phi_adj = phi_arr * multiplier[:, None]
        base_adj = target - phi_adj.sum(axis=1)
        # base_adj equals g(s0) exactly on regular rows (telescoping); the
        # assignment above additionally zeroes reconstruction error on
        # degenerate rows where the local-slope multiplier is approximate.
        method_used = "aumann-shapley"

    recon_err = float(np.max(np.abs(base_adj + phi_adj.sum(axis=1) - target), initial=0.0))
    return AdjustedAttribution(
        phi_adj=phi_adj,
        base_adj=base_adj,
        target=target,
        method_used=method_used,
        max_reconstruction_error=recon_err,
    )
