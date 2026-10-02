"""Shared input preparation and the sample-weight convention of ``probcal.metrics``.

Sample-weight convention
------------------------
Every ``sample_weight`` in ``probcal.metrics`` (and ``probcal.curves``) must be
strictly positive and finite. Weights are **relative frequency weights**:

* Point estimates (scores, binned/smooth calibration errors, fitted
  intercept/slope) are weighted means, so they are invariant to rescaling the
  weights, and unit weights reproduce the unweighted numbers exactly.
* Wherever a *sample size* enters -- a test statistic's variance, a
  finite-sample bias correction, a likelihood-ratio statistic -- the weights
  are first rescaled to ``w_eff = w * sum(w) / sum(w**2)``, so that they sum
  to Kish's effective sample size ``n_eff = sum(w)**2 / sum(w**2)``. This
  applies to ``spiegelhalter_z``, ``ece_debiased``,
  ``murphy_decomposition(bias_corrected=True)``, ``hosmer_lemeshow``,
  ``calibration_test``, ``calibration_belt`` and the ``ecce_curve`` null
  envelope. Unit weights give ``w_eff == w`` exactly (old numbers unchanged);
  multiplying all weights by a constant changes nothing; and a weight vector
  concentrated on a few rows cannot manufacture significance.

Two count-form functions document a different, literal reading:
``pluto_tasche_from_arrays`` sums weights as obligor/default counts (integer
weights equal row duplication) and ``hl_e_test`` uses weights as exponents on
the likelihood-ratio factors, matching ``CalibrationMonitor``. The per-grade
binomial/Jeffreys tests use raw integer counts and ignore weights.
"""

import numpy as np

from .._validation import validate_binary_y, validate_scores, validate_weights


def binary_y(y: object, *, require_both_classes: bool = True) -> np.ndarray:
    """``validate_binary_y``, optionally accepting a single-class target.

    ``require_both_classes=False`` serves the low-/zero-default portfolio
    functions (Pluto-Tasche, Jeffreys bands), where an all-zero ``y`` is the
    motivating case. The single-class check is the last one
    ``validate_binary_y`` performs, so every other check has passed when it is
    the one that fails.
    """
    try:
        return validate_binary_y(y)
    except ValueError as exc:
        if require_both_classes or "both classes" not in str(exc):
            raise
        return np.asarray(y, dtype=np.float64)


def _prep(
    y: object, p: object, sample_weight: object, *, require_both_classes: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validated ``(y, p, w)``: binary ``y``, clipped ``p``, positive ``w`` (ones if ``None``)."""
    y_arr = binary_y(y, require_both_classes=require_both_classes)
    p_arr = validate_scores(p, name="p")
    if len(y_arr) != len(p_arr):
        raise ValueError("y and p must have equal length")
    w_arr = validate_weights(sample_weight, len(p_arr))
    return y_arr, p_arr, w_arr


def effective_weights(w: np.ndarray) -> np.ndarray:
    """Kish-rescaled weights ``w * sum(w) / sum(w**2)``; they sum to ``n_eff``.

    Unit weights come back bit-identical (the factor is exactly 1.0).
    """
    return w * (float(w.sum()) / float(np.dot(w, w)))


def is_uniform(w: np.ndarray | None) -> bool:
    """``True`` for ``None`` or all-equal weights."""
    return w is None or bool(np.all(w == w[0]))
