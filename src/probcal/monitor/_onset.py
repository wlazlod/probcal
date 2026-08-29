"""Backward-CUSUM drift-onset estimate (spec M3).

``CalibrationMonitor`` stores, per batch, the additive plug-in log-LR
increment (``MonitorStep.log_e_increment``): the offset plug-in's own
log-LR contribution plus the shape plug-in's, when either is engaged (an
identity plug-in contributes exactly 0). The global e-value itself is a
logsumexp mixture and is *not* additive across batches, so localizing where
evidence started accumulating needs this separate, purely additive series.

:func:`estimate_onset` is a point estimate, not a change-point test: it has
no type-I control and no confidence set, and it can be reported alongside
an alarm to answer "which batch does the evidence trail point to", not "is
there a change point here with error probability alpha".
"""

import numpy as np


def estimate_onset(increments: np.ndarray) -> int:
    """Backward-CUSUM estimate of the batch index where drift began.

    Computes ``k* = argmax_k sum_{j >= k} increments[j]`` — the start of
    the suffix with the largest total plug-in log-LR evidence. Ties (equal
    maximal suffix sums) resolve to the LATEST ``k``.

    Parameters
    ----------
    increments : ndarray of shape (n_steps,)
        Per-batch additive plug-in log-LR increments, in arrival order
        (``MonitorStep.log_e_increment`` for each processed batch).

    Returns
    -------
    int
        The estimated onset index into ``increments`` (0-based). An
        estimate, not a test: it carries no error-rate guarantee.

    Raises
    ------
    ValueError
        If ``increments`` is empty.

    Examples
    --------
    >>> import numpy as np
    >>> from probcal.monitor._onset import estimate_onset
    >>> estimate_onset(np.array([-0.1, -0.2, -0.15, 5.0, 4.5, 4.8]))
    3
    """
    arr = np.asarray(increments, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("increments must be non-empty")
    suffix_sums = np.cumsum(arr[::-1])[::-1]
    best = np.max(suffix_sums)
    tied = np.flatnonzero(suffix_sums == best)
    return int(tied[-1])


__all__ = ["estimate_onset"]
