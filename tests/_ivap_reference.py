"""Brute-force IVAP pair refit: the equivalence gate's ground truth for the
Vovk-Petej precomputation.

Since 0.4.0 the definition pools tied calibration scores into one point (weights
summed, weighted-mean label) and merges a query whose score equals a calibration
score into that point, so the result is invariant to row order. Up to 0.3.x the
reference (and the calibrator) inserted the query at ``searchsorted(...,
side="left")`` among the *unpooled* rows, which made the output depend on how
tied rows were ordered. Do not optimize this file — it is the slow, literal
definition: two full PAVA fits per query, read back at the query's point.
"""

import numpy as np

from probcal._math import pava


def pooled_calibration(
    s: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distinct sorted scores, weighted-mean labels, summed weights."""
    s_u, inv = np.unique(s, return_inverse=True)
    w_u = np.bincount(inv, weights=w)
    y_u = np.bincount(inv, weights=w * y) / w_u
    return s_u, y_u, w_u


def pair_at(s_u: np.ndarray, y_u: np.ndarray, w_u: np.ndarray, x: float) -> tuple[float, float]:
    """``(p0, p1)``: isotonic fit at a unit-weight query ``x`` labeled 0 / 1."""
    idx = int(np.searchsorted(s_u, x, side="left"))
    tie = idx < len(s_u) and s_u[idx] == x
    p = []
    for label in (0.0, 1.0):
        if tie:
            w_aug = w_u.copy()
            y_aug = y_u.copy()
            y_aug[idx] = (y_u[idx] * w_u[idx] + label) / (w_u[idx] + 1.0)
            w_aug[idx] = w_u[idx] + 1.0
        else:
            w_aug = np.insert(w_u, idx, 1.0)
            y_aug = np.insert(y_u, idx, label)
        p.append(float(pava(y_aug, w_aug).fitted[idx]))
    return p[0], p[1]
