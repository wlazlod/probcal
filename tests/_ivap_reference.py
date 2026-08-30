"""Frozen v0.1.2 IVAP pair refit, kept verbatim as the equivalence gate's ground
truth for the Vovk-Petej precomputation. Do not "fix" or
optimize this file — its only job is to keep reproducing the pre-rewrite behavior,
tie convention included (``searchsorted`` with ``side="left"``, query inserted at
unit weight, two full PAVA fits read back at the insertion index).
"""

import numpy as np

from probcal._math import pava


def pair_at(
    s_sorted: np.ndarray, y_sorted: np.ndarray, w_sorted: np.ndarray, x: float
) -> tuple[float, float]:
    """Verbatim copy of the v0.1.2 ``VennAbersCalibrator._pair_at``."""
    idx = int(np.searchsorted(s_sorted, x, side="left"))
    w_aug = np.insert(w_sorted, idx, 1.0)
    p = []
    for label in (0.0, 1.0):
        y_aug = np.insert(y_sorted, idx, label)
        p.append(float(pava(y_aug, w_aug).fitted[idx]))
    return p[0], p[1]
