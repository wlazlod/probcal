"""Recovery simulation for SegmentedCalibrator's empirical-Bayes shrinkage.

Usage: ``uv run python docs/scripts/segmented_sim.py [--fast]``

Compares three estimators of the per-segment residual log-odds offset
against six segments' true deltas: no pooling (the raw per-segment offset
MLE, ``delta_hat_`` — high variance on small segments), complete pooling
(one offset MLE fit on the pooled data across all segments, ignoring
segment identity — biased whenever segments truly differ), and the shipped
empirical-Bayes shrinkage (``delta_tilde_``). The heterogeneous scenario
(true deltas spread ``-0.6 .. +0.6``, segment sizes ``30 .. 3000``) gates
``mean_mse_eb <= mean_mse_no_pooling`` and ``mean_mse_eb <=
mean_mse_complete_pooling``; a second, homogeneous scenario (true deltas
all 0) checks empirical Bayes degrades gracefully to complete pooling as
the true segment spread shrinks to 0. ``tests/test_segmented_sim.py``
(``pytest.mark.slow``) enforces the same gates at a reduced run count in
CI; this script prints the full-size table pasted into
``docs/concepts/segmented.md``.
"""

import sys

import numpy as np

from probcal import SegmentedCalibrator
from probcal._math import expit, logit
from probcal.offset import estimate_offset

N_PER_SEGMENT = (30, 100, 300, 1000, 2000, 3000)
TRUE_DELTAS = (-0.6, -0.36, -0.12, 0.12, 0.36, 0.6)


def recovery(
    runs: int,
    n_per_segment: object,
    true_deltas: object,
    seed: int = 42,
) -> dict:
    """Simulated MSE of no-pooling, complete-pooling, and empirical-Bayes offsets.

    Each run draws ``len(n_per_segment)`` independent segments (score
    ``s ~ sigma(N(-1, 1))``, outcome ``y ~ Bernoulli(sigma(logit(s) +
    true_delta))``), fits a :class:`~probcal.segmented.SegmentedCalibrator`
    on the pooled data, and compares its per-segment ``delta_hat_`` (no
    pooling) and ``delta_tilde_`` (empirical Bayes) against a single offset
    MLE fit on the same pooled ``(y, p0)`` ignoring segment identity
    (complete pooling), all relative to the segment's ``true_delta``.

    Parameters
    ----------
    runs : int
        Number of independent draws to average over.
    n_per_segment : array_like of int
        Observation count per segment.
    true_deltas : array_like of float
        True residual log-odds offset per segment (same length as
        ``n_per_segment``).
    seed : int
        Seed for ``numpy.random.default_rng``.

    Returns
    -------
    dict
        ``mse_no_pooling``, ``mse_complete_pooling``, ``mse_eb`` (per
        segment, averaged over runs), their ``mean_*`` (averaged over
        segments too), and ``mean_delta_tilde`` (the empirical-Bayes
        estimate itself, averaged over runs — used to check it shrinks to
        0 when every segment's true delta is 0).
    """
    rng = np.random.default_rng(seed)
    n_arr = np.asarray(n_per_segment, dtype=np.int64)
    delta_arr = np.asarray(true_deltas, dtype=np.float64)
    n_segments = len(n_arr)
    labels = [f"seg{i}" for i in range(n_segments)]

    sq_no_pooling = np.zeros(n_segments)
    sq_complete = np.zeros(n_segments)
    sq_eb = np.zeros(n_segments)
    sum_delta_tilde = np.zeros(n_segments)

    for _ in range(runs):
        s_parts, y_parts, seg_parts = [], [], []
        for g in range(n_segments):
            n_g = int(n_arr[g])
            s_g = expit(rng.normal(-1.0, 1.0, n_g))
            p_true = expit(logit(s_g) + delta_arr[g])
            y_g = (rng.random(n_g) < p_true).astype(np.float64)
            s_parts.append(s_g)
            y_parts.append(y_g)
            seg_parts.append(np.full(n_g, labels[g]))
        s = np.concatenate(s_parts)
        y = np.concatenate(y_parts)
        segments = np.concatenate(seg_parts)

        cal = SegmentedCalibrator().fit(s, y, segments=segments)
        order = [cal.segments_.index(label) for label in labels]
        delta_hat = cal.delta_hat_[order]
        delta_tilde = cal.delta_tilde_[order]

        pooled = estimate_offset(y, cal.base_.predict_proba(s))
        delta_complete = np.full(n_segments, pooled.delta)

        sq_no_pooling += (delta_hat - delta_arr) ** 2
        sq_complete += (delta_complete - delta_arr) ** 2
        sq_eb += (delta_tilde - delta_arr) ** 2
        sum_delta_tilde += delta_tilde

    mse_no_pooling = sq_no_pooling / runs
    mse_complete = sq_complete / runs
    mse_eb = sq_eb / runs
    return {
        "mse_no_pooling": mse_no_pooling,
        "mse_complete_pooling": mse_complete,
        "mse_eb": mse_eb,
        "mean_mse_no_pooling": float(mse_no_pooling.mean()),
        "mean_mse_complete_pooling": float(mse_complete.mean()),
        "mean_mse_eb": float(mse_eb.mean()),
        "mean_delta_tilde": sum_delta_tilde / runs,
    }


def main() -> None:
    fast = "--fast" in sys.argv
    runs = 300 if fast else 2000

    hetero = recovery(runs, N_PER_SEGMENT, TRUE_DELTAS, seed=42)
    zero_spread = recovery(runs, (3000,) * 6, (0.0,) * 6, seed=123)

    rows = [
        (
            "heterogeneous (spread -0.6..+0.6)",
            runs,
            f"{hetero['mean_mse_no_pooling']:.4f}",
            f"{hetero['mean_mse_complete_pooling']:.4f}",
            f"{hetero['mean_mse_eb']:.4f}",
        ),
        (
            "homogeneous (all true delta = 0, n=3000)",
            runs,
            "-",
            "-",
            f"max|mean delta_tilde| = {float(np.max(np.abs(zero_spread['mean_delta_tilde']))):.4f}",
        ),
    ]
    headers = ("scenario", "runs", "MSE no-pooling", "MSE complete-pooling", "MSE / stat EB")
    widths = [len(h) for h in headers]
    str_rows = [tuple(str(c) for c in row) for row in rows]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    print("| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |")
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in str_rows:
        print("| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(row)) + " |")


if __name__ == "__main__":
    main()
