"""Venn–Abers calibrators: inductive (IVAP) and cross (CVAP).

Theory, validity guarantee scope, and the scalarization caveat:
``docs/concepts/methods-distribution-free.md``. The guarantee attaches to the
interval returned by :meth:`VennAbersCalibrator.predict_interval`; the scalar
from ``predict_proba`` is the log-loss-minimax merger and is not itself covered
by the validity theorem.

References
----------
Vovk & Petej (2014) — full record in the documentation.
"""

import numpy as np

from ._results import Interpretation
from .base import BaseCalibrator


def _csd_sweep(c: np.ndarray, z: np.ndarray, dy: float) -> np.ndarray:
    """Fitted value of a unit-weight query labeled ``dy`` inserted at every
    position k = 0..n (Vovk & Petej 2014 precomputation, weighted CSD).

    With ``c`` the cumulative weights and ``z`` the cumulative ``w * y`` (both of
    length n+1 with a leading zero), the isotonic fit at the inserted query is
    ``max_{i<=k} min_{j>=k} (z[j] - z[i] + dy) / (c[j] - c[i] + 1)``. Shifting the
    prefix points by ``(-1, -dy)`` turns that into the slope of the bridge between
    the lower hull of the shifted prefix and the lower hull of the suffix, so one
    left-to-right sweep with incremental hull maintenance yields every position.

    Two slopes recur: the within-chain slope ``(z[j] - z[i]) / (c[j] - c[i])``
    (shift-free) and the cross-chain bridge ``(z[j] - z[i] + dy) /
    (c[j] - c[i] + 1)``. Both are written out inline over Python lists rather than
    through helpers over ``c``/``z``: the arithmetic is identical (bit for bit),
    but the sweep runs ~6x faster at n = 100,000.
    """
    cl = c.tolist()
    zl = z.tolist()
    n = len(cl) - 1

    # Suffix lower hull A_k..A_n, built right to left; top = leftmost alive. Each
    # push journals the points it hides so they can be resurfaced in O(1) later.
    hr = [n]
    journal: list[list[int]] = [[] for _ in range(n)]
    for k in range(n - 1, -1, -1):
        popped: list[int] = []
        while len(hr) >= 2:
            j1, j2 = hr[-1], hr[-2]
            if (zl[j1] - zl[k]) / (cl[j1] - cl[k]) < (zl[j2] - zl[j1]) / (cl[j2] - cl[j1]):
                break
            popped.append(hr.pop())
        journal[k] = popped
        hr.append(k)

    out = np.empty(n + 1)
    hl = [0]  # prefix hull of L_0..L_k; top = rightmost
    for k in range(n + 1):
        a = len(hl) - 1
        b = len(hr) - 1
        while True:  # walk both hull tops down to the bridge's tangent points
            i0, j0 = hl[a], hr[b]
            br = (zl[j0] - zl[i0] + dy) / (cl[j0] - cl[i0] + 1.0)
            if a > 0:
                im = hl[a - 1]
                if (zl[i0] - zl[im]) / (cl[i0] - cl[im]) >= br:
                    a -= 1
                    continue
            if b > 0:
                jm = hr[b - 1]
                if br >= (zl[jm] - zl[j0]) / (cl[jm] - cl[j0]):
                    b -= 1
                    continue
            break
        out[k] = br
        if k < n:
            hr.pop()  # A_k is always the current hull top
            hr.extend(reversed(journal[k]))  # resurface what A_k had hidden
            i = k + 1
            while len(hl) >= 2:
                i1, i2 = hl[-1], hl[-2]
                if (zl[i1] - zl[i2]) / (cl[i1] - cl[i2]) < (zl[i] - zl[i1]) / (cl[i] - cl[i1]):
                    break
                hl.pop()
            hl.append(i)
    return out


class VennAbersCalibrator(BaseCalibrator):
    """Inductive Venn–Abers predictor (IVAP).

    For a query score, two isotonic fits on the calibration set augmented
    with the query labeled 0 (resp. 1) yield the interval ``[p0, p1]``;
    ``predict_proba`` scalarizes it as ``p1 / (1 - p0 + p1)``.

    Both fits are precomputed at fit time by the Vovk & Petej (2014) cumulative-
    sum-diagram sweep, so prediction is a ``searchsorted`` gather rather than a
    pair of PAVA refits per unique query score.

    Attributes
    ----------
    F0_, F1_ : numpy.ndarray of shape (n + 1,)
        Fitted probabilities for a unit-weight query labeled 0 (resp. 1)
        inserted at each of the n+1 positions of the sorted calibration set.
        Both are non-decreasing, and ``F0_ <= F1_`` elementwise.

    Notes
    -----
    With non-unit sample weights the query still enters at weight 1, which is the
    natural generalization but sits outside the validity theorem as proved; see
    the scope note in ``docs/concepts/methods-distribution-free.md``.
    """

    def _fit(self, s: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        order = np.argsort(s, kind="stable")
        self._s = s[order]
        self._y = y[order]
        self._w = w[order]
        self._widths_cache: tuple[float, float] | None = None
        c = np.concatenate([[0.0], np.cumsum(self._w)])
        z = np.concatenate([[0.0], np.cumsum(self._w * self._y)])
        self.F0_ = _csd_sweep(c, z, 0.0)
        self.F1_ = _csd_sweep(c, z, 1.0)

    def predict_interval(self, s: object) -> np.ndarray:
        """Venn–Abers intervals ``[p0, p1]`` for new scores.

        Parameters
        ----------
        s : array_like
            Raw scores/probabilities in ``[0, 1]``.

        Returns
        -------
        numpy.ndarray of shape (n, 2)
            Columns ``p0`` (lower) and ``p1`` (upper). The distribution-free
            validity guarantee attaches to this pair, not to the scalarized
            ``predict_proba`` output.
        """
        self._check_fitted()
        from ._validation import validate_scores

        arr = validate_scores(s)
        idx = np.searchsorted(self._s, arr, side="left")
        return np.column_stack([self.F0_[idx], self.F1_[idx]])

    def _predict(self, s: np.ndarray) -> np.ndarray:
        intervals = self.predict_interval(s)
        p0, p1 = intervals[:, 0], intervals[:, 1]
        return p1 / (1.0 - p0 + p1)

    def interpret(self) -> Interpretation:
        """Report interval widths over the calibration scores — where to trust the map."""
        self._check_fitted()
        if self._widths_cache is None:
            intervals = self.predict_interval(self._s)
            widths = intervals[:, 1] - intervals[:, 0]
            self._widths_cache = (float(widths.mean()), float(widths.max()))
        mean_w, max_w = self._widths_cache
        return Interpretation(
            method=type(self).__name__,
            param_names=("mean_width", "max_width"),
            param_values=(mean_w, max_w),
            messages=(
                f"mean Venn–Abers interval width {mean_w:.4f}, maximum {max_w:.4f} over the "
                "calibration scores: width is per-score calibration uncertainty",
                "the validity guarantee holds for the interval [p0, p1] from "
                "predict_interval(); the scalar from predict_proba() is the log-loss-minimax "
                "merger p1/(1-p0+p1) and is not itself covered by the guarantee",
            ),
        )


class CrossVennAbersCalibrator(BaseCalibrator):
    """Cross Venn–Abers predictor (CVAP): fold-wise IVAPs, geometric-mean merge.

    Splits the calibration data into ``cv`` stratified folds; each fold's
    IVAP is fitted on the remaining folds. The scalar output merges the
    fold-wise pairs by the log-loss rule of Vovk & Petej:
    ``GM(p1) / (GM(1 - p0) + GM(p1))``. ``predict_interval`` returns the
    conservative envelope ``[min_k p0_k, max_k p1_k]`` (DECISIONS entry —
    the paper defines only the scalar merge).
    """

    def __init__(self, cv: int = 5, random_state: int = 42) -> None:
        self.cv = cv
        self.random_state = random_state

    def _fit(self, s: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        if self.cv < 2:
            raise ValueError("cv must be at least 2")
        rng = np.random.default_rng(self.random_state)
        folds = np.empty(len(y), dtype=np.int64)
        for cls in (0.0, 1.0):
            idx = np.flatnonzero(y == cls)
            perm = rng.permutation(idx)
            folds[perm] = np.arange(len(perm)) % self.cv
        self._ivaps: list[VennAbersCalibrator] = []
        for k in range(self.cv):
            mask = folds != k
            ivap = VennAbersCalibrator()
            ivap.fit(s[mask], y[mask], sample_weight=w[mask])
            self._ivaps.append(ivap)

    def _fold_pairs(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pairs = np.stack([ivap.predict_interval(s) for ivap in self._ivaps])  # (K, n, 2)
        return pairs[:, :, 0], pairs[:, :, 1]

    def _predict(self, s: np.ndarray) -> np.ndarray:
        p0, p1 = self._fold_pairs(s)
        gm_p1 = np.exp(np.mean(np.log(np.clip(p1, 1e-300, None)), axis=0))
        gm_1mp0 = np.exp(np.mean(np.log(np.clip(1.0 - p0, 1e-300, None)), axis=0))
        return gm_p1 / (gm_1mp0 + gm_p1)

    def predict_interval(self, s: object) -> np.ndarray:
        """Conservative fold envelope ``[min_k p0_k, max_k p1_k]`` (see class docs)."""
        self._check_fitted()
        from ._validation import validate_scores

        arr = validate_scores(s)
        p0, p1 = self._fold_pairs(arr)
        return np.column_stack([p0.min(axis=0), p1.max(axis=0)])

    def interpret(self) -> Interpretation:
        """Report fold count and envelope widths over a probe grid."""
        self._check_fitted()
        probe = np.linspace(0.01, 0.99, 99)
        env = self.predict_interval(probe)
        widths = env[:, 1] - env[:, 0]
        return Interpretation(
            method=type(self).__name__,
            param_names=("cv", "mean_envelope_width"),
            param_values=(float(self.cv), float(widths.mean())),
            messages=(
                f"{self.cv} stratified folds, one IVAP per fold; scalar output is the "
                "geometric-mean merge GM(p1)/(GM(1-p0)+GM(p1))",
                "predict_interval() returns the conservative fold envelope "
                "[min p0, max p1]; per-fold IVAP intervals carry the validity guarantee",
            ),
        )
