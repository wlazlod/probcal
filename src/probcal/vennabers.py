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

from ._math import pava
from ._results import Interpretation
from .base import BaseCalibrator


class VennAbersCalibrator(BaseCalibrator):
    """Inductive Venn–Abers predictor (IVAP).

    For a query score, two isotonic fits on the calibration set augmented
    with the query labeled 0 (resp. 1) yield the interval ``[p0, p1]``;
    ``predict_proba`` scalarizes it as ``p1 / (1 - p0 + p1)``.

    Batch prediction deduplicates query scores and runs two PAVA fits per
    unique score (DECISIONS entry: the O((n+m)log(n+m)) precomputation of
    Vovk & Petej is a planned optimization, not yet implemented).
    """

    def _fit(self, s: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        order = np.argsort(s, kind="stable")
        self._s = s[order]
        self._y = y[order]
        self._w = w[order]
        self._widths_cache: tuple[float, float] | None = None

    def _pair_at(self, x: float) -> tuple[float, float]:
        idx = int(np.searchsorted(self._s, x, side="left"))
        w_aug = np.insert(self._w, idx, 1.0)
        p = []
        for label in (0.0, 1.0):
            y_aug = np.insert(self._y, idx, label)
            p.append(float(pava(y_aug, w_aug).fitted[idx]))
        return p[0], p[1]

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
        uniq, inverse = np.unique(arr, return_inverse=True)
        pairs = np.array([self._pair_at(float(x)) for x in uniq])
        return pairs[inverse]

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
