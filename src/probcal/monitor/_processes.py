"""Log-space Bernoulli likelihood-ratio processes and predictable plug-ins.

Every e-process in the monitor multiplies factors
``LR_i(q) = q^y (1-q)^(1-y) / (p^y (1-p)^(1-y))`` whose conditional
expectation under the conditional-calibration null is exactly 1 for any
*predictable* alternative ``q`` — see ``docs/concepts/monitoring.md``.
Accumulation is in log space throughout; mixtures combine with logsumexp.
"""

import numpy as np

from .._math import bern_log_lr, expit, irls_logistic, logsumexp
from .._validation import EPS
from ..offset import _solve_delta


def plug_in_delta(z: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Predictable level plug-in: the LogitOffset mode-B shift on past data.

    Solves ``mean_w(sigma(z + delta)) = mean_w(y)`` by bisection — the same
    root ``LogitOffset(target_mean=...)`` finds. Degenerate pasts (no data,
    or an outcome rate outside (0, 1)) return 0.0: an honest "no evidence
    yet", which makes the plug-in factor exactly 1.
    """
    if z.size == 0:
        return 0.0
    target = float(np.average(y, weights=w))
    if not 0.0 < target < 1.0:
        return 0.0
    return _solve_delta(z, w, target)


def plug_in_shape(z: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Predictable Cox plug-in ``(c, a)`` from IRLS on past data.

    Returns the identity ``(0.0, 1.0)`` for degenerate pasts (no data, one
    class, or a non-finite fit) — the shape factor is then exactly 1.
    """
    if z.size == 0 or np.unique(y).size < 2:
        return 0.0, 1.0
    X = np.column_stack([np.ones_like(z), z])
    try:
        res = irls_logistic(X, y, w=w)
    except Exception:
        return 0.0, 1.0
    c, a = float(res.beta[0]), float(res.beta[1])
    if not (np.isfinite(c) and np.isfinite(a)):
        return 0.0, 1.0
    return c, a


class OffsetProcess:
    """Level e-process: average of the predictable plug-in and a grid mixture.

    Holds only log-accumulators; the caller supplies the predictable
    ``delta_hat`` computed from strictly earlier data.
    """

    def __init__(self, grid: tuple[float, ...]) -> None:
        self.grid = np.asarray(grid, dtype=np.float64)
        self.log_plug = 0.0
        self.log_mix = np.zeros(len(self.grid))

    def update(
        self, z: np.ndarray, p: np.ndarray, y: np.ndarray, w: np.ndarray, delta_hat: float
    ) -> float:
        """Advance the process one batch; returns the plug-in log-LR increment.

        Returns
        -------
        float
            ``bern_log_lr(y, p, sigma(z + delta_hat), w)``, or ``0.0`` when
            ``delta_hat == 0`` (the plug-in is the identity null). Used by
            :class:`~probcal.monitor.CalibrationMonitor` to accumulate
            ``MonitorStep.log_e_increment`` for drift-onset localization.
        """
        # delta_hat == 0 means the alternative IS the null: the factor is
        # exactly 1, no expit(logit(p)) round-trip noise.
        inc = 0.0
        if delta_hat != 0.0:
            inc = bern_log_lr(y, p, expit(z + delta_hat), w)
            self.log_plug += inc
        for j, d in enumerate(self.grid):
            self.log_mix[j] += bern_log_lr(y, p, expit(z + d), w)
        return inc

    def log_e(self) -> float:
        """log of ``(E_plug + E_mix) / 2`` (averages of e-values are e-values)."""
        log_e_mix = logsumexp(self.log_mix) - np.log(len(self.grid))
        return float(np.logaddexp(self.log_plug, log_e_mix) - np.log(2.0))

    def state(self) -> dict[str, object]:
        return {"log_plug": self.log_plug, "log_mix": self.log_mix.tolist()}

    def set_state(self, state: dict[str, object]) -> None:
        self.log_plug = float(state["log_plug"])  # type: ignore[arg-type]
        self.log_mix = np.asarray(state["log_mix"], dtype=np.float64)


class ConfidenceSequence:
    """Time-uniform confidence sequence for a log-odds offset.

    One e-process per shifted null ``delta_0`` on ``grid`` ("the forecast,
    moved by ``delta_0``, is calibrated"), each against the same predictable
    plug-in alternative ``sigma(z + delta_hat)``. A null whose running
    maximum log-e ever reaches ``-log(alpha)`` stays rejected, so the
    surviving set is a running intersection. Shared by the portfolio-level
    CS and every per-grade CS (same grid, each with its own plug-in).
    """

    def __init__(self, grid: np.ndarray) -> None:
        self.grid = grid
        self.log = np.zeros(len(grid))
        self.max = np.zeros(len(grid))

    def update(
        self, z: np.ndarray, p: np.ndarray, y: np.ndarray, w: np.ndarray, delta_hat: float
    ) -> None:
        """Advance every shifted-null e-process by one batch."""
        q_alt = np.clip(p if delta_hat == 0.0 else expit(z + delta_hat), EPS, 1.0 - EPS)
        log_q = np.log(q_alt)
        log_1mq = np.log1p(-q_alt)
        p0 = np.clip(expit(z[None, :] + self.grid[:, None]), EPS, 1.0 - EPS)
        terms = y * (log_q[None, :] - np.log(p0)) + (1.0 - y) * (log_1mq[None, :] - np.log1p(-p0))
        self.log = self.log + (w[None, :] * terms).sum(axis=1)
        self.max = np.maximum(self.max, self.log)

    def surviving(self, threshold: float) -> np.ndarray:
        """Grid nulls whose running maximum log-e stays below ``threshold``."""
        return self.grid[self.max < threshold]

    def interval(self, threshold: float) -> tuple[float, float] | None:
        """Hull of the surviving nulls; ``None`` when every null is rejected."""
        s = self.surviving(threshold)
        return (float(s.min()), float(s.max())) if s.size else None

    def set_state(self, log: object, max_: object) -> None:
        self.log = np.asarray(log, dtype=np.float64)
        self.max = np.asarray(max_, dtype=np.float64)


__all__ = [
    "ConfidenceSequence",
    "OffsetProcess",
    "plug_in_delta",
    "plug_in_shape",
]
