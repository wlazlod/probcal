"""Log-space Bernoulli likelihood-ratio arithmetic and predictable plug-ins.

Every e-process in the monitor multiplies factors
``LR_i(q) = q^y (1-q)^(1-y) / (p^y (1-p)^(1-y))`` whose conditional
expectation under the conditional-calibration null is exactly 1 for any
*predictable* alternative ``q`` — see ``docs/concepts/monitoring.md``.
Accumulation is in log space throughout; mixtures combine with logsumexp.
"""

import numpy as np

from .._math import expit, irls_logistic, logit
from ..offset import _offset_mle

_CLIP = 1e-12


def bern_log_lr(y: np.ndarray, p_null: np.ndarray, q_alt: np.ndarray, w: np.ndarray) -> float:
    """Weighted log Bernoulli likelihood ratio of ``q_alt`` against ``p_null``."""
    q = np.clip(q_alt, _CLIP, 1.0 - _CLIP)
    p = np.clip(p_null, _CLIP, 1.0 - _CLIP)
    terms = y * (np.log(q) - np.log(p)) + (1.0 - y) * (np.log1p(-q) - np.log1p(-p))
    return float(np.sum(w * terms))


def logsumexp(values: np.ndarray) -> float:
    """Overflow-safe ``log(sum(exp(values)))``."""
    m = float(np.max(values))
    return m + float(np.log(np.sum(np.exp(values - m))))


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
    return _offset_mle(z, y, w)


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
    ) -> None:
        # delta_hat == 0 means the alternative IS the null: the factor is
        # exactly 1, no expit(logit(p)) round-trip noise.
        if delta_hat != 0.0:
            self.log_plug += bern_log_lr(y, p, expit(z + delta_hat), w)
        for j, d in enumerate(self.grid):
            self.log_mix[j] += bern_log_lr(y, p, expit(z + d), w)

    def log_e(self) -> float:
        """log of ``(E_plug + E_mix) / 2`` (averages of e-values are e-values)."""
        log_e_mix = logsumexp(self.log_mix) - np.log(len(self.grid))
        return float(np.logaddexp(self.log_plug, log_e_mix) - np.log(2.0))

    def state(self) -> dict[str, object]:
        return {"log_plug": self.log_plug, "log_mix": self.log_mix.tolist()}

    def set_state(self, state: dict[str, object]) -> None:
        self.log_plug = float(state["log_plug"])  # type: ignore[arg-type]
        self.log_mix = np.asarray(state["log_mix"], dtype=np.float64)


__all__ = [
    "OffsetProcess",
    "bern_log_lr",
    "logit",
    "logsumexp",
    "plug_in_delta",
    "plug_in_shape",
]
