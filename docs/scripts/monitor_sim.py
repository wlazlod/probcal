"""Verification simulations for probcal.monitor.

Usage: ``uv run python docs/scripts/monitor_sim.py [--fast]``

Vectorized across runs: every run shares the monthly score vector, so the
pooled-past plug-ins reduce to per-score event counts and the whole fleet
advances one batch per matrix operation. ``tests/test_monitor_sim.py``
cross-checks this replay against the shipped ``CalibrationMonitor`` class
(same e-process trajectory) and enforces reduced-size versions of the gates
below; the full-size table printed here is pasted into
``docs/concepts/monitoring.md``.

Gates (full sizes):
- type I: alarm fraction <= alpha + 2*sqrt(alpha*(1-alpha)/runs) for
  alpha in {0.05, 0.01}, per component and global, 2000 runs x 24 monthly
  batches x n=2000 (event rate 5%), plus per-grade and heterogeneous-size
  variants;
- power: delta=0.4 detected within 6 batches of onset (median), slope 0.8
  within 12; recommendation correct in >= 90% of pure-offset / pure-slope
  runs (via the real CalibrationMonitor);
- CS: the true offset is covered at every step in >= 1-alpha of runs.
"""

import sys

import numpy as np

from probcal._math import expit, logit
from probcal.datasets import make_pd_portfolio

GRID = (-1.0, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0)  # symmetrized default


def _scores(n: int, seed: int = 42) -> np.ndarray:
    return make_pd_portfolio(n=n, event_rate=0.05, random_state=seed).scores


class FleetReplay:
    """All runs at once: the monitor's offset/shape/global math on shared scores."""

    def __init__(self, runs: int, z: np.ndarray, alpha: float = 0.05) -> None:
        self.runs = runs
        self.z = z
        self.p = expit(z)
        self.alpha = alpha
        n = len(z)
        self.k = 0
        self.cum_events = np.zeros((runs, n))  # per-score event counts over past
        self.log_plug = np.zeros(runs)
        self.log_mix = np.zeros((runs, len(GRID)))
        self.log_shape = np.zeros(runs)
        self.max_log = {
            "offset": np.full(runs, -np.inf),
            "shape": np.full(runs, -np.inf),
            "global": np.full(runs, -np.inf),
        }
        # Tabulated M(d) = mean sigma(z + d) for the delta plug-in inversion.
        self._dgrid = np.linspace(-5.0, 5.0, 4001)
        self._m_of_d = expit(z[None, :] + self._dgrid[:, None]).mean(axis=1)

    def _delta_hat(self) -> np.ndarray:
        if self.k == 0:
            return np.zeros(self.runs)
        ybar = self.cum_events.mean(axis=1) / self.k
        return np.interp(ybar, self._m_of_d, self._dgrid)

    def _shape_hat(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-run (c, a) by 2-parameter Newton on the aggregated counts."""
        if self.k == 0:
            return np.zeros(self.runs), np.ones(self.runs)
        z, k = self.z, float(self.k)
        s = self.cum_events  # successes per score point, k trials each
        c = np.zeros(self.runs)
        a = np.ones(self.runs)
        for _ in range(25):
            eta = c[:, None] + a[:, None] * z[None, :]
            mu = expit(eta)
            wgt = k * mu * (1.0 - mu) + 1e-12
            r0 = (s - k * mu).sum(axis=1)
            r1 = ((s - k * mu) * z[None, :]).sum(axis=1)
            h00 = wgt.sum(axis=1)
            h01 = (wgt * z[None, :]).sum(axis=1)
            h11 = (wgt * z[None, :] ** 2).sum(axis=1)
            det = h00 * h11 - h01**2 + 1e-12
            dc = (h11 * r0 - h01 * r1) / det
            da = (h00 * r1 - h01 * r0) / det
            step = np.clip(np.maximum(np.abs(dc), np.abs(da)), None, 4.0)
            scale = np.where(step > 2.0, 2.0 / np.maximum(step, 1e-12), 1.0)
            c += scale * dc
            a += scale * da
            if float(np.max(np.abs(dc)) + np.max(np.abs(da))) < 1e-10:
                break
        bad = ~(np.isfinite(c) & np.isfinite(a))
        c[bad], a[bad] = 0.0, 1.0
        return c, a

    def update(self, y: np.ndarray) -> dict[str, np.ndarray]:
        """Advance every run one batch; y has shape (runs, n)."""
        z, p = self.z, self.p
        delta = self._delta_hat()
        c, a = self._shape_hat()

        log_p, log_1mp = np.log(p), np.log1p(-p)
        # mixture: shared coefficients per grid point
        for j, d in enumerate(GRID):
            q = expit(z + d)
            aa = np.log(q) - log_p
            bb = np.log1p(-q) - log_1mp
            self.log_mix[:, j] += y @ (aa - bb) + bb.sum()
        # plug-in: per-run delta
        engaged = delta != 0.0
        if np.any(engaged):
            q = expit(z[None, :] + delta[engaged, None])
            aa = np.log(q) - log_p[None, :]
            bb = np.log1p(-q) - log_1mp[None, :]
            self.log_plug[engaged] += (y[engaged] * (aa - bb)).sum(axis=1) + bb.sum(axis=1)
        # shape: per-run (c, a)
        engaged = ~((c == 0.0) & (a == 1.0))
        if np.any(engaged):
            q = expit(c[engaged, None] + a[engaged, None] * z[None, :])
            aa = np.log(q) - log_p[None, :]
            bb = np.log1p(-q) - log_1mp[None, :]
            self.log_shape[engaged] += (y[engaged] * (aa - bb)).sum(axis=1) + bb.sum(axis=1)

        self.cum_events += y
        self.k += 1

        log_e_mix = _logsumexp_rows(self.log_mix) - np.log(len(GRID))
        log_e_off = np.logaddexp(self.log_plug, log_e_mix) - np.log(2.0)
        log_e_glob = np.logaddexp(log_e_off, self.log_shape) - np.log(2.0)
        self.max_log["offset"] = np.maximum(self.max_log["offset"], log_e_off)
        self.max_log["shape"] = np.maximum(self.max_log["shape"], self.log_shape)
        self.max_log["global"] = np.maximum(self.max_log["global"], log_e_glob)
        return {
            "offset": log_e_off,
            "shape": self.log_shape.copy(),
            "global": log_e_glob,
            "delta_hat": delta,
        }

    def alarm_fraction(self, alpha: float) -> dict[str, float]:
        thr = -np.log(alpha)
        return {name: float(np.mean(mx >= thr)) for name, mx in self.max_log.items()}


def _logsumexp_rows(m: np.ndarray) -> np.ndarray:
    mx = m.max(axis=1)
    return mx + np.log(np.exp(m - mx[:, None]).sum(axis=1))


def draw_outcomes(rng, p_true: np.ndarray, runs: int) -> np.ndarray:
    return (rng.random((runs, len(p_true))) < p_true[None, :]).astype(np.float64)


def sim_type1(runs: int = 2000, batches: int = 24, n: int = 2000, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    z = logit(_scores(n))
    fleet = FleetReplay(runs, z)
    for _ in range(batches):
        fleet.update(draw_outcomes(rng, expit(z), runs))
    return {a: fleet.alarm_fraction(a) for a in (0.05, 0.01)}


def sim_type1_hetero(runs: int = 2000, batches: int = 24, seed: int = 2) -> dict:
    """Heterogeneous batch sizes: alternate n=1000 / n=3000 slices of one pool."""
    rng = np.random.default_rng(seed)
    z_small, z_big = logit(_scores(1000, seed=7)), logit(_scores(3000, seed=8))
    # Two interleaved fleets sharing accumulators is equivalent to per-batch
    # varying scores; replay by alternating score vectors on one fleet is not
    # possible with the aggregation trick, so run the exact per-batch math on
    # the smaller run count implied by memory: use pairs of fleets whose logs
    # add (independent factors on disjoint batches).
    fa = FleetReplay(runs, z_small)
    fb = FleetReplay(runs, z_big)
    for k in range(batches):
        if k % 2 == 0:
            fa.update(draw_outcomes(rng, expit(z_small), runs))
        else:
            fb.update(draw_outcomes(rng, expit(z_big), runs))
    out = {}
    for a in (0.05, 0.01):
        thr = -np.log(a)
        combined = {}
        for name in ("offset", "shape", "global"):
            # conservative combination: product of the two independent
            # e-processes is an e-process on the interleaved stream
            mx = fa.max_log[name] + fb.max_log[name]
            combined[name] = float(np.mean(mx >= thr))
        out[a] = combined
    return out


def sim_type1_grades(runs: int = 2000, batches: int = 24, n: int = 2000, seed: int = 3) -> dict:
    """Two grades of n/2 each; the grade component is the mean of two offset fleets."""
    rng = np.random.default_rng(seed)
    z1, z2 = logit(_scores(n // 2, seed=11)), logit(_scores(n // 2, seed=12))
    f1, f2 = FleetReplay(runs, z1), FleetReplay(runs, z2)
    max_log = np.full(runs, -np.inf)
    for _ in range(batches):
        o1 = f1.update(draw_outcomes(rng, expit(z1), runs))
        o2 = f2.update(draw_outcomes(rng, expit(z2), runs))
        # grade component: mean of the two per-grade offset e-values
        log_grades = np.logaddexp(o1["offset"], o2["offset"]) - np.log(2.0)
        # portfolio offset/shape processes: product over the two halves
        # Product of the two half-portfolio offset processes: a valid
        # e-process on the combined stream (independent factors), which is
        # all the type-I gate requires.
        log_off = o1["offset"] + o2["offset"]
        log_shape = o1["shape"] + o2["shape"]
        parts = np.stack([log_off, log_shape, log_grades])
        log_glob = _logsumexp_rows(parts.T) - np.log(3)
        max_log = np.maximum(max_log, log_glob)
    return {a: {"global": float(np.mean(max_log >= -np.log(a)))} for a in (0.05, 0.01)}


def sim_power(
    shift: float = 0.0,
    slope: float = 1.0,
    runs: int = 500,
    batches: int = 24,
    n: int = 2000,
    onset: int = 12,
    alpha: float = 0.05,
    seed: int = 4,
) -> dict:
    rng = np.random.default_rng(seed)
    z = logit(_scores(n))
    fleet = FleetReplay(runs, z, alpha=alpha)
    thr = -np.log(alpha)
    first_alarm = np.full(runs, np.inf)
    for k in range(batches):
        drifted = k >= onset
        p_true = expit(slope * z + shift) if drifted else expit(z)
        out = fleet.update(draw_outcomes(rng, p_true, runs))
        mask = (out["global"] >= thr) & np.isinf(first_alarm)
        first_alarm[mask] = k
    delay = first_alarm - onset
    detected = np.isfinite(delay) & (delay >= 0)
    return {
        "detect_rate": float(np.mean(detected)),
        "median_delay": float(np.median(delay[detected])) if detected.any() else float("nan"),
    }


def sim_cs_coverage(
    runs: int = 500,
    batches: int = 24,
    n: int = 2000,
    true_delta: float = 0.0,
    alpha: float = 0.05,
    seed: int = 5,
) -> float:
    """Fraction of runs whose CS covers true_delta at EVERY step (time-uniform)."""
    rng = np.random.default_rng(seed)
    z = logit(_scores(n))
    p_true = expit(z + true_delta)
    grid = np.linspace(-3.0, 3.0, 241)
    fleet = FleetReplay(runs, z, alpha=alpha)
    cs_log = np.zeros((runs, len(grid)))
    cs_max = np.zeros((runs, len(grid)))
    covered = np.ones(runs, dtype=bool)
    thr = -np.log(alpha)
    j_true = int(np.argmin(np.abs(grid - true_delta)))
    log_p0 = np.log(np.clip(expit(z[None, :] + grid[:, None]), 1e-12, 1 - 1e-12))
    log_1mp0 = np.log1p(-np.clip(expit(z[None, :] + grid[:, None]), 1e-12, 1 - 1e-12))
    sum_log_1mp0 = log_1mp0.sum(axis=1)
    for _ in range(batches):
        y = draw_outcomes(rng, p_true, runs)
        delta = fleet._delta_hat()
        q = np.clip(expit(z[None, :] + delta[:, None]), 1e-12, 1 - 1e-12)
        alt = (y * np.log(q) + (1 - y) * np.log1p(-q)).sum(axis=1)
        null = y @ (log_p0 - log_1mp0).T + sum_log_1mp0[None, :]
        cs_log += alt[:, None] - null
        cs_max = np.maximum(cs_max, cs_log)
        fleet.cum_events += y
        fleet.k += 1
        covered &= cs_max[:, j_true] < thr
    return float(np.mean(covered))


def main() -> None:
    fast = "--fast" in sys.argv
    runs = 300 if fast else 2000
    pruns = 120 if fast else 500
    rows = []
    t1 = sim_type1(runs=runs)
    for a in (0.05, 0.01):
        bound = a + 2 * np.sqrt(a * (1 - a) / runs)
        for comp in ("offset", "shape", "global"):
            rows.append((f"type-I {comp} (alpha={a})", f"{t1[a][comp]:.4f}", f"<= {bound:.4f}"))
    tg = sim_type1_grades(runs=runs)
    th = sim_type1_hetero(runs=runs)
    for a in (0.05, 0.01):
        bound = a + 2 * np.sqrt(a * (1 - a) / runs)
        rows.append(
            (f"type-I global, per-grade (alpha={a})", f"{tg[a]['global']:.4f}", f"<= {bound:.4f}")
        )
        rows.append(
            (
                f"type-I global, hetero sizes (alpha={a})",
                f"{th[a]['global']:.4f}",
                f"<= {bound:.4f}",
            )
        )
    for shift, slope, gate, name in (
        (0.2, 1.0, None, "power delta=0.2"),
        (0.4, 1.0, 6, "power delta=0.4"),
        (0.0, 0.8, 12, "power slope=0.8"),
        (0.0, 1.25, None, "power slope=1.25"),
    ):
        r = sim_power(shift=shift, slope=slope, runs=pruns)
        gate_txt = f"median delay <= {gate}" if gate else "reported"
        rows.append(
            (name, f"detect {r['detect_rate']:.2f}, median delay {r['median_delay']:.1f}", gate_txt)
        )
    cov = sim_cs_coverage(runs=pruns)
    rows.append(("CS time-uniform coverage (delta=0)", f"{cov:.4f}", ">= 0.95"))
    cov2 = sim_cs_coverage(runs=pruns, true_delta=0.4, seed=6)
    rows.append(("CS time-uniform coverage (delta=0.4)", f"{cov2:.4f}", ">= 0.95"))

    width = max(len(r[0]) for r in rows)
    print(f"| {'experiment'.ljust(width)} | result | gate |")
    print(f"|{'-' * (width + 2)}|--------|------|")
    for name, res, gate in rows:
        print(f"| {name.ljust(width)} | {res} | {gate} |")


if __name__ == "__main__":
    main()
