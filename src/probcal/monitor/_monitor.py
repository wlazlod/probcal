"""CalibrationMonitor: anytime-valid e-process monitoring (spec W7/W8).

Statistical design, validity conditions, and references:
``docs/concepts/monitoring.md``.
"""

import json
import os
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np

from .._math import expit, logit
from .._registry import register
from .._serialize import SCHEMA_VERSION, check_schema, fingerprint_of_dict
from .._validation import validate_scores, validate_weights
from ..metrics.regression import calibration_slope
from ._onset import estimate_onset
from ._processes import OffsetProcess, bern_log_lr, logsumexp, plug_in_delta, plug_in_shape

_RECOMMENDATION_WINDOWS = ("since_onset", "trailing")

_COMPONENTS = ("offset", "shape")


def _validate_outcomes(y: object) -> np.ndarray:
    """Binary outcomes; unlike fit-time validation, one-class batches are legal
    (a quiet month can mature with zero events)."""
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"y must be a 1-D array, got shape {arr.shape}")
    if not np.all((arr == 0.0) | (arr == 1.0)):
        raise ValueError("y must contain only values in {0, 1}")
    return arr


@dataclass(frozen=True)
class MonitorStep:
    """One matured batch's monitoring record (all e-values are running values).

    Attributes
    ----------
    label : str
        Caller-supplied batch label (opaque; arrival order is what counts).
    n : int
        Batch size.
    n_events : float
        Weighted event count of the batch.
    e_offset, e_shape : float
        Running component e-values after this batch (``nan`` for components
        not in ``components``).
    e_grades : dict[str, float]
        Running per-grade offset e-values (empty when no grades were given).
    e_global : float
        Running mean of the active components — the alarm statistic.
    p_anytime : float
        ``min(1, 1 / max_k E_k)`` — a p-value valid at every stopping time.
    alarm : bool
        Whether ``E`` has ever reached ``1/alpha`` (sticky).
    delta_ci : tuple[float, float] or None
        Time-uniform confidence sequence for the current offset (grid
        endpoints still surviving); ``None`` if every grid null is rejected.
    delta_hat, slope_hat : float
        The predictable plug-ins used for this batch (from past batches
        only) — recorded for auditability.
    grade_delta_ci : dict[str, tuple[float, float] | None]
        Per-grade time-uniform confidence sequence for that grade's own
        offset, same construction and grid as ``delta_ci`` (empty when no
        grades were given; absent for steps loaded from a pre-0.3 payload).
    log_e_increment : float
        This batch's additive plug-in log-LR increment: the offset
        plug-in's ``bern_log_lr`` contribution (0.0 when ``delta_hat ==
        0``) plus the shape plug-in's (0.0 when its plug-in is the
        identity). Unlike ``e_global`` — a logsumexp mixture, not additive
        across batches — this is the purely additive series
        :func:`~probcal.monitor._onset.estimate_onset` localizes drift
        onset from. Steps loaded from a pre-0.3 payload default to 0.0:
        onset is unavailable for those steps.
    """

    label: str
    n: int
    n_events: float
    e_offset: float
    e_shape: float
    e_grades: dict[str, float]
    e_global: float
    p_anytime: float
    alarm: bool
    delta_ci: tuple[float, float] | None
    delta_hat: float
    slope_hat: float
    grade_delta_ci: dict[str, tuple[float, float] | None] = field(default_factory=dict)
    log_e_increment: float = 0.0


@dataclass(frozen=True)
class MonitorReport:
    """Full monitoring trajectory with the diagnostic recommendation.

    Attributes
    ----------
    steps : tuple[MonitorStep, ...]
        Every processed batch, in arrival order.
    alarm_at : str or None
        Label of the first batch at which the alarm fired.
    recommendation : {"none", "re-offset", "re-fit"}
        Diagnostic (no error guarantee — the component e-values are the
        evidence; see the monitoring chapter).
    reasoning : tuple[str, ...]
        Plain-language trail behind the recommendation.
    alpha : float
        The monitor's alarm level (drawn as the 1/alpha line by
        ``probcal.plots.plot_e_process``).
    grade_table : dict[str, float]
        Latest per-grade e-values.
    onset_label : str or None
        Label of the batch :func:`~probcal.monitor._onset.estimate_onset`
        points to as the drift onset (backward-CUSUM argmax of
        ``MonitorStep.log_e_increment``); ``None`` unless ``alarm_at`` is
        also set. An estimate, not a test.
    """

    steps: tuple[MonitorStep, ...]
    alarm_at: str | None
    recommendation: str
    reasoning: tuple[str, ...]
    alpha: float = 0.05
    grade_table: dict[str, float] = field(default_factory=dict)
    onset_label: str | None = None

    def to_frame(self) -> object:
        """Steps as a list of dicts, or a pandas DataFrame when pandas is importable."""
        rows = [asdict(s) for s in self.steps]
        try:
            import pandas as pd
        except ImportError:
            return rows
        return pd.DataFrame(rows)


@register
class CalibrationMonitor:
    """Anytime-valid calibration monitoring by e-processes (spec W7/W8).

    Feed matured outcome batches in arrival order; the alarm rule
    "``E >= 1/alpha``" has type-I error at most ``alpha`` at every stopping
    time (Ville's inequality), however long monitoring runs. Persist the
    state between batches with :meth:`to_json` — never re-run or reorder
    past batches. Theory: ``docs/concepts/monitoring.md``.

    Parameters
    ----------
    alpha : float
        Alarm level in ``(0, 1)``.
    components : tuple of {"offset", "shape"}
        Which portfolio-level processes drive the global alarm (per-grade
        processes join automatically when ``grade`` arrays are passed).
    grades : tuple or None
        Optional explicit grade universe; ``None`` discovers grades from
        the ``grade`` arrays.
    mixture_grid : tuple of float
        Positive shifts for the offset mixture (symmetrized to ±).
    delta_ci_grid : tuple(lo, hi, count)
        Grid of offset nulls for the confidence sequence.
    min_history : int
        Number of past batches required before the plug-ins engage
        (before that they are the identity and their factors equal 1).
    plug_in_window : int or None
        Trailing number of past batches used by the plug-ins, and by the
        recommendation rule when ``recommendation_window="trailing"``;
        ``None`` uses all past batches.
    recommendation_window : {"since_onset", "trailing"}
        Which batches feed ``report()``'s trailing-window diagnostics
        (``delta_now``, the Cox slope CI, the residual-shape LR) once an
        alarm has fired. ``"since_onset"`` (the default) uses batches from
        the estimated drift onset (:func:`~probcal.monitor._onset.
        estimate_onset` on ``MonitorStep.log_e_increment`) onward — the
        rationale is that a window starting where the evidence trail
        actually turns is more informative than one anchored to
        ``plug_in_window``, which predates any alarm. ``"trailing"`` is
        the escape hatch restoring 0.2.0 behaviour exactly: it ignores the
        onset estimate and uses ``plug_in_window`` (or all past batches)
        instead, unconditionally.

    Attributes
    ----------
    steps_ : list[MonitorStep]
        The processed batches, in arrival order.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        components: tuple[str, ...] = ("offset", "shape"),
        grades: tuple | None = None,
        mixture_grid: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0),
        delta_ci_grid: tuple[float, float, int] = (-3.0, 3.0, 241),
        min_history: int = 1,
        plug_in_window: int | None = None,
        *,
        recommendation_window: str = "since_onset",
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
        unknown = [c for c in components if c not in _COMPONENTS]
        if unknown or not components:
            raise ValueError(
                f"components must be a non-empty subset of {_COMPONENTS}, got {components!r}"
            )
        if recommendation_window not in _RECOMMENDATION_WINDOWS:
            raise ValueError(
                "recommendation_window must be one of "
                f"{_RECOMMENDATION_WINDOWS}, got {recommendation_window!r}"
            )
        self.alpha = alpha
        self.components = tuple(components)
        self.grades = grades
        self.mixture_grid = tuple(mixture_grid)
        self.delta_ci_grid = tuple(delta_ci_grid)
        self.min_history = min_history
        self.plug_in_window = plug_in_window
        self.recommendation_window = recommendation_window
        self._init_state()

    # ------------------------------------------------------------------ state

    def _sym_grid(self) -> tuple[float, ...]:
        return tuple(sorted({s * d for d in self.mixture_grid for s in (-1.0, 1.0)}))

    def _init_state(self) -> None:
        self.steps_: list[MonitorStep] = []
        self._z: list[np.ndarray] = []
        self._y: list[np.ndarray] = []
        self._w: list[np.ndarray] = []
        self._g: list[np.ndarray | None] = []
        self._offset = OffsetProcess(self._sym_grid())
        self._log_shape = 0.0
        self._grade_procs: dict[str, OffsetProcess] = {}
        if self.grades is not None:
            for g in self.grades:
                self._grade_procs[str(g)] = OffsetProcess(self._sym_grid())
        lo, hi, count = self.delta_ci_grid
        self._cs_grid = np.linspace(float(lo), float(hi), int(count))
        self._cs_log = np.zeros(len(self._cs_grid))
        self._cs_max = np.zeros(len(self._cs_grid))
        self._grade_cs_log: dict[str, np.ndarray] = {}
        self._grade_cs_max: dict[str, np.ndarray] = {}
        self._max_log_global = -np.inf
        self._alarmed = False
        self._warned_weights = False

    def _past(self, grade: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        window = slice(None) if self.plug_in_window is None else slice(-self.plug_in_window, None)
        zs, ys, ws = self._z[window], self._y[window], self._w[window]
        gs = self._g[window]
        if grade is None:
            if not zs:
                return np.empty(0), np.empty(0), np.empty(0)
            return np.concatenate(zs), np.concatenate(ys), np.concatenate(ws)
        parts = [
            (z[g == grade], y[g == grade], w[g == grade])
            for z, y, w, g in zip(zs, ys, ws, gs, strict=True)
            if g is not None
        ]
        if not parts:
            return np.empty(0), np.empty(0), np.empty(0)
        return tuple(np.concatenate(a) for a in zip(*parts, strict=True))  # type: ignore[return-value]

    def _since(self, start: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Portfolio-level batches from index ``start`` onward (for the since-onset window)."""
        zs, ys, ws = self._z[start:], self._y[start:], self._w[start:]
        if not zs:
            return np.empty(0), np.empty(0), np.empty(0)
        return np.concatenate(zs), np.concatenate(ys), np.concatenate(ws)

    # ------------------------------------------------------------------ updates

    def update(
        self,
        y: object,
        p: object,
        sample_weight: object = None,
        grade: object = None,
        label: str | None = None,
    ) -> MonitorStep:
        """Process one matured batch (arrival order is the process order).

        Parameters
        ----------
        y : array_like
            Matured binary outcomes in ``{0, 1}`` (a one-class batch is
            legal — quiet months happen).
        p : array_like
            The probabilities the deployed forecast assigned to this batch.
        sample_weight : array_like or None
            Positive weights; non-uniform weights break the exact
            martingale property and warn once (reporting parity).
        grade : array_like or None
            Optional per-observation grade labels; activates the per-grade
            offset processes.
        label : str or None
            Batch label for reporting; defaults to ``batch-<k>``.

        Returns
        -------
        MonitorStep
            The running record after this batch.
        """
        y_arr = _validate_outcomes(y)
        p_arr = validate_scores(p, name="p")
        if len(y_arr) != len(p_arr):
            raise ValueError("y and p must have equal length")
        w_arr = validate_weights(sample_weight, len(p_arr))
        if not self._warned_weights and not np.all(w_arr == w_arr[0]):
            warnings.warn(
                "non-uniform sample weights break the exact martingale property; "
                "the e-values remain reported for parity but the type-I guarantee "
                "is approximate",
                UserWarning,
                stacklevel=2,
            )
            self._warned_weights = True
        g_arr: np.ndarray | None = None
        if grade is not None:
            g_arr = np.asarray(grade).astype(str)
            if len(g_arr) != len(p_arr):
                raise ValueError("grade and p must have equal length")
        z = logit(p_arr)

        # Predictable plug-ins: strictly past data only.
        history_ready = len(self._z) >= self.min_history
        if history_ready:
            pz, py, pw = self._past()
            delta_hat = plug_in_delta(pz, py, pw)
            c_hat, a_hat = plug_in_shape(pz, py, pw)
        else:
            delta_hat, (c_hat, a_hat) = 0.0, (0.0, 1.0)

        offset_inc = self._offset.update(z, p_arr, y_arr, w_arr, delta_hat)
        shape_inc = 0.0
        if (c_hat, a_hat) != (0.0, 1.0):  # identity plug-in: factor exactly 1
            shape_inc = bern_log_lr(y_arr, p_arr, expit(c_hat + a_hat * z), w_arr)
            self._log_shape += shape_inc
        log_e_increment = offset_inc + shape_inc

        if g_arr is not None:
            for g in np.unique(g_arr):
                if g not in self._grade_procs:
                    self._grade_procs[g] = OffsetProcess(self._sym_grid())
                if g not in self._grade_cs_log:
                    self._grade_cs_log[g] = np.zeros(len(self._cs_grid))
                    self._grade_cs_max[g] = np.zeros(len(self._cs_grid))
                mask = g_arr == g
                if history_ready:
                    gz, gy, gw = self._past(grade=g)
                    d_g = plug_in_delta(gz, gy, gw)
                else:
                    d_g = 0.0
                zg, pg, yg, wg = z[mask], p_arr[mask], y_arr[mask], w_arr[mask]
                self._grade_procs[g].update(zg, pg, yg, wg, d_g)

                # Per-grade confidence sequence: same construction as the
                # global one, restricted to this grade's own batch slice and
                # using its own plug-in as the alternative.
                gq_alt = np.clip(pg if d_g == 0.0 else expit(zg + d_g), 1e-12, 1.0 - 1e-12)
                glog_q = np.log(gq_alt)
                glog_1mq = np.log1p(-gq_alt)
                gp0 = np.clip(expit(zg[None, :] + self._cs_grid[:, None]), 1e-12, 1.0 - 1e-12)
                gterms = yg * (glog_q[None, :] - np.log(gp0)) + (1.0 - yg) * (
                    glog_1mq[None, :] - np.log1p(-gp0)
                )
                self._grade_cs_log[g] = self._grade_cs_log[g] + (wg[None, :] * gterms).sum(axis=1)
                self._grade_cs_max[g] = np.maximum(self._grade_cs_max[g], self._grade_cs_log[g])

        # Confidence sequence: e-process per shifted null, plug-in alternative.
        q_alt = np.clip(p_arr if delta_hat == 0.0 else expit(z + delta_hat), 1e-12, 1.0 - 1e-12)
        log_q = np.log(q_alt)
        log_1mq = np.log1p(-q_alt)
        p0 = np.clip(expit(z[None, :] + self._cs_grid[:, None]), 1e-12, 1.0 - 1e-12)
        terms = y_arr * (log_q[None, :] - np.log(p0)) + (1.0 - y_arr) * (
            log_1mq[None, :] - np.log1p(-p0)
        )
        self._cs_log += (w_arr[None, :] * terms).sum(axis=1)
        self._cs_max = np.maximum(self._cs_max, self._cs_log)

        # Store the batch AFTER the plug-ins consumed only the past.
        self._z.append(z)
        self._y.append(y_arr)
        self._w.append(w_arr)
        self._g.append(g_arr)

        step = self._make_step(label, y_arr, w_arr, delta_hat, a_hat, log_e_increment)
        self.steps_.append(step)
        return step

    def _make_step(
        self,
        label: str | None,
        y_arr: np.ndarray,
        w_arr: np.ndarray,
        delta_hat: float,
        a_hat: float,
        log_e_increment: float,
    ) -> MonitorStep:
        log_parts: list[float] = []
        e_offset = e_shape = float("nan")
        if "offset" in self.components:
            log_e_off = self._offset.log_e()
            e_offset = float(np.exp(log_e_off))
            log_parts.append(log_e_off)
        if "shape" in self.components:
            e_shape = float(np.exp(self._log_shape))
            log_parts.append(self._log_shape)
        e_grades = {g: float(np.exp(proc.log_e())) for g, proc in self._grade_procs.items()}
        if e_grades:
            grade_logs = np.array([proc.log_e() for proc in self._grade_procs.values()])
            log_parts.append(logsumexp(grade_logs) - np.log(len(grade_logs)))
        log_global = logsumexp(np.array(log_parts)) - np.log(len(log_parts))
        self._max_log_global = max(self._max_log_global, log_global)
        threshold = -np.log(self.alpha)
        if log_global >= threshold:
            self._alarmed = True
        surviving = self._cs_grid[self._cs_max < threshold]
        delta_ci = (float(surviving.min()), float(surviving.max())) if surviving.size else None
        grade_delta_ci: dict[str, tuple[float, float] | None] = {}
        for g, cs_max_g in self._grade_cs_max.items():
            surviving_g = self._cs_grid[cs_max_g < threshold]
            grade_delta_ci[g] = (
                (float(surviving_g.min()), float(surviving_g.max())) if surviving_g.size else None
            )
        return MonitorStep(
            label=label if label is not None else f"batch-{len(self.steps_)}",
            n=int(len(y_arr)),
            n_events=float(np.sum(w_arr * y_arr)),
            e_offset=e_offset,
            e_shape=e_shape,
            e_grades=e_grades,
            e_global=float(np.exp(log_global)),
            p_anytime=float(min(1.0, np.exp(-self._max_log_global))),
            alarm=self._alarmed,
            delta_ci=delta_ci,
            delta_hat=float(delta_hat),
            slope_hat=float(a_hat),
            grade_delta_ci=grade_delta_ci,
            log_e_increment=float(log_e_increment),
        )

    # ------------------------------------------------------------------ report

    def report(self) -> MonitorReport:
        """Trajectory plus the diagnostic re-offset/re-fit recommendation."""
        alarm_at = next((s.label for s in self.steps_ if s.alarm), None)
        grade_table = dict(self.steps_[-1].e_grades) if self.steps_ else {}
        if alarm_at is None:
            return MonitorReport(
                steps=tuple(self.steps_),
                alarm_at=None,
                recommendation="none",
                reasoning=("no alarm: the global e-process never reached 1/alpha",),
                alpha=self.alpha,
                grade_table=grade_table,
            )
        increments = np.array([s.log_e_increment for s in self.steps_])
        onset_idx = estimate_onset(increments)
        onset_label = self.steps_[onset_idx].label
        if self.recommendation_window == "since_onset":
            pz, py, pw = self._since(onset_idx)
        else:
            pz, py, pw = self._past()
        delta_now = plug_in_delta(pz, py, pw)
        e_shape = self.steps_[-1].e_shape
        lo, hi = self._slope_ci(pz, py, pw)
        slope_ok = lo <= 1.0 <= hi
        # Residual-shape check: does the 2-parameter Cox correction explain the
        # trailing window materially better than the offset-only correction?
        # (The shape e-process itself also fires under pure level drift — its
        # alternative family contains the intercept — so it cannot separate
        # the two failure modes on its own; DECISIONS 75.)
        resid_lr = self._residual_shape_lr(pz, py, pw, delta_now)
        shape_needed = resid_lr > 3.841  # chi-square(1) at 5%
        reasoning = [
            f"alarm at {alarm_at!r}; trailing-window offset {delta_now:+.3f} log-odds",
            f"shape e-process {e_shape:.3g} vs 1/alpha = {1.0 / self.alpha:.1f} "
            "(reported; fires under level drift too, so not decisive alone)",
            f"trailing-window Cox slope 95% bootstrap CI [{lo:.3f}, {hi:.3f}] "
            + ("contains" if slope_ok else "excludes")
            + " 1",
            f"Cox-vs-offset residual LR on the trailing window {resid_lr:.2f} "
            + ("exceeds" if shape_needed else "is within")
            + " the chi-square(1) 5% bound 3.84",
            f"estimated drift onset at {onset_label} (backward-CUSUM argmax of the plug-in "
            "log-LR increments — an estimate, not a test)",
            "the recommendation is a diagnostic, not a test — see the monitoring chapter",
        ]
        recommendation = "re-offset" if (slope_ok and not shape_needed) else "re-fit"
        return MonitorReport(
            steps=tuple(self.steps_),
            alarm_at=alarm_at,
            recommendation=recommendation,
            reasoning=tuple(reasoning),
            alpha=self.alpha,
            grade_table=grade_table,
            onset_label=onset_label,
        )

    @staticmethod
    def _residual_shape_lr(z: np.ndarray, y: np.ndarray, w: np.ndarray, delta: float) -> float:
        """2*(loglik of the Cox fit - loglik of the offset-only fit) on the window."""
        if z.size == 0 or np.unique(y).size < 2:
            return 0.0
        c, a = plug_in_shape(z, y, w)

        def ll(q: np.ndarray) -> float:
            qc = np.clip(q, 1e-12, 1.0 - 1e-12)
            return float(np.sum(w * (y * np.log(qc) + (1.0 - y) * np.log1p(-qc))))

        return max(0.0, 2.0 * (ll(expit(c + a * z)) - ll(expit(z + delta))))

    def _slope_ci(
        self, z: np.ndarray, y: np.ndarray, w: np.ndarray, n_boot: int = 200
    ) -> tuple[float, float]:
        """Percentile bootstrap CI of the Cox slope on the trailing window (seeded)."""
        if z.size < 10 or np.unique(y).size < 2:
            return -np.inf, np.inf
        rng = np.random.default_rng(0)
        p = expit(z)
        slopes = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(z), len(z))
            if np.unique(y[idx]).size < 2:
                continue
            slopes.append(calibration_slope(y[idx], p[idx], sample_weight=w[idx]))
        if len(slopes) < 20:
            return -np.inf, np.inf
        return float(np.quantile(slopes, 0.025)), float(np.quantile(slopes, 0.975))

    # ------------------------------------------------------------------ serialization

    def to_dict(self) -> dict[str, object]:
        """Versioned snapshot; the state includes every past batch — that is
        what makes each decision reproducible (spec invariant)."""
        from .. import __version__

        return {
            "probcal_schema": SCHEMA_VERSION,
            "probcal_version": __version__,
            "class": type(self).__name__,
            "params": {
                "alpha": self.alpha,
                "components": list(self.components),
                "grades": list(self.grades) if self.grades is not None else None,
                "mixture_grid": list(self.mixture_grid),
                "delta_ci_grid": list(self.delta_ci_grid),
                "min_history": self.min_history,
                "plug_in_window": self.plug_in_window,
                "recommendation_window": self.recommendation_window,
            },
            "state": {
                "z": [a.tolist() for a in self._z],
                "y": [a.tolist() for a in self._y],
                "w": [a.tolist() for a in self._w],
                "g": [a.tolist() if a is not None else None for a in self._g],
                "offset": self._offset.state(),
                "log_shape": self._log_shape,
                "grade_procs": {g: p.state() for g, p in self._grade_procs.items()},
                "cs_log": self._cs_log.tolist(),
                "cs_max": self._cs_max.tolist(),
                "grade_cs_log": {g: a.tolist() for g, a in self._grade_cs_log.items()},
                "grade_cs_max": {g: a.tolist() for g, a in self._grade_cs_max.items()},
                "max_log_global": float(self._max_log_global),
                "alarmed": self._alarmed,
                "warned_weights": self._warned_weights,
                "steps": [self._step_to_dict(s) for s in self.steps_],
            },
            "fit_meta": {
                "n_batches": len(self._z),
                "n_obs": int(sum(len(a) for a in self._z)),
            },
        }

    @staticmethod
    def _step_to_dict(s: MonitorStep) -> dict[str, object]:
        d = asdict(s)
        d["delta_ci"] = list(s.delta_ci) if s.delta_ci is not None else None
        d["grade_delta_ci"] = {
            g: (list(v) if v is not None else None) for g, v in s.grade_delta_ci.items()
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationMonitor":
        """Rebuild a monitor mid-stream; the trajectory continues bit-for-bit.

        Raises
        ------
        ValueError
            If the schema version is unknown or the payload class differs.
        """
        check_schema(d)
        if d.get("class") != cls.__name__:
            raise ValueError(f"payload was written by {d.get('class')!r}, not {cls.__name__}")
        params = dict(d["params"])
        params["components"] = tuple(params["components"])
        params["grades"] = tuple(params["grades"]) if params["grades"] is not None else None
        params["mixture_grid"] = tuple(params["mixture_grid"])
        params["delta_ci_grid"] = tuple(params["delta_ci_grid"])
        params["recommendation_window"] = params.get("recommendation_window", "since_onset")
        mon = cls(**params)
        st = d["state"]
        mon._z = [np.asarray(a, dtype=np.float64) for a in st["z"]]
        mon._y = [np.asarray(a, dtype=np.float64) for a in st["y"]]
        mon._w = [np.asarray(a, dtype=np.float64) for a in st["w"]]
        mon._g = [np.asarray(a).astype(str) if a is not None else None for a in st["g"]]
        mon._offset.set_state(st["offset"])
        mon._log_shape = float(st["log_shape"])
        mon._grade_procs = {}
        for g, ps in st["grade_procs"].items():
            proc = OffsetProcess(mon._sym_grid())
            proc.set_state(ps)
            mon._grade_procs[g] = proc
        mon._cs_log = np.asarray(st["cs_log"], dtype=np.float64)
        mon._cs_max = np.asarray(st["cs_max"], dtype=np.float64)
        mon._grade_cs_log = {
            g: np.asarray(a, dtype=np.float64) for g, a in st.get("grade_cs_log", {}).items()
        }
        mon._grade_cs_max = {
            g: np.asarray(a, dtype=np.float64) for g, a in st.get("grade_cs_max", {}).items()
        }
        mon._max_log_global = float(st["max_log_global"])
        mon._alarmed = bool(st["alarmed"])
        mon._warned_weights = bool(st["warned_weights"])
        mon.steps_ = []
        for sd in st["steps"]:
            sd = dict(sd)
            sd["delta_ci"] = tuple(sd["delta_ci"]) if sd["delta_ci"] is not None else None
            sd["e_grades"] = dict(sd["e_grades"])
            sd["grade_delta_ci"] = {
                g: (tuple(v) if v is not None else None)
                for g, v in sd.get("grade_delta_ci", {}).items()
            }
            mon.steps_.append(MonitorStep(**sd))
        return mon

    def to_json(
        self, path: "str | os.PathLike[str] | None" = None, *, indent: int = 2
    ) -> str | None:
        """Serialize to JSON text, or to ``path`` when given (returns None then)."""
        text = json.dumps(self.to_dict(), indent=indent)
        if path is None:
            return text
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return None

    @classmethod
    def from_json(cls, path_or_str: object) -> "CalibrationMonitor":
        """Load from a JSON string or a filesystem path."""
        text = str(path_or_str)
        if not text.lstrip().startswith("{"):
            with open(text, encoding="utf-8") as fh:
                text = fh.read()
        return cls.from_dict(json.loads(text))

    def fingerprint(self) -> str:
        """SHA-256 of the canonical serialized state (version-blind)."""
        return fingerprint_of_dict(self.to_dict())
