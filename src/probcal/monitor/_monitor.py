"""CalibrationMonitor: anytime-valid e-process monitoring.

Statistical design, validity conditions, and references:
``docs/concepts/monitoring.md``.
"""

import re
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .._math import bern_log_lr, expit, logit, logsumexp
from .._registry import register
from .._serialize import JsonIO, check_payload, decode_value, encode_value, envelope
from .._validation import validate_positive_int, validate_scores, validate_weights
from ._diagnostics import recommend
from ._onset import estimate_onset
from ._processes import ConfidenceSequence, OffsetProcess, plug_in_delta, plug_in_shape

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


def grade_key(v: object) -> str:
    """Canonical string key for one grade label.

    Integers and integer-valued floats share a key (``1``, ``1.0``,
    ``np.int64(1)`` all map to ``"1"``), other floats use their shortest
    repr, and everything else is ``str(v)`` as a plain ``str`` (never
    ``numpy.str_``). Strings are opaque labels: ``"1.0"`` stays ``"1.0"``.
    """
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return str(int(f)) if f.is_integer() else repr(f)
    return str(v)


def grade_keys(grade: object) -> np.ndarray:
    """Vectorized :func:`grade_key` over a 1-D label array (unicode array out)."""
    arr = np.asarray(grade)
    if arr.ndim != 1:
        raise ValueError(f"grade must be a 1-D array, got shape {arr.shape}")
    if arr.dtype.kind in "USb":
        return arr.astype(str)
    if arr.dtype.kind in "iu":
        return arr.astype(np.int64).astype(str)
    if arr.dtype.kind == "f":
        out = arr.astype(str)
        whole = np.isfinite(arr) & (arr == np.round(arr))
        out[whole] = arr[whole].astype(np.int64).astype(str)
        return out
    return np.array([grade_key(x) for x in arr.tolist()], dtype=str)


_LEGACY_FLOAT_KEY = re.compile(r"^-?\d+\.0$")


def _version_tuple(v: object) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", str(v))[:3])


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
    log_e_increment : float or None
        This batch's additive plug-in log-LR increment: the sum of the
        plug-in ``bern_log_lr`` contributions of the components in the
        alarm — offset and shape when monitored (0.0 when a plug-in is the
        identity), plus the declared grades' plug-ins. Unlike ``e_global``
        — a logsumexp mixture, not additive across batches — this is the purely additive series
        :func:`~probcal.monitor._onset.estimate_onset` localizes drift
        onset from. Steps written by :meth:`CalibrationMonitor.update` always
        carry a float; steps loaded from a pre-0.3 payload carry ``None``
        (that payload records no increments), and a monitor holding any
        such step reports no onset at all.
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
    log_e_increment: float | None = None


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
        also set, and ``None`` as well when any step carries no increment
        (a pre-0.3 payload). An estimate, not a test.
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
class CalibrationMonitor(JsonIO):
    """Anytime-valid calibration monitoring by e-processes.

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
        Which portfolio-level processes drive the global alarm.
    grades : tuple or None
        The grade universe, fixed for the monitor's lifetime. When given,
        the per-grade component (the equal-weight average of one offset
        e-process per declared grade) is part of the global mixture from
        the first batch on — a grade without data yet contributes ``e = 1``
        — and a ``grade`` array carrying a label outside this universe
        raises ``ValueError``. When ``None``, per-grade e-processes and
        confidence sequences are still tracked and reported for every label
        seen, but they are **not** part of the global mixture (a warning
        says so the first time a ``grade`` array is passed). Labels are
        compared through one normalization: ``1``, ``1.0`` and
        ``np.int64(1)`` are the same grade ``"1"``.
    mixture_grid : tuple of float
        Positive shifts for the offset mixture (symmetrized to ±).
    delta_ci_grid : tuple(lo, hi, count)
        Grid of offset nulls for the confidence sequence.
    min_history : int
        Number of past batches (``>= 0``) required before the plug-ins
        engage (before that they are the identity and their factors equal 1).
    plug_in_window : int or None
        Trailing number of past batches (``>= 1``) used by the plug-ins, and
        by the recommendation rule when ``recommendation_window="trailing"``;
        ``None`` uses all past batches. Every batch refits the plug-ins on
        the window, so with ``None`` the per-batch cost grows linearly with
        the history and the total cost of ``k`` batches quadratically —
        set a window for long-running monitors.
    recommendation_window : {"since_onset", "trailing"}
        Which batches feed ``report()``'s trailing-window diagnostics
        (``delta_now``, the Cox slope CI, the residual-shape LR) once an
        alarm has fired. ``"since_onset"`` (the default) uses batches from
        the estimated drift onset (:func:`~probcal.monitor._onset.
        estimate_onset` on ``MonitorStep.log_e_increment``) onward — the
        rationale is that a window starting where the evidence trail
        actually turns is more informative than one anchored to
        ``plug_in_window``, which predates any alarm. When ``plug_in_window``
        is also set, the window starts at the LATER of the two starts
        (``max(onset_idx, n_batches - plug_in_window)``), so a short
        ``plug_in_window`` still bounds how far back the since-onset window
        can reach. ``"trailing"`` is the escape hatch restoring 0.2.0
        behaviour exactly for those diagnostic INPUTS: it ignores the onset
        estimate and uses ``plug_in_window`` (or all past batches) instead,
        unconditionally. ``onset_label`` and the onset sentence in
        ``reasoning`` are populated under both modes — only the
        diagnostic window differs. When onset is unavailable (any step
        loaded from a pre-0.3 payload carries no increment), both modes
        fall back to ``"trailing"`` and ``onset_label`` is ``None``.

    Attributes
    ----------
    steps_ : list[MonitorStep]
        The processed batches, in arrival order.
    masterscale_fingerprint_ : str or None
        Fingerprint of the ``Masterscale`` passed to :meth:`update`, once one
        has been; serialized with the state.

    Notes
    -----
    Anytime validity needs the global mixture's components and weights to
    be fixed in advance. Before 0.3.4 a grade first seen mid-stream joined
    the average with ``e = 1``, re-weighting the existing components — the
    global value could then jump toward 1 without any evidence. Declaring
    ``grades`` fixes the universe up front; without it the per-grade
    processes are diagnostics only.
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
        if (
            isinstance(min_history, bool)
            or not isinstance(min_history, (int, np.integer))
            or min_history < 0
        ):
            raise ValueError(f"min_history must be an integer >= 0, got {min_history!r}")
        if plug_in_window is not None:
            validate_positive_int(plug_in_window, "plug_in_window")
        if grades is not None:
            keys = [grade_key(g) for g in grades]
            if not keys:
                raise ValueError("grades must be None or a non-empty tuple of labels")
            if len(set(keys)) != len(keys):
                raise ValueError(
                    f"grades contains duplicate labels after normalization: {grades!r}"
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

    @property
    def _grade_universe(self) -> tuple[str, ...] | None:
        """Normalized declared grade keys, or ``None`` when undeclared."""
        if self.grades is None:
            return None
        return tuple(grade_key(g) for g in self.grades)

    def _init_state(self) -> None:
        self.steps_: list[MonitorStep] = []
        self._z: list[np.ndarray] = []
        self._y: list[np.ndarray] = []
        self._w: list[np.ndarray] = []
        self._g: list[np.ndarray | None] = []
        self._offset = OffsetProcess(self._sym_grid())
        self._log_shape = 0.0
        self._grade_procs: dict[str, OffsetProcess] = {}
        for g in self._grade_universe or ():
            self._grade_procs[g] = OffsetProcess(self._sym_grid())
        lo, hi, count = self.delta_ci_grid
        self._cs = ConfidenceSequence(np.linspace(float(lo), float(hi), int(count)))
        self._grade_cs: dict[str, ConfidenceSequence] = {}
        self._max_log_global = -np.inf
        self._alarmed = False
        self._warned_weights = False
        self._warned_grades = False
        self.masterscale_fingerprint_: str | None = None

    @property
    def _cs_grid(self) -> np.ndarray:
        """Offset nulls of the portfolio-level confidence sequence."""
        return self._cs.grid

    @property
    def _cs_max(self) -> np.ndarray:
        """Running maximum log-e per null of the portfolio-level CS."""
        return self._cs.max

    def _trailing_start(self) -> int:
        """First batch index of the ``plug_in_window`` trailing window."""
        if self.plug_in_window is None:
            return 0
        return max(0, len(self._z) - self.plug_in_window)

    def _since(
        self, start: int, grade: str | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(z, y, w)`` of the batches from index ``start`` onward.

        With ``grade``, only that grade's observations (batches recorded
        without a grade array contribute nothing).
        """
        zs, ys, ws, gs = self._z[start:], self._y[start:], self._w[start:], self._g[start:]
        if grade is None:
            parts = list(zip(zs, ys, ws, strict=True))
        else:
            parts = [
                (z[g == grade], y[g == grade], w[g == grade])
                for z, y, w, g in zip(zs, ys, ws, gs, strict=True)
                if g is not None
            ]
        if not parts:
            return np.empty(0), np.empty(0), np.empty(0)
        z_c, y_c, w_c = (np.concatenate(a) for a in zip(*parts, strict=True))
        return z_c, y_c, w_c

    def _past(self, grade: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The plug-ins' training data: the ``plug_in_window`` trailing batches."""
        return self._since(self._trailing_start(), grade)

    def _recommendation_window_start(self, onset_idx: int | None) -> int:
        """Start index of the post-alarm diagnostic/action window.

        Shared by :meth:`report` (the trailing-window diagnostics) and
        :meth:`apply_recommendation` (the re-offset estimation window), so
        the two can never disagree about which batches "the window" means.

        ``"since_onset"``: ``onset_idx``, or the LATER of ``onset_idx`` and
        the ``plug_in_window`` trailing start when both are set -- a short
        ``plug_in_window`` still bounds how far back the since-onset window
        can reach. ``"trailing"``: the ``plug_in_window`` trailing start (or
        0, i.e. all history, when ``plug_in_window`` is ``None``) --
        ``onset_idx`` is ignored, matching 0.2.0 behaviour exactly. An
        ``onset_idx`` of ``None`` (onset unavailable, see
        :meth:`_onset_available`) takes the ``"trailing"`` branch whatever
        ``recommendation_window`` says.
        """
        if self.recommendation_window == "since_onset" and onset_idx is not None:
            return max(onset_idx, self._trailing_start())
        return self._trailing_start()

    def _onset_available(self) -> bool:
        """Whether every step carries a plug-in log-LR increment.

        Steps loaded from a pre-0.3 payload carry ``log_e_increment =
        None`` -- that payload simply does not record the series
        :func:`~probcal.monitor._onset.estimate_onset` reads, and there is
        no way to reconstruct it from the retained batches. Onset is then
        unavailable and both :meth:`report` and
        :meth:`apply_recommendation` fall back to the ``"trailing"``
        window.
        """
        return all(s.log_e_increment is not None for s in self.steps_)

    def _onset_index(self) -> int | None:
        """Backward-CUSUM argmax onset index, by index -- not by label.

        Shared by :meth:`report` and :meth:`apply_recommendation` so the
        two never disagree about which batch onset points to (batch labels
        are opaque and may repeat, so a lookup by label could point at the
        wrong index). Returns ``None`` when onset is unavailable
        (:meth:`_onset_available`). Meaningful only once at least one batch
        has been processed.
        """
        if not self._onset_available():
            return None
        increments = np.array([s.log_e_increment for s in self.steps_], dtype=np.float64)
        return estimate_onset(increments)

    # ------------------------------------------------------------------ updates

    def _grade_array(self, grade: object, p_arr: np.ndarray) -> np.ndarray:
        """Normalized grade keys for this batch (masterscale-assigned or given)."""
        if hasattr(grade, "assign") and hasattr(grade, "fingerprint"):
            fp = grade.fingerprint()
            if self.masterscale_fingerprint_ is None:
                self.masterscale_fingerprint_ = fp
            elif fp != self.masterscale_fingerprint_:
                raise ValueError(
                    "this monitor was started with masterscale "
                    f"{self.masterscale_fingerprint_[:12]}...; a different masterscale "
                    f"({fp[:12]}...) changes the grade universe. Start a new monitor."
                )
            g_arr = grade_keys(grade.assign(p_arr))
        else:
            g_arr = grade_keys(grade)
        if len(g_arr) != len(p_arr):
            raise ValueError("grade and p must have equal length")
        universe = self._grade_universe
        if universe is not None:
            unknown = sorted(set(np.unique(g_arr).tolist()) - set(universe))
            if unknown:
                raise ValueError(
                    f"grade labels {unknown} are not in the declared grades {universe}; "
                    "the grade universe is fixed at construction (anytime validity "
                    "needs fixed mixture weights). Start a new monitor to change it."
                )
        elif not self._warned_grades:
            warnings.warn(
                "grade arrays were passed but grades=None: per-grade e-processes are "
                "tracked and reported, but they are not part of the global alarm "
                "(a grade first seen mid-stream would re-weight the mixture and break "
                "anytime validity). Declare the universe up front, e.g. "
                "CalibrationMonitor(grades=masterscale.names), to include them.",
                UserWarning,
                stacklevel=3,
            )
            self._warned_grades = True
        return g_arr

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
        grade : array_like, Masterscale, or None
            Optional per-observation grade labels, or a
            :class:`probcal.Masterscale` that assigns them from ``p``;
            either advances the per-grade offset processes (part of the
            global alarm only when ``grades`` was declared). The first
            masterscale seen is recorded as ``masterscale_fingerprint_`` and
            serialized; a different one later raises.
        label : str or None
            Batch label for reporting; defaults to ``batch-<k>``.

        Returns
        -------
        MonitorStep
            The running record after this batch.

        Raises
        ------
        ValueError
            On an empty batch, mismatched lengths, invalid values, or a
            grade label outside the declared ``grades``.
        """
        y_arr = _validate_outcomes(y)
        if y_arr.size == 0:
            raise ValueError("batch is empty: y and p must contain at least one observation")
        p_arr = validate_scores(p, name="p")
        if len(y_arr) != len(p_arr):
            raise ValueError("y and p must have equal length")
        w_arr = validate_weights(sample_weight, len(p_arr))
        g_arr = self._grade_array(grade, p_arr) if grade is not None else None
        if not self._warned_weights and not np.all(w_arr == w_arr[0]):
            warnings.warn(
                "non-uniform sample weights break the exact martingale property; "
                "the e-values remain reported for parity but the type-I guarantee "
                "is approximate",
                UserWarning,
                stacklevel=2,
            )
            self._warned_weights = True
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
        # The onset series sums the plug-in increments of the processes that
        # drive the alarm, and nothing else.
        log_e_increment = 0.0
        if "offset" in self.components:
            log_e_increment += offset_inc
        if "shape" in self.components:
            log_e_increment += shape_inc

        if g_arr is not None:
            grade_inc = 0.0
            for g in (str(x) for x in np.unique(g_arr)):
                if g not in self._grade_procs:
                    self._grade_procs[g] = OffsetProcess(self._sym_grid())
                if g not in self._grade_cs:
                    self._grade_cs[g] = ConfidenceSequence(self._cs.grid)
                mask = g_arr == g
                if history_ready:
                    d_g = plug_in_delta(*self._past(grade=g))
                else:
                    d_g = 0.0
                zg, pg, yg, wg = z[mask], p_arr[mask], y_arr[mask], w_arr[mask]
                grade_inc += self._grade_procs[g].update(zg, pg, yg, wg, d_g)
                # Per-grade CS: the global construction on this grade's own
                # slice, with the grade's own plug-in as the alternative.
                self._grade_cs[g].update(zg, pg, yg, wg, d_g)
            if self.grades is not None:
                log_e_increment += grade_inc

        self._cs.update(z, p_arr, y_arr, w_arr, delta_hat)

        # Store the batch AFTER the plug-ins consumed only the past.
        self._z.append(z)
        self._y.append(y_arr)
        self._w.append(w_arr)
        self._g.append(g_arr)

        step = self._make_step(label, y_arr, w_arr, delta_hat, a_hat, log_e_increment)
        self.steps_.append(step)
        return step

    def _log_global(self) -> float:
        """log of the global mixture: the equal-weight mean of the fixed components."""
        log_parts: list[float] = []
        if "offset" in self.components:
            log_parts.append(self._offset.log_e())
        if "shape" in self.components:
            log_parts.append(self._log_shape)
        universe = self._grade_universe
        if universe is not None:
            grade_logs = np.array([self._grade_procs[g].log_e() for g in universe])
            log_parts.append(logsumexp(grade_logs) - np.log(len(grade_logs)))
        return float(logsumexp(np.array(log_parts)) - np.log(len(log_parts)))

    def _make_step(
        self,
        label: str | None,
        y_arr: np.ndarray,
        w_arr: np.ndarray,
        delta_hat: float,
        a_hat: float,
        log_e_increment: float,
    ) -> MonitorStep:
        e_offset = float(np.exp(self._offset.log_e())) if "offset" in self.components else np.nan
        e_shape = float(np.exp(self._log_shape)) if "shape" in self.components else np.nan
        e_grades = {g: float(np.exp(proc.log_e())) for g, proc in self._grade_procs.items()}
        log_global = self._log_global()
        self._max_log_global = max(self._max_log_global, log_global)
        threshold = -np.log(self.alpha)
        if log_global >= threshold:
            self._alarmed = True
        return MonitorStep(
            label=str(label) if label is not None else f"batch-{len(self.steps_)}",
            n=int(len(y_arr)),
            n_events=float(np.sum(w_arr * y_arr)),
            e_offset=float(e_offset),
            e_shape=float(e_shape),
            e_grades=e_grades,
            e_global=float(np.exp(log_global)),
            p_anytime=float(min(1.0, np.exp(-self._max_log_global))),
            alarm=self._alarmed,
            delta_ci=self._cs.interval(threshold),
            delta_hat=float(delta_hat),
            slope_hat=float(a_hat),
            grade_delta_ci={g: cs.interval(threshold) for g, cs in self._grade_cs.items()},
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
        onset_idx = self._onset_index()
        onset_label = self.steps_[onset_idx].label if onset_idx is not None else None
        # Shared with apply_recommendation(), so the diagnostic window here
        # and the action window there never disagree.
        pz, py, pw = self._since(self._recommendation_window_start(onset_idx))
        recommendation, reasoning = recommend(
            pz,
            py,
            pw,
            alarm_at=alarm_at,
            e_shape=self.steps_[-1].e_shape,
            alpha=self.alpha,
            onset_label=onset_label,
        )
        return MonitorReport(
            steps=tuple(self.steps_),
            alarm_at=alarm_at,
            recommendation=recommendation,
            reasoning=reasoning,
            alpha=self.alpha,
            grade_table=grade_table,
            onset_label=onset_label,
        )

    def apply_recommendation(self, target: object = None) -> Any:
        """Apply :meth:`report`'s recommendation once, closing the re-offset loop.

        ``"re-offset"``: estimates the log-odds shift by maximum likelihood
        (:func:`~probcal.offset.estimate_offset`) on the batches of the
        recommendation window (the same window :meth:`report` uses for its
        trailing-window diagnostics; the trailing one when onset is
        unavailable), composes the fitted offset onto ``target`` (see
        below), and returns a **fresh** monitor with the same constructor
        parameters (:meth:`_ctor_params`) to watch the corrected pipeline.
        The monitor is fresh, not continued: its e-process is a martingale
        under the null "the CURRENTLY DEPLOYED forecast is calibrated";
        once ``target`` changes, the accumulated evidence describes a
        forecast that no longer exists.

        ``"re-fit"``/``"none"``: no offset, composed target, or fresh
        monitor is produced. Automatic re-fitting is deliberately out of
        scope: a slope drift needs a human to choose and validate a new
        calibrator, not a mechanical action this method could take safely.

        Composing the fitted offset onto ``target``:

        - ``None`` (default) -- ``composed`` is ``None``; only the offset
          (and the fresh monitor) come back.
        - :class:`~probcal.chain.Chain` -- a new
          ``Chain([target.calibrator_, *target.offsets_, offset])``;
          ``target`` itself is untouched.
        - :class:`~probcal.wrapper.CalibratedModel` --
          ``target.with_offset(offset)``: a copy with the fitted offset
          appended (works for a model loaded from JSON too, since no
          calibration data is needed); ``target`` itself is untouched.

        Parameters
        ----------
        target : Chain, CalibratedModel, or None
            The currently deployed pipeline to correct. ``None`` (default)
            returns the fitted offset alone.

        Returns
        -------
        AppliedAction
            ``kind``, the fitted ``offset`` (``None`` unless
            ``kind="re-offset"``), the ``composed`` pipeline (``None``
            unless ``kind="re-offset"`` and a ``target`` was given), a
            fresh ``monitor`` (``None`` unless ``kind="re-offset"``), the
            ``window`` of batch labels the estimate used, and an ``audit``
            trail of fingerprints and the estimated ``delta``/``se``.

        Raises
        ------
        TypeError
            If ``target`` is not ``None``, a ``Chain``, or a
            ``CalibratedModel``.

        Notes
        -----
        ``self`` is never mutated: the returned monitor is a brand-new
        object.

        Examples
        --------
        >>> import numpy as np
        >>> from probcal._math import expit, logit
        >>> from probcal.datasets import make_pd_portfolio
        >>> from probcal.monitor import CalibrationMonitor
        >>> mon = CalibrationMonitor(alpha=0.05)
        >>> for k in range(6):
        ...     d = make_pd_portfolio(n=1000, random_state=k)
        ...     rng = np.random.default_rng(k + 1000)
        ...     y = (rng.random(1000) < expit(logit(d.scores) + 0.8)).astype(float)
        ...     _ = mon.update(y, d.scores, label=f"m{k}")
        >>> action = mon.apply_recommendation()
        >>> action.kind
        're-offset'
        >>> action.monitor is not mon
        True
        """
        from ._actions import apply_recommendation

        return apply_recommendation(self, target)

    # ------------------------------------------------------------------ serialization

    def _ctor_params(self) -> dict[str, Any]:
        """Constructor parameters as a plain dict.

        Shared by :meth:`to_dict`'s ``params`` section and
        :meth:`apply_recommendation`'s fresh monitor (``CalibrationMonitor
        (**self._ctor_params())``), so the two can never drift apart.
        """
        return {
            "alpha": self.alpha,
            "components": self.components,
            "grades": self.grades,
            "mixture_grid": self.mixture_grid,
            "delta_ci_grid": self.delta_ci_grid,
            "min_history": self.min_history,
            "plug_in_window": self.plug_in_window,
            "recommendation_window": self.recommendation_window,
        }

    def to_dict(self) -> dict[str, object]:
        """Versioned snapshot; the state includes every past batch — that is
        what makes each decision reproducible (spec invariant)."""
        p = self._ctor_params()
        return envelope(
            self,
            params={
                "alpha": p["alpha"],
                "components": list(p["components"]),
                "grades": list(p["grades"]) if p["grades"] is not None else None,
                "mixture_grid": list(p["mixture_grid"]),
                "delta_ci_grid": list(p["delta_ci_grid"]),
                "min_history": p["min_history"],
                "plug_in_window": p["plug_in_window"],
                "recommendation_window": p["recommendation_window"],
            },
            state=encode_value(self._state_dict()),
            fit_meta={
                "n_batches": len(self._z),
                "n_obs": int(sum(len(a) for a in self._z)),
            },
        )

    def _state_dict(self) -> dict[str, object]:
        state: dict[str, object] = {
            "z": [a.tolist() for a in self._z],
            "y": [a.tolist() for a in self._y],
            "w": [a.tolist() for a in self._w],
            "g": [a.tolist() if a is not None else None for a in self._g],
            "offset": self._offset.state(),
            "log_shape": self._log_shape,
            "grade_procs": {g: p.state() for g, p in self._grade_procs.items()},
            "cs_log": self._cs.log.tolist(),
            "cs_max": self._cs.max.tolist(),
            "grade_cs_log": {g: cs.log.tolist() for g, cs in self._grade_cs.items()},
            "grade_cs_max": {g: cs.max.tolist() for g, cs in self._grade_cs.items()},
            "max_log_global": float(self._max_log_global),
            "alarmed": self._alarmed,
            "warned_weights": self._warned_weights,
            "steps": [self._step_to_dict(s) for s in self.steps_],
        }
        if self._warned_grades:  # written only when set: older payloads stay byte-identical
            state["warned_grades"] = True
        if self.masterscale_fingerprint_ is not None:
            state["masterscale_fingerprint"] = self.masterscale_fingerprint_
        return state

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

        Payloads written before 0.3.4 keyed float grade labels as
        ``"1.0"``; those keys are renamed to the normalized ``"1"`` on load
        so new batches continue the same per-grade processes.

        Raises
        ------
        ValueError
            If the schema version is unknown or the payload class differs.
        """
        check_payload(cls, d)
        params = dict(d["params"])
        params["components"] = tuple(params["components"])
        params["grades"] = tuple(params["grades"]) if params["grades"] is not None else None
        params["mixture_grid"] = tuple(params["mixture_grid"])
        params["delta_ci_grid"] = tuple(params["delta_ci_grid"])
        params["recommendation_window"] = params.get("recommendation_window", "since_onset")
        mon = cls(**params)
        st: dict[str, Any] = decode_value(d["state"], arrays=False)  # type: ignore[assignment]
        rekey = _legacy_rekey(d, params["grades"])
        mon._z = [np.asarray(a, dtype=np.float64) for a in st["z"]]
        mon._y = [np.asarray(a, dtype=np.float64) for a in st["y"]]
        mon._w = [np.asarray(a, dtype=np.float64) for a in st["w"]]
        mon._g = [
            np.asarray([rekey(x) for x in a], dtype=str) if a is not None else None for a in st["g"]
        ]
        mon._offset.set_state(st["offset"])
        mon._log_shape = float(st["log_shape"])
        mon._grade_procs = {}
        for g, ps in st["grade_procs"].items():
            proc = OffsetProcess(mon._sym_grid())
            proc.set_state(ps)
            mon._grade_procs[rekey(g)] = proc
        mon._cs.set_state(st["cs_log"], st["cs_max"])
        grade_cs_log = st.get("grade_cs_log", {})
        grade_cs_max = st.get("grade_cs_max", {})
        mon._grade_cs = {}
        for g in grade_cs_log:
            cs = ConfidenceSequence(mon._cs.grid)
            cs.set_state(grade_cs_log[g], grade_cs_max[g])
            mon._grade_cs[rekey(g)] = cs
        mon._max_log_global = float(st["max_log_global"])
        mon._alarmed = bool(st["alarmed"])
        mon._warned_weights = bool(st["warned_weights"])
        mon._warned_grades = bool(st.get("warned_grades", False))
        mon.masterscale_fingerprint_ = st.get("masterscale_fingerprint")
        mon.steps_ = []
        for sd in st["steps"]:
            sd = dict(sd)
            sd["delta_ci"] = tuple(sd["delta_ci"]) if sd["delta_ci"] is not None else None
            sd["e_grades"] = {rekey(g): v for g, v in sd["e_grades"].items()}
            sd["grade_delta_ci"] = {
                rekey(g): (tuple(v) if v is not None else None)
                for g, v in sd.get("grade_delta_ci", {}).items()
            }
            mon.steps_.append(MonitorStep(**sd))
        return mon


def _legacy_rekey(d: dict, grades: tuple | None) -> Any:
    """Key renaming for payloads written before grade-label normalization.

    Before 0.3.4 declared grades were keyed ``str(g)`` and data labels
    ``np.asarray(grade).astype(str)``, so a float label ``1.0`` was keyed
    ``"1.0"``. For such payloads, keys of declared grades map to
    :func:`grade_key`, and undeclared ``"<int>.0"`` keys (float arrays) map
    to ``"<int>"``. Newer payloads are already normalized: identity.
    """
    if _version_tuple(d.get("probcal_version", "0")) >= (0, 4, 0):
        return str
    declared = {str(g): grade_key(g) for g in grades or ()}

    def rekey(k: object) -> str:
        k = str(k)
        if k in declared:
            return declared[k]
        if _LEGACY_FLOAT_KEY.match(k):
            return k[:-2]
        return k

    return rekey
