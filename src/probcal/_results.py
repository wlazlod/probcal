"""Frozen result dataclasses returned by probcal APIs.

All results are immutable dataclasses of numpy arrays (no pandas), each with an
``as_dict()`` accessor and a readable aligned-table ``__repr__``. Field sets here
are the initial minimum and may be extended by later releases.

Results are *reports*, not serializable artifacts: ``as_dict()`` returns the
fields as they are (numpy arrays included) for inspection or a DataFrame,
whereas the fitted objects' ``to_dict()`` returns a versioned, JSON-native
payload that ``from_dict`` reads back. The two names are deliberately
different because the two contracts are.

Equality is array-aware: two results compare equal when every field is equal,
arrays by value (``nan`` equal to ``nan``). Results holding arrays are
unhashable (``__hash__ = None``), like the arrays themselves.
"""

import dataclasses
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np


def _format_cell(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}"
    return str(value)


def _aligned_table(headers: tuple[str, ...], rows: Sequence[Sequence[object]]) -> str:
    """Render rows as a plain-text table with left-aligned, padded columns."""
    cells = [tuple(_format_cell(v) for v in row) for row in rows]
    widths = [len(h) for h in headers]
    for row in cells:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = [
        "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)),
        "  ".join("-" * w for w in widths),
    ]
    lines += ["  ".join(c.ljust(widths[i]) for i, c in enumerate(row)) for row in cells]
    return "\n".join(lines)


def _values_equal(a: object, b: object) -> bool:
    """Field equality that understands numpy arrays (``nan == nan``) and nests."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a_arr, b_arr = np.asarray(a), np.asarray(b)
        if a_arr.shape != b_arr.shape:
            return False
        try:
            return bool(np.array_equal(a_arr, b_arr, equal_nan=True))
        except TypeError:  # non-numeric dtype: nan-equality is meaningless
            return bool(np.array_equal(a_arr, b_arr))
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return (
            type(a) is type(b)
            and len(a) == len(b)
            and all(_values_equal(x, y) for x, y in zip(a, b, strict=True))
        )
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True
    return bool(a == b)


def fields_equal(a: object, b: object) -> bool:
    """``True`` iff ``a`` and ``b`` are the same dataclass type with equal fields.

    Arrays compare by value (shape and elements, ``nan`` equal to ``nan``).
    The shared ``__eq__`` of every result dataclass holding arrays.
    """
    if type(a) is not type(b) or not dataclasses.is_dataclass(a):
        return False
    return all(_values_equal(getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a))


class _ResultBase:
    """Shared ``as_dict``, array-aware ``__eq__``, and table ``__repr__``.

    Subclasses holding arrays are declared ``@dataclass(frozen=True,
    eq=False, repr=False)`` so they inherit the methods below (a dataclass
    with ``eq=True`` would generate an ``__eq__`` that raises on arrays).

    The table mechanism: ``_TABLE`` names the column headers and
    :meth:`_rows` returns one tuple per row (by default the fields named in
    ``_TABLE``, zipped); ``__repr__`` renders ``_title()`` above the aligned
    table, and the HTML/markdown report renders the very same rows.
    """

    _TABLE: ClassVar[tuple[str, ...]] = ()

    def as_dict(self) -> dict[str, object]:
        """Return the result's fields as a plain dict (arrays as-is; not JSON)."""
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}  # type: ignore[arg-type]

    def _rows(self) -> list[tuple[object, ...]]:
        """One tuple per table row, aligned with ``_TABLE``."""
        cols = [getattr(self, h) for h in self._TABLE]
        return [tuple(r) for r in zip(*cols, strict=True)]

    def _headers(self) -> tuple[str, ...]:
        """Column headers of the table (``_TABLE`` unless a subclass varies them)."""
        return self._TABLE

    def _title(self) -> str:
        return type(self).__name__

    def __repr__(self) -> str:
        if not self._headers():
            return object.__repr__(self)
        return f"{self._title()}\n{_aligned_table(self._headers(), self._rows())}"

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return fields_equal(self, other)

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Interpretation(_ResultBase):
    """Fitted parameters of a calibrator with plain-language readings.

    Attributes
    ----------
    method : str
        Class name of the calibrator that produced the interpretation.
    param_names : tuple of str
        Names of the fitted parameters.
    param_values : tuple of float
        Fitted values, aligned with ``param_names``.
    messages : tuple of str
        Domain-aware reading of each parameter (and of the fit as a whole).
    """

    method: str
    param_names: tuple[str, ...]
    param_values: tuple[float, ...]
    messages: tuple[str, ...]

    def __repr__(self) -> str:
        table = _aligned_table(
            ("parameter", "value"),
            [(n, v) for n, v in zip(self.param_names, self.param_values, strict=True)],
        )
        notes = "\n".join(f"- {m}" for m in self.messages)
        return f"Interpretation[{self.method}]\n{table}\n{notes}"


@dataclass(frozen=True, eq=False, repr=False)
class ReliabilityCurve(_ResultBase):
    """Binned or smoothed reliability curve on both probability and logit scales.

    Attributes
    ----------
    pred_mean : numpy.ndarray
        Mean predicted probability per bin (or grid point).
    event_rate : numpy.ndarray
        Observed event rate per bin.
    count : numpy.ndarray
        Observation count per bin.
    ci_low, ci_high : numpy.ndarray
        Wilson confidence bounds for the event rate.
    pred_mean_logit : numpy.ndarray
        ``logit(pred_mean)`` — the logit-scale x-coordinates.
    """

    pred_mean: np.ndarray
    event_rate: np.ndarray
    count: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    pred_mean_logit: np.ndarray

    _TABLE: ClassVar[tuple[str, ...]] = ("pred_mean", "event_rate", "count", "ci_low", "ci_high")

    def _title(self) -> str:
        return f"ReliabilityCurve ({len(self.pred_mean)} bins)"


@dataclass(frozen=True, eq=False, repr=False)
class MetricReport(_ResultBase):
    """Named metric values with bootstrap percentile confidence intervals.

    Attributes
    ----------
    names : tuple of str
        Metric names.
    values : numpy.ndarray
        Point estimates, aligned with ``names``.
    ci_low, ci_high : numpy.ndarray
        Bootstrap percentile interval bounds.
    """

    names: tuple[str, ...]
    values: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray

    _TABLE: ClassVar[tuple[str, ...]] = ("metric", "value", "ci_low", "ci_high")

    def _rows(self) -> list[tuple[object, ...]]:
        return list(zip(self.names, self.values, self.ci_low, self.ci_high, strict=True))


@dataclass(frozen=True, eq=False, repr=False)
class GroupedMetricReport(_ResultBase):
    """Per-group metric reports plus a pooled report, from ``metrics.evaluate(by=...)``.

    Attributes
    ----------
    pooled : MetricReport
        Report computed on the full, ungrouped data (the ``by=None`` report,
        using ``seed`` unchanged).
    groups : tuple of str
        Sorted, stringified group labels.
    reports : tuple of MetricReport
        Per-group reports, aligned with ``groups``. Group ``i`` (in this
        sorted order) is computed with ``seed + 1000 * i``, so results are
        reproducible independent of the label values themselves.
    counts : numpy.ndarray
        Observation count per group, aligned with ``groups``.
    """

    pooled: MetricReport
    groups: tuple[str, ...]
    reports: tuple[MetricReport, ...]
    counts: np.ndarray

    _TABLE: ClassVar[tuple[str, ...]] = ("group", "metric", "value", "ci_low", "ci_high")

    def _rows(self) -> list[tuple[object, ...]]:
        panels = (("pooled", self.pooled), *zip(self.groups, self.reports, strict=True))
        return [(group, *row) for group, rep in panels for row in rep._rows()]

    def to_frame(self) -> object:
        """Rows as a list of dicts, or a pandas DataFrame when pandas is importable.

        Each row is ``{"group", "metric", "value", "ci_low", "ci_high"}``;
        the pooled report is included under the group label ``"pooled"``,
        which is therefore reserved — a group of your own named "pooled"
        is indistinguishable from it in this frame.
        """
        rows = [
            {"group": group, "metric": n, "value": v, "ci_low": lo, "ci_high": hi}
            for group, n, v, lo, hi in self._rows()
        ]
        try:
            import pandas as pd
        except ImportError:
            return rows
        return pd.DataFrame(rows)

    def _title(self) -> str:
        return f"GroupedMetricReport ({len(self.groups)} groups)"


@dataclass(frozen=True, eq=False, repr=False)
class SelectionReport(_ResultBase):
    """Ranked outcome of automatic calibrator selection.

    Attributes
    ----------
    methods : tuple of str
        Candidate identifiers.
    score_mean, score_sd : numpy.ndarray
        Out-of-fold criterion mean and standard deviation per candidate.
    guardrails_ok : numpy.ndarray
        Boolean guardrail summary per candidate.
    chosen : numpy.ndarray
        Boolean flag marking the selected candidate.
    criterion : str
        Name of the scoring criterion.
    mcb, dsc : numpy.ndarray or None
        Per-candidate CORP miscalibration/discrimination terms
        (:func:`probcal._corp.decompose`) on the same out-of-fold
        predictions as ``score_mean``, decomposing Brier when
        ``criterion == "brier"`` and log loss otherwise. ``None`` for
        reports produced before probcal 0.3 (e.g. loaded from an older
        golden), since the columns did not exist to compute.
    unc : float or None
        CORP uncertainty term, identical across candidates (it depends
        only on ``y`` and the sample weights, not on the predictions).
        ``None`` exactly when ``mcb``/``dsc`` are ``None``.
    """

    methods: tuple[str, ...]
    score_mean: np.ndarray
    score_sd: np.ndarray
    guardrails_ok: np.ndarray
    chosen: np.ndarray
    criterion: str
    mcb: np.ndarray | None = None
    dsc: np.ndarray | None = None
    unc: float | None = None

    def _has_corp(self) -> bool:
        return self.mcb is not None and self.dsc is not None

    def _headers(self) -> tuple[str, ...]:
        base = ("method", self.criterion, "sd", "guardrails", "chosen")
        return (*base, "mcb", "dsc") if self._has_corp() else base

    def _rows(self) -> list[tuple[object, ...]]:
        cols: list[object] = [
            self.methods,
            self.score_mean,
            self.score_sd,
            [bool(g) for g in self.guardrails_ok],
            ["*" if c else "" for c in self.chosen],
        ]
        if self._has_corp():
            cols += [self.mcb, self.dsc]
        return [tuple(r) for r in zip(*cols, strict=True)]  # type: ignore[call-overload]

    def _title(self) -> str:
        return f"SelectionReport (criterion: {self.criterion})"


@dataclass(frozen=True, eq=False)
class BeltResult(_ResultBase):
    """GiViTI-style calibration belt: bands, polynomial degree, and test p-value.

    Attributes
    ----------
    grid_p, grid_logit : numpy.ndarray
        Evaluation grid on the probability and logit scales.
    levels : tuple of float
        The two confidence levels requested (``confidence=`` argument).
    bands : dict
        ``{level: (lower, upper)}`` band bounds on the probability scale,
        one entry per level in ``levels``.
    degree : int
        Polynomial degree selected by forward likelihood-ratio testing.
    p_value : float
        P-value of the associated calibration test.
    """

    grid_p: np.ndarray
    grid_logit: np.ndarray
    levels: tuple[float, float]
    bands: dict[float, tuple[np.ndarray, np.ndarray]]
    degree: int
    p_value: float

    def _legacy(self, which: int, side: int, name: str) -> np.ndarray:
        level = self.levels[which]
        warnings.warn(
            f"BeltResult.{name} is deprecated and will be removed in 0.5.0; use "
            f"belt.bands[{level!r}][{side}] (it holds the confidence={level!r} band, "
            "whatever the attribute name says)",
            DeprecationWarning,
            stacklevel=3,
        )
        return self.bands[level][side]

    @property
    def lower_80(self) -> np.ndarray:
        """Deprecated: ``bands[levels[0]][0]``."""
        return self._legacy(0, 0, "lower_80")

    @property
    def upper_80(self) -> np.ndarray:
        """Deprecated: ``bands[levels[0]][1]``."""
        return self._legacy(0, 1, "upper_80")

    @property
    def lower_95(self) -> np.ndarray:
        """Deprecated: ``bands[levels[1]][0]``."""
        return self._legacy(1, 0, "lower_95")

    @property
    def upper_95(self) -> np.ndarray:
        """Deprecated: ``bands[levels[1]][1]``."""
        return self._legacy(1, 1, "upper_95")

    def __repr__(self) -> str:
        return (
            f"BeltResult(degree={self.degree}, p_value={self.p_value:.4g}, "
            f"levels={self.levels}, grid of {len(self.grid_p)} points)"
        )


@dataclass(frozen=True, eq=False)
class SmoothReliabilityCurve(_ResultBase):
    """Smoothed reliability curve evaluated on a grid, on both scales.

    Attributes
    ----------
    grid_p, grid_logit : numpy.ndarray
        Evaluation grid on the probability and logit scales.
    event_rate : numpy.ndarray
        Smoothed conditional event rate at each grid point.
    """

    grid_p: np.ndarray
    grid_logit: np.ndarray
    event_rate: np.ndarray

    def __repr__(self) -> str:
        return f"SmoothReliabilityCurve (grid of {len(self.grid_p)} points)"


@dataclass(frozen=True, eq=False)
class KernelReliabilityCurve(_ResultBase):
    """smECE-consistent kernel reliability curve (``curves.reliability_smooth``).

    Attributes
    ----------
    grid_p, grid_logit : numpy.ndarray
        Evaluation grid on the probability and logit scales.
    event_rate : numpy.ndarray
        Kernel-smoothed ``E[y | logit p]`` at ``sigma_star``, evaluated at
        each grid point.
    density : numpy.ndarray
        Kernel-smoothed prediction density at each grid point, normalized
        to sum to 1 over the grid.
    ci_low, ci_high : numpy.ndarray
        Seeded bootstrap percentile band for ``event_rate`` at the fixed
        ``sigma_star`` (empty band collapses to ``event_rate`` when
        ``n_boot=0``).
    sigma_star : float
        The smECE fixed-point bandwidth the curve is smoothed at.
    smooth_ece : float
        ``metrics.smooth_ece(y, p, bins=bins)``, reproduced exactly (same
        lattice and path selection) from the same ``sigma_star``.
    """

    grid_p: np.ndarray
    grid_logit: np.ndarray
    event_rate: np.ndarray
    density: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    sigma_star: float
    smooth_ece: float

    def __repr__(self) -> str:
        return (
            f"KernelReliabilityCurve (grid of {len(self.grid_p)} points, "
            f"sigma_star={self.sigma_star:.4g}, smooth_ece={self.smooth_ece:.4g})"
        )


@dataclass(frozen=True, eq=False)
class CorpResult(_ResultBase):
    """CORP reliability fit: PAV recalibration with the MCB-DSC-UNC decomposition.

    Attributes
    ----------
    block_lo, block_hi : numpy.ndarray
        Left and right edge (min/max ``p``) of each PAV block.
    block_level : numpy.ndarray
        PAV fitted event rate per block.
    block_weight : numpy.ndarray
        Pooled weight per block.
    pav : numpy.ndarray
        PAV fit expanded to observations, in the original input order.
    brier, brier_mcb, brier_dsc, brier_unc : float
        Brier score and its miscalibration/discrimination/uncertainty terms
        (``brier == brier_mcb - brier_dsc + brier_unc``).
    log_loss, log_loss_mcb, log_loss_dsc, log_loss_unc : float
        Log loss and its miscalibration/discrimination/uncertainty terms
        (``log_loss == log_loss_mcb - log_loss_dsc + log_loss_unc``).
    bands : {"consistency", "confidence", None}
        Band type requested.
    level : float
        Nominal coverage level of the bands.
    band_grid, band_low, band_high : numpy.ndarray
        Band evaluation grid and bounds (empty when ``bands`` is ``None``).
    n : int
        Number of observations.
    events : int
        Number of events (``sum(y)``).
    """

    block_lo: np.ndarray
    block_hi: np.ndarray
    block_level: np.ndarray
    block_weight: np.ndarray
    pav: np.ndarray
    brier: float
    brier_mcb: float
    brier_dsc: float
    brier_unc: float
    log_loss: float
    log_loss_mcb: float
    log_loss_dsc: float
    log_loss_unc: float
    bands: str | None
    level: float
    band_grid: np.ndarray
    band_low: np.ndarray
    band_high: np.ndarray
    n: int
    events: int

    _TABLE: ClassVar[tuple[str, ...]] = ("block_lo", "block_hi", "block_level", "block_weight")

    def _title(self) -> str:
        return f"CorpResult (n={self.n}, events={self.events})"

    def __repr__(self) -> str:
        return (
            f"{super().__repr__()}\n"
            f"Brier: {self.brier:.6g} = MCB {self.brier_mcb:.6g} - DSC {self.brier_dsc:.6g} "
            f"+ UNC {self.brier_unc:.6g}\n"
            f"Log loss: {self.log_loss:.6g} = MCB {self.log_loss_mcb:.6g} - "
            f"DSC {self.log_loss_dsc:.6g} + UNC {self.log_loss_unc:.6g}"
        )


@dataclass(frozen=True)
class OffsetEstimate(_ResultBase):
    """Offset-only logistic MLE of ``delta`` given ``p``, with its Fisher standard error.

    Attributes
    ----------
    delta : float
        MLE of the logit-offset shift: the mean-matching root of
        ``sum(w * (y - sigma(logit(p) + delta))) = 0`` (the offset-only
        logistic score equation), found by bisection.
    se : float
        Asymptotic (Fisher-information) standard error of ``delta``,
        ``1 / sqrt(sum(w * q * (1 - q)))`` at ``q = sigma(logit(p) + delta)``.
    n : int
        Number of observations.
    events : float
        Weighted event count, ``sum(w * y)``.
    weight_sum : float
        Sum of weights (equals ``n`` for unit weights).
    """

    delta: float
    se: float
    n: int
    events: float
    weight_sum: float

    def __repr__(self) -> str:
        return (
            f"OffsetEstimate(delta={self.delta:+.4f} +/- {self.se:.4f}, "
            f"n={self.n}, events={self.events:.1f})"
        )
