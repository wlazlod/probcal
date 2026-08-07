"""Frozen result dataclasses returned by probcal APIs.

All results are immutable dataclasses of numpy arrays (no pandas), each with an
``as_dict()`` accessor and a readable aligned-table ``__repr__``. Field sets here
are the initial minimum and may be extended by later releases.
"""

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

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


class _ResultBase:
    """Shared ``as_dict`` for frozen result dataclasses."""

    def as_dict(self) -> dict[str, object]:
        """Return the result's fields as a plain dict."""
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}  # type: ignore[arg-type]


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


@dataclass(frozen=True)
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

    def __repr__(self) -> str:
        rows = [
            (p, e, c, lo, hi)
            for p, e, c, lo, hi in zip(
                self.pred_mean,
                self.event_rate,
                self.count,
                self.ci_low,
                self.ci_high,
                strict=True,
            )
        ]
        table = _aligned_table(("pred_mean", "event_rate", "count", "ci_low", "ci_high"), rows)
        return f"ReliabilityCurve ({len(rows)} bins)\n{table}"


@dataclass(frozen=True)
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

    def __repr__(self) -> str:
        rows = [
            (n, v, lo, hi)
            for n, v, lo, hi in zip(self.names, self.values, self.ci_low, self.ci_high, strict=True)
        ]
        table = _aligned_table(("metric", "value", "ci_low", "ci_high"), rows)
        return f"MetricReport\n{table}"


@dataclass(frozen=True)
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
    """

    methods: tuple[str, ...]
    score_mean: np.ndarray
    score_sd: np.ndarray
    guardrails_ok: np.ndarray
    chosen: np.ndarray
    criterion: str

    def __repr__(self) -> str:
        rows = [
            (m, sm, sd, bool(g), "*" if c else "")
            for m, sm, sd, g, c in zip(
                self.methods,
                self.score_mean,
                self.score_sd,
                self.guardrails_ok,
                self.chosen,
                strict=True,
            )
        ]
        table = _aligned_table(("method", self.criterion, "sd", "guardrails", "chosen"), rows)
        return f"SelectionReport (criterion: {self.criterion})\n{table}"


@dataclass(frozen=True)
class BeltResult(_ResultBase):
    """GiViTI-style calibration belt: bands, polynomial degree, and test p-value.

    Attributes
    ----------
    grid_p, grid_logit : numpy.ndarray
        Evaluation grid on the probability and logit scales.
    lower_80, upper_80, lower_95, upper_95 : numpy.ndarray
        Pointwise confidence band bounds at the two default levels.
    degree : int
        Polynomial degree selected by forward likelihood-ratio testing.
    p_value : float
        P-value of the associated calibration test.
    """

    grid_p: np.ndarray
    grid_logit: np.ndarray
    lower_80: np.ndarray
    upper_80: np.ndarray
    lower_95: np.ndarray
    upper_95: np.ndarray
    degree: int
    p_value: float

    def __repr__(self) -> str:
        return (
            f"BeltResult(degree={self.degree}, p_value={self.p_value:.4g}, "
            f"grid of {len(self.grid_p)} points)"
        )


@dataclass(frozen=True)
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
