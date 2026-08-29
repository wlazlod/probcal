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

    def _rows(self) -> list[tuple[str, str, float, float, float]]:
        panels = (("pooled", self.pooled), *zip(self.groups, self.reports, strict=True))
        return [
            (group, n, v, lo, hi)
            for group, rep in panels
            for n, v, lo, hi in zip(rep.names, rep.values, rep.ci_low, rep.ci_high, strict=True)
        ]

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

    def __repr__(self) -> str:
        table = _aligned_table(("group", "metric", "value", "ci_low", "ci_high"), self._rows())
        return f"GroupedMetricReport ({len(self.groups)} groups)\n{table}"


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

    def __repr__(self) -> str:
        has_corp = self.mcb is not None and self.dsc is not None
        headers: tuple[str, ...]
        rows: list[tuple[object, ...]]
        if has_corp:
            headers = ("method", self.criterion, "sd", "guardrails", "chosen", "mcb", "dsc")
            rows = [
                (m, sm, sd, bool(g), "*" if c else "", mc, ds)
                for m, sm, sd, g, c, mc, ds in zip(
                    self.methods,
                    self.score_mean,
                    self.score_sd,
                    self.guardrails_ok,
                    self.chosen,
                    self.mcb,  # type: ignore[arg-type]
                    self.dsc,  # type: ignore[arg-type]
                    strict=True,
                )
            ]
        else:
            headers = ("method", self.criterion, "sd", "guardrails", "chosen")
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
        table = _aligned_table(headers, rows)
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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

    def __repr__(self) -> str:
        rows = [
            (lo, hi, lvl, wt)
            for lo, hi, lvl, wt in zip(
                self.block_lo, self.block_hi, self.block_level, self.block_weight, strict=True
            )
        ]
        table = _aligned_table(("block_lo", "block_hi", "block_level", "block_weight"), rows)
        return (
            f"CorpResult (n={self.n}, events={self.events})\n{table}\n"
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
