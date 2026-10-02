"""``reliability_summary``: the stats box of the annotated reliability diagram."""

from dataclasses import dataclass

from ._common import _prep
from .regression import _intercept, _slope_fit
from .smooth import _ici_family, _spiegelhalter


@dataclass(frozen=True)
class ReliabilitySummary:
    """Stats-box aggregate for the annotated reliability diagram.

    Attributes
    ----------
    n : int
        Observation count.
    events : int
        Event count (``sum(y)``).
    intercept : float
        Calibration-in-the-large intercept (log-odds).
    slope : float
        Cox calibration slope.
    ici : float
        Integrated calibration index.
    e90 : float
        90th percentile of the LOESS distances.
    spiegelhalter_p : float
        Spiegelhalter test p-value.
    """

    n: int
    events: int
    intercept: float
    slope: float
    ici: float
    e90: float
    spiegelhalter_p: float


def reliability_summary(
    y: object,
    p: object,
    *,
    sample_weight: object = None,
    grid_size: int | None = 512,
) -> ReliabilitySummary:
    """Assemble the annotated-reliability stats box from existing metrics.

    No new math: intercept and slope from the recalibration regression, ICI
    and E90 from the LOESS distance family, and Spiegelhalter's p-value.
    Lives here because, like `evaluate`, it aggregates across submodules;
    ``probcal.plots`` only formats the result. ``grid_size=None`` recovers
    0.1.2 values exactly.

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``.
    p : array_like
        Predicted probabilities in ``[0, 1]``.
    sample_weight : array_like or None, keyword-only
        Optional positive weights, same length as ``y``.
    grid_size : int or None, keyword-only
        LOESS evaluation grid size for the ICI/E90 terms; ``None`` recovers
        0.1.2 values exactly.

    Returns
    -------
    ReliabilitySummary
        Stats-box fields for the annotated reliability diagram.
    """
    y_arr, p_arr, w = _prep(y, p, sample_weight)
    wq = None if sample_weight is None else w
    fam = _ici_family(y_arr, p_arr, wq, ("ici", "e90"), grid_size=grid_size)
    return ReliabilitySummary(
        n=len(y_arr),
        events=int(y_arr.sum()),
        intercept=_intercept(y_arr, p_arr, w),
        slope=float(_slope_fit(y_arr, p_arr, w)[1][1]),
        ici=fam["ici"],
        e90=fam["e90"],
        spiegelhalter_p=_spiegelhalter(y_arr, p_arr, w).p_value,
    )
