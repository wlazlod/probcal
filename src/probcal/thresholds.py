"""Calibrated-to-raw interval and masterscale-band mapping.

Thin functional wrappers over the calibrators' ``interval_inverse`` protocol
(spec §10): numpy-only, arrays and floats, no knowledge of any consumer. The
canonical rating-grade workflow — masterscale bands defined on calibrated PD,
translated once per recalibration into raw-score intervals — is
``calibrated_bands_to_raw``; its output plugs directly into band-style raw
targets of a counterfactual engine.
"""


def calibrated_interval_to_raw(
    calibrator: object,
    lo: float,
    hi: float,
    *,
    space: str = "probability",
    buffer_logit: float = 0.0,
) -> tuple[float, float]:
    """Translate one calibrated-probability interval into raw-score bounds.

    Parameters
    ----------
    calibrator : fitted calibrator
        Any object implementing the duck-typed protocol
        ``interval_inverse(lo, hi, *, space, buffer_logit)`` with
        ``is_monotone_``.
    lo, hi : float
        Calibrated bounds; ``lo=0`` / ``hi=1`` map to the full raw range.
    space : {"probability", "logit"}
        Scale of the returned bounds.
    buffer_logit : float
        Robustness margin applied in logit space before inversion.
    """
    return calibrator.interval_inverse(lo, hi, space=space, buffer_logit=buffer_logit)  # type: ignore[attr-defined]


def calibrated_bands_to_raw(
    calibrator: object,
    bands: dict,
    *,
    space: str = "probability",
    buffer_logit: float = 0.0,
) -> dict:
    """Translate a masterscale ``{grade: (lo, hi)}`` on calibrated PD to raw intervals.

    Grade edges are policy artifacts that outlive model versions; this
    translation is what changes when the calibrator is refitted.
    """
    return {
        grade: calibrated_interval_to_raw(
            calibrator, lo, hi, space=space, buffer_logit=buffer_logit
        )
        for grade, (lo, hi) in bands.items()
    }
