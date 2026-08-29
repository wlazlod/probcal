"""probcal: universal post-hoc probability calibration for binary classifiers.

Methods, metrics, diagnostics, and auditable offsetting — numpy-only.
"""

from . import metrics, monitor
from ._math import expit, logit
from .attribution import AdjustedAttribution, adjust_attributions
from .base import BaseCalibrator, UnattainableTargetError
from .bayesian import BBQCalibrator, ENIRCalibrator
from .binning import HistogramBinningCalibrator, ScalingBinningCalibrator
from .chain import Chain
from .curves import (
    calibration_belt,
    corp_reliability,
    reliability_binned,
    reliability_loess,
    reliability_smooth,
    reliability_spline,
)
from .datasets import make_pd_portfolio
from .isotonic import CenteredIsotonicCalibrator, IsotonicCalibrator
from .monitor import moc_offset, moc_offset_from_counts
from .offset import LogitOffset, OffsetEstimate, estimate_offset, offset_from_estimate
from .parametric import BetaCalibrator, PlattCalibrator, TemperatureCalibrator
from .selection import CalibratorSelector
from .spline import SplineCalibrator
from .thresholds import calibrated_bands_to_raw, calibrated_interval_to_raw
from .vennabers import CrossVennAbersCalibrator, VennAbersCalibrator
from .wrapper import CalibratedModel

__version__ = "0.2.0"

__all__: list[str] = [
    "AdjustedAttribution",
    "BBQCalibrator",
    "BaseCalibrator",
    "CalibratedModel",
    "CalibratorSelector",
    "BetaCalibrator",
    "CenteredIsotonicCalibrator",
    "Chain",
    "CrossVennAbersCalibrator",
    "ENIRCalibrator",
    "HistogramBinningCalibrator",
    "IsotonicCalibrator",
    "LogitOffset",
    "OffsetEstimate",
    "PlattCalibrator",
    "ScalingBinningCalibrator",
    "SplineCalibrator",
    "TemperatureCalibrator",
    "UnattainableTargetError",
    "VennAbersCalibrator",
    "adjust_attributions",
    "calibrated_bands_to_raw",
    "calibrated_interval_to_raw",
    "calibration_belt",
    "corp_reliability",
    "estimate_offset",
    "expit",
    "metrics",
    "monitor",
    "offset_from_estimate",
    "reliability_binned",
    "reliability_loess",
    "reliability_smooth",
    "reliability_spline",
    "logit",
    "make_pd_portfolio",
    "moc_offset",
    "moc_offset_from_counts",
]
