"""probcal: universal post-hoc probability calibration for binary classifiers.

Methods, metrics, diagnostics, and auditable offsetting — numpy-only.
"""

from . import metrics
from ._math import expit, logit
from .attribution import AdjustedAttribution, adjust_attributions
from .base import BaseCalibrator, UnattainableTargetError
from .bayesian import BBQCalibrator, ENIRCalibrator
from .binning import HistogramBinningCalibrator, ScalingBinningCalibrator
from .curves import calibration_belt, reliability_binned, reliability_loess, reliability_spline
from .datasets import make_pd_portfolio
from .isotonic import CenteredIsotonicCalibrator, IsotonicCalibrator
from .offset import LogitOffset
from .parametric import BetaCalibrator, PlattCalibrator, TemperatureCalibrator
from .selection import CalibratorSelector
from .spline import SplineCalibrator
from .thresholds import calibrated_bands_to_raw, calibrated_interval_to_raw
from .vennabers import CrossVennAbersCalibrator, VennAbersCalibrator
from .wrapper import CalibratedModel

__version__ = "0.1.1"

__all__: list[str] = [
    "AdjustedAttribution",
    "BBQCalibrator",
    "BaseCalibrator",
    "CalibratedModel",
    "CalibratorSelector",
    "BetaCalibrator",
    "CenteredIsotonicCalibrator",
    "CrossVennAbersCalibrator",
    "ENIRCalibrator",
    "HistogramBinningCalibrator",
    "IsotonicCalibrator",
    "LogitOffset",
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
    "expit",
    "metrics",
    "reliability_binned",
    "reliability_loess",
    "reliability_spline",
    "logit",
    "make_pd_portfolio",
]
