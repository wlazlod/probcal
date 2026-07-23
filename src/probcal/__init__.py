"""probcal: universal post-hoc probability calibration for binary classifiers.

Methods, metrics, diagnostics, and auditable offsetting — numpy-only.
"""

from . import metrics
from ._math import expit, logit
from .attribution import AdjustedAttribution, adjust_attributions
from .base import BaseCalibrator
from .bayesian import BBQCalibrator, ENIRCalibrator
from .binning import HistogramBinningCalibrator, ScalingBinningCalibrator
from .curves import calibration_belt, reliability_binned, reliability_loess, reliability_spline
from .isotonic import CenteredIsotonicCalibrator, IsotonicCalibrator
from .offset import LogitOffset
from .parametric import BetaCalibrator, PlattCalibrator, TemperatureCalibrator
from .spline import SplineCalibrator
from .vennabers import CrossVennAbersCalibrator, VennAbersCalibrator

__version__ = "0.0.1"

__all__: list[str] = [
    "AdjustedAttribution",
    "BBQCalibrator",
    "BaseCalibrator",
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
    "VennAbersCalibrator",
    "adjust_attributions",
    "calibration_belt",
    "expit",
    "metrics",
    "reliability_binned",
    "reliability_loess",
    "reliability_spline",
    "logit",
]
