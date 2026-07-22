"""probcal: universal post-hoc probability calibration for binary classifiers.

Methods, metrics, diagnostics, and auditable offsetting — numpy-only.
"""

from ._math import expit, logit
from .base import BaseCalibrator
from .parametric import BetaCalibrator, PlattCalibrator, TemperatureCalibrator

__version__ = "0.0.1"

__all__: list[str] = [
    "BaseCalibrator",
    "BetaCalibrator",
    "PlattCalibrator",
    "TemperatureCalibrator",
    "expit",
    "logit",
]
