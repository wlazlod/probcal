"""probcal: universal post-hoc probability calibration for binary classifiers.

Methods, metrics, diagnostics, and auditable offsetting — numpy-only.
"""

from ._math import expit, logit
from .base import BaseCalibrator
from .isotonic import CenteredIsotonicCalibrator, IsotonicCalibrator
from .parametric import BetaCalibrator, PlattCalibrator, TemperatureCalibrator
from .vennabers import CrossVennAbersCalibrator, VennAbersCalibrator

__version__ = "0.0.1"

__all__: list[str] = [
    "BaseCalibrator",
    "BetaCalibrator",
    "CenteredIsotonicCalibrator",
    "CrossVennAbersCalibrator",
    "IsotonicCalibrator",
    "PlattCalibrator",
    "TemperatureCalibrator",
    "VennAbersCalibrator",
    "expit",
    "logit",
]
