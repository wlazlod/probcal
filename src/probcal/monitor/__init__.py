"""Anytime-valid calibration monitoring by e-processes (spec W7/W8).

Theory, validity conditions, and the simulation verification:
``docs/concepts/monitoring.md``. numpy + stdlib only, like the core.
"""

from ._monitor import CalibrationMonitor, MonitorReport, MonitorStep

__all__ = ["CalibrationMonitor", "MonitorReport", "MonitorStep"]
