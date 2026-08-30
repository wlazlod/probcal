"""Anytime-valid calibration monitoring by e-processes.

Theory, validity conditions, and the simulation verification:
``docs/concepts/monitoring.md``. numpy + stdlib only, like the core.
"""

from ._actions import AppliedAction, moc_offset, moc_offset_from_counts
from ._monitor import CalibrationMonitor, MonitorReport, MonitorStep

__all__ = [
    "AppliedAction",
    "CalibrationMonitor",
    "MonitorReport",
    "MonitorStep",
    "moc_offset",
    "moc_offset_from_counts",
]
