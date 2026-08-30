"""The drift scenario shared by the e-process figure and the sample report.

One deployed beta calibration watched over twelve monthly cohorts. Three
business segments sit at different levels throughout (retail below the
deployed map, corporate above it, sme on it); on top of that, from month
seven the whole portfolio's true PD is the deployed forecast shifted by
+0.6 log-odds — a sustained central-tendency deterioration the monitor is
meant to catch. Everything is seeded, so ``generate_figures.py`` and
``generate_sample_report.py`` render the same trajectory.
"""

from dataclasses import dataclass

import numpy as np

from probcal import BetaCalibrator, expit, logit, make_pd_portfolio
from probcal.monitor import CalibrationMonitor

GRADE_EDGES = (0.005, 0.01, 0.02, 0.05)
GRADE_NAMES = np.array(["G1", "G2", "G3", "G4", "G5"])
SEGMENT_NAMES = np.array(["retail", "sme", "corporate"])
# Segment levels as deviations from the deployed map — retail below it,
# corporate above, sme between — then centered so the pooled portfolio is
# still calibrated before the drift starts. Without the centering the spread
# alone is a portfolio-level error the monitor would (correctly) alarm on in
# the first batches, hiding the drift the scenario is actually about.
_SEGMENT_SPREAD = np.array([-0.4, 0.0, 0.45])
_LEVEL_CENTERING = 0.07
SEGMENT_SHIFT = _SEGMENT_SPREAD - _LEVEL_CENTERING


@dataclass(frozen=True)
class DriftScenario:
    """A fitted calibrator, the monitored cohorts, and the monitor itself."""

    calibrator: BetaCalibrator
    y: np.ndarray
    p: np.ndarray
    grades: np.ndarray
    segments: np.ndarray
    monitor: CalibrationMonitor


def _grades(p: np.ndarray) -> np.ndarray:
    return GRADE_NAMES[np.searchsorted(GRADE_EDGES, p)]


def build(
    *,
    n_batches: int = 12,
    batch_size: int = 2500,
    onset: int = 6,
    drift_logit: float = 0.6,
    seed: int = 20260829,
) -> DriftScenario:
    """Fit the deployed map, then walk `n_batches` monthly cohorts past it."""
    cal_portfolio = make_pd_portfolio(n=8000, random_state=0)
    calibrator = BetaCalibrator().fit(cal_portfolio.scores, cal_portfolio.y)

    monitor = CalibrationMonitor(alpha=0.05)
    rng = np.random.default_rng(seed)
    # Segments draw from their own stream, so changing the segment structure
    # never perturbs the outcome draw (and with it the monitor's trajectory).
    seg_rng = np.random.default_rng(seed + 1)
    ys, ps, gs, segs = [], [], [], []
    for k in range(n_batches):
        cohort = make_pd_portfolio(n=batch_size, random_state=100 + k)
        p_batch = calibrator.predict_proba(cohort.scores)
        segment_idx = seg_rng.integers(0, SEGMENT_NAMES.size, size=batch_size)
        drift = drift_logit if k >= onset else 0.0
        true_pd = expit(logit(p_batch) + drift + SEGMENT_SHIFT[segment_idx])
        y_batch = (rng.random(batch_size) < true_pd).astype(float)
        grade_batch = _grades(p_batch)
        monitor.update(y_batch, p_batch, grade=grade_batch, label=f"m{k + 1:02d}")
        ys.append(y_batch)
        ps.append(p_batch)
        gs.append(grade_batch)
        segs.append(SEGMENT_NAMES[segment_idx])

    return DriftScenario(
        calibrator,
        np.concatenate(ys),
        np.concatenate(ps),
        np.concatenate(gs),
        np.concatenate(segs),
        monitor,
    )
