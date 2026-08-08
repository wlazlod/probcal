"""Regenerate every figure referenced by docs/concepts/visualization.md.

Deterministic (seeded portfolio, no RNG in plotting) and idempotent:

    uv run python docs/scripts/generate_figures.py

Requires the [viz] extra. Output: docs/concepts/img/*.png at dpi 130.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from probcal import (  # noqa: E402
    BetaCalibrator,
    CalibratorSelector,
    LogitOffset,
    calibration_belt,
    make_pd_portfolio,
    reliability_binned,
    reliability_loess,
)
from probcal.curves import ecce_curve  # noqa: E402
from probcal.metrics import jeffreys_grade_test  # noqa: E402
from probcal.plots import (  # noqa: E402
    plot_belt,
    plot_comparison,
    plot_ecce,
    plot_grade_backtest,
    plot_interval,
    plot_offset_audit,
    plot_reliability,
    plot_selection,
)
from probcal.vennabers import VennAbersCalibrator  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "concepts" / "img"
SAVE = {"dpi": 130, "bbox_inches": "tight"}


def save(artist, name: str) -> None:
    fig = artist if hasattr(artist, "savefig") else artist.figure
    fig.savefig(OUT / name, **SAVE)
    plt.close(fig)
    print(f"wrote {OUT / name}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    port = make_pd_portfolio(n=12000, random_state=42)
    y, scores = port.y, port.scores
    p_cal = BetaCalibrator().fit(scores, y).predict_proba(scores)

    curve = reliability_binned(y, scores)
    smooth = reliability_loess(y, scores)
    save(plot_reliability(curve, smooth=smooth, y=y, p=scores), "reliability_probability.png")
    save(
        plot_reliability(curve, smooth=smooth, scale="logit", y=y, p=scores),
        "reliability_logit.png",
    )

    save(plot_belt(calibration_belt(y, scores)), "belt.png")

    save(
        plot_comparison(
            reliability_binned(y, scores),
            reliability_binned(y, p_cal),
            scale="logit",
            labels=("raw scores", "beta-calibrated"),
        ),
        "comparison.png",
    )

    ivap = VennAbersCalibrator().fit(scores[:2000], y[:2000])
    grid = np.linspace(0.005, 0.5, 120)
    save(plot_interval(ivap.predict_interval(grid), grid), "interval.png")

    sel = CalibratorSelector(cv=4, random_state=42).fit(scores[:3000], y[:3000])
    save(plot_selection(sel.report_), "selection.png")

    save(
        plot_ecce(
            [ecce_curve(y, scores), ecce_curve(y, p_cal)],
            labels=["raw scores", "beta-calibrated"],
        ),
        "ecce.png",
    )

    grade_edges = [0.005, 0.01, 0.02, 0.05]
    grade_names = np.array(["G1", "G2", "G3", "G4", "G5"])
    grades = grade_names[np.searchsorted(grade_edges, p_cal)]
    save(plot_grade_backtest(jeffreys_grade_test(y, p_cal, grades)), "grade_backtest.png")

    offset = LogitOffset(target_mean=float(y.mean())).fit(scores)
    save(plot_offset_audit(offset), "offset_audit.png")


if __name__ == "__main__":
    main()
