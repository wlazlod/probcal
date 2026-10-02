"""Regenerate the committed sample validation report and its thumbnail.

Deterministic and idempotent (apart from the report's single
``Generated ... UTC`` line):

    uv run python docs/scripts/generate_sample_report.py

One `validation_report` call with every optional section switched on, over
the drift scenario in ``_drift_scenario.py`` — the same twelve cohorts the
e-process figure plots. Requires the [viz] extra. Outputs:
``docs/assets/sample_validation_report.html`` and its first-screen
thumbnail ``docs/assets/sample_validation_report.png``.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from probcal.curves import corp_reliability  # noqa: E402
from probcal.plots import plot_corp  # noqa: E402
from probcal.report import validation_report  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _drift_scenario import build as build_drift_scenario  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "assets"
TITLE = "probcal validation report — sample portfolio"
N_BOOT = 200
SEED = 42


def thumbnail(y: np.ndarray, p: np.ndarray, path: Path) -> None:
    """Redraw the report's first screen: the header block and the CORP diagram.

    The report itself is HTML; rasterizing it would need a browser, so the
    thumbnail re-renders the same header numbers and the same first figure
    (``plot_corp`` on the same ``corp_reliability`` call) with matplotlib. It
    is a reconstruction, not a screenshot, so it carries no version string or
    timestamp it would have to be kept in sync with.
    """
    corp = corp_reliability(y, p, n_resamples=N_BOOT, random_state=SEED)

    fig = plt.figure(figsize=(7.5, 8.2))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 3.0], hspace=0.32)

    head = fig.add_subplot(grid[0])
    head.axis("off")
    head.text(0, 1.0, TITLE, fontsize=15, fontweight="bold", va="top")
    head.text(0, 0.76, "Portfolio summary", fontsize=11, fontweight="bold", va="top")
    rows = [
        ("n", f"{len(y):,}"),
        ("events", f"{int(np.sum(y)):,}"),
        ("event rate", f"{float(np.mean(y)):.4%}"),
        ("mean predicted probability", f"{float(np.mean(p)):.4%}"),
    ]
    for i, (key, value) in enumerate(rows):
        head.text(0, 0.58 - 0.13 * i, key, fontsize=9, color="0.35", va="top")
        head.text(0.42, 0.58 - 0.13 * i, value, fontsize=9, va="top")
    head.text(0, -0.06, "Reliability", fontsize=11, fontweight="bold", va="top")

    plot_corp(corp, ax=fig.add_subplot(grid[1]))
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenario = build_drift_scenario()

    html_path = OUT / "sample_validation_report.html"
    validation_report(
        scenario.y,
        scenario.p,
        calibrator=scenario.calibrator,
        monitor=scenario.monitor,
        grades=scenario.grades,
        by=scenario.segments,
        title=TITLE,
        n_boot=N_BOOT,
        seed=SEED,
        path=html_path,
    )
    print(f"wrote {html_path}")

    thumbnail(scenario.y, scenario.p, OUT / "sample_validation_report.png")


if __name__ == "__main__":
    main()
