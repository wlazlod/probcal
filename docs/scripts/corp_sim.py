"""Coverage simulation for probcal.curves.corp_reliability bands (spec V1).

Usage: ``uv run python docs/scripts/corp_sim.py``

For each run, draws ``p = expit(N(-2, 1.2))`` and ``y ~ Bernoulli(p)``, fits
the consistency band via ``probcal._corp.corp_bands``, and checks whether the
*empirical* CORP fit (evaluated on the band grid with ``eval_step``) lies
inside the band — pointwise (mean over grid points) and uniformly (every
grid point). ``tests/test_corp_sim.py`` cross-checks this at a reduced size
(``n=1000, runs=60, n_resamples=100``) and enforces the pointwise gate only;
the full-size table printed here is pasted into ``docs/concepts/corp.md``.

Gate (spec V1, full size): pointwise coverage at level ``L`` is close to
``L`` (bands are pointwise, not simultaneous); uniform coverage is lower and
reported, not gated.
"""

import numpy as np

from probcal._corp import corp_bands, corp_fit, eval_step
from probcal._math import expit


def _draw(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = expit(rng.normal(-2.0, 1.2, n))
    y = (rng.random(n) < p).astype(float)
    return y, p


def coverage(n: int, runs: int, n_resamples: int, level: float, seed: int) -> dict:
    """Fraction of runs whose empirical CORP fit lies inside its consistency band."""
    pointwise = np.empty(runs)
    uniform = np.empty(runs, dtype=bool)
    for r in range(runs):
        y, p = _draw(n, seed=seed * 1_000_003 + r)
        w = np.ones(n)
        grid, lo, hi = corp_bands(y, p, w, "consistency", level, n_resamples, seed + r)
        block_lo, block_hi, block_level, _, _ = corp_fit(y, p, w)
        fit = eval_step(block_lo, block_hi, block_level, grid)
        inside = (fit >= lo) & (fit <= hi)
        pointwise[r] = inside.mean()
        uniform[r] = bool(inside.all())
    return {
        "pointwise_coverage": float(pointwise.mean()),
        "uniform_coverage": float(uniform.mean()),
    }


def main() -> None:
    rows = []
    for n in (1000, 5000):
        for level in (0.8, 0.9):
            out = coverage(n=n, runs=500, n_resamples=100, level=level, seed=0)
            rows.append(
                (
                    n,
                    level,
                    f"{out['pointwise_coverage']:.4f}",
                    f"{out['uniform_coverage']:.4f}",
                    f"pointwise >= {level - 0.05:.2f}",
                )
            )

    print("| n | level | pointwise coverage | uniform coverage | gate |")
    print("|---|-------|---------------------|-------------------|------|")
    for n, level, pw, uc, gate in rows:
        print(f"| {n} | {level} | {pw} | {uc} | {gate} |")


if __name__ == "__main__":
    main()
