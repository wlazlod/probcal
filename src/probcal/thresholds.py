"""Calibrated-to-raw interval and masterscale-band mapping.

Thin functional wrappers over the calibrators' ``interval_inverse`` protocol:
numpy-only, arrays and floats, no knowledge of any consumer. The
canonical rating-grade workflow — masterscale bands defined on calibrated PD,
translated once per recalibration into raw-score intervals — is
``calibrated_bands_to_raw``; its output plugs directly into band-style raw
targets of a counterfactual engine. ``build_masterscale`` runs the other way:
it designs the bands from data.
"""

import numpy as np

from ._validation import validate_binary_y, validate_scores, validate_weights
from .masterscale import Masterscale


def calibrated_interval_to_raw(
    calibrator: object,
    lo: float,
    hi: float,
    *,
    space: str = "probability",
    buffer_logit: float = 0.0,
) -> tuple[float, float]:
    """Translate one calibrated-probability interval into raw-score bounds.

    Parameters
    ----------
    calibrator : fitted calibrator
        Any object implementing the duck-typed protocol
        ``interval_inverse(lo, hi, *, space, buffer_logit)`` with
        ``is_monotone_``.
    lo, hi : float
        Calibrated bounds; ``lo=0`` / ``hi=1`` map to the full raw range.
    space : {"probability", "logit"}
        Scale of the returned bounds.
    buffer_logit : float
        Robustness margin applied in logit space before inversion.

    Returns
    -------
    tuple of float
        ``(raw_lo, raw_hi)`` bounds, on the scale requested by ``space``.
    """
    return calibrator.interval_inverse(lo, hi, space=space, buffer_logit=buffer_logit)  # type: ignore[attr-defined]


def calibrated_bands_to_raw(
    calibrator: object,
    bands: object,
    *,
    space: str = "probability",
    buffer_logit: float = 0.0,
) -> dict:
    """Translate a masterscale ``{grade: (lo, hi)}`` on calibrated PD to raw intervals.

    Grade edges are policy artifacts that outlive model versions; this
    translation is what changes when the calibrator is refitted.

    Parameters
    ----------
    calibrator : fitted calibrator
        Any object implementing the duck-typed protocol
        ``interval_inverse(lo, hi, *, space, buffer_logit)`` with
        ``is_monotone_``.
    bands : dict or Masterscale
        Mapping of grade label to ``(lo, hi)`` calibrated-probability bounds,
        or a :class:`probcal.Masterscale` (its ``bands`` are read). Bands are
        inverted as closed intervals; the scale's own ``assign`` is half-open
        (``lo <= p < hi``), which differs only at a shared edge.
    space : {"probability", "logit"}, keyword-only
        Scale of the returned bounds.
    buffer_logit : float, keyword-only
        Robustness margin applied in logit space before inversion.

    Returns
    -------
    dict
        Mapping of grade label to ``(raw_lo, raw_hi)`` bounds, on the scale
        requested by ``space``.
    """
    if hasattr(bands, "bands"):  # a Masterscale
        bands = bands.bands  # type: ignore[attr-defined]
    return {
        grade: calibrated_interval_to_raw(
            calibrator, lo, hi, space=space, buffer_logit=buffer_logit
        )
        for grade, (lo, hi) in bands.items()  # type: ignore[attr-defined]
    }


# ------------------------------------------------------------ build_masterscale


def _prebin(
    p: np.ndarray, y: np.ndarray, w: np.ndarray, prebins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equal-mass pre-bins of sorted ``p`` whose boundaries are observed values.

    Returns per-pre-bin weighted counts ``W``, weighted events ``E``, and the
    cut values ``cuts`` (length ``len(W) - 1``): pre-bin ``j`` is exactly
    ``{cuts[j-1] <= p < cuts[j]}``, the same rule :meth:`Masterscale.assign`
    applies, so any run of consecutive pre-bins is a valid grade.
    """
    order = np.argsort(p, kind="stable")
    ps, ys, ws = p[order], y[order], w[order]
    n = len(ps)
    ranks = (np.arange(1, prebins) * n) // prebins
    cuts = np.unique(ps[ranks])
    cuts = cuts[cuts > ps[0]]  # a cut at the minimum would make an empty first pre-bin
    idx = np.searchsorted(cuts, ps, side="right")
    k = len(cuts) + 1
    W = np.bincount(idx, weights=ws, minlength=k).astype(np.float64)
    E = np.bincount(idx, weights=ws * ys, minlength=k).astype(np.float64)
    return W, E, cuts


def _segment_objective_vec(
    w: np.ndarray, e: np.ndarray, w_total: float, objective: str, target: float | None
) -> np.ndarray:
    """Objective contribution of grades with weights ``w`` and events ``e`` (vectorized)."""
    if objective == "likelihood":
        out = np.zeros_like(w, dtype=np.float64)
        inner = (w > 0) & (e > 0) & (e < w)
        r = e[inner] / w[inner]
        out[inner] = e[inner] * np.log(r) + (w[inner] - e[inner]) * np.log1p(-r)
        return out
    return -((w / w_total - float(target)) ** 2)  # type: ignore[arg-type]


def _segment_objective(
    w: float, e: float, w_total: float, objective: str, target: float | None
) -> float:
    """Scalar form of :func:`_segment_objective_vec` (used by the brute-force test)."""
    return float(
        _segment_objective_vec(np.array([w]), np.array([e]), w_total, objective, target)[0]
    )


def _solve(
    W: np.ndarray,
    E: np.ndarray,
    k: int,
    objective: str,
    target: np.ndarray | None,
    min_count: float,
    min_events: float,
) -> tuple[float, list[int]] | None:
    """Exact DP: best partition of the pre-bins into ``k`` consecutive grades.

    ``f[g][j]`` is the best objective of ``g`` grades covering pre-bins
    ``[0, j)``; segment costs come from prefix sums. Complexity
    ``O(k * B^2)`` for ``B`` pre-bins. Returns ``(objective_value,
    cut_positions)`` or ``None`` when no partition satisfies the floors.
    """
    B = len(W)
    cw = np.concatenate([[0.0], np.cumsum(W)])
    ce = np.concatenate([[0.0], np.cumsum(E)])
    w_total = float(cw[-1])
    i_idx, j_idx = np.triu_indices(B + 1, k=1)
    seg_w = cw[j_idx] - cw[i_idx]
    seg_e = ce[j_idx] - ce[i_idx]
    feasible = (seg_w >= min_count - 1e-12) & (seg_e >= min_events - 1e-12) & (seg_w > 0)

    f = np.full((k + 1, B + 1), -np.inf)
    arg = np.zeros((k + 1, B + 1), dtype=np.int64)
    f[0, 0] = 0.0
    for g in range(1, k + 1):
        t = None if target is None else float(target[g - 1])
        vals = _segment_objective_vec(seg_w, seg_e, w_total, objective, t)
        cost = np.full((B + 1, B + 1), -np.inf)
        cost[i_idx, j_idx] = np.where(feasible, vals, -np.inf)
        cand = f[g - 1][:, None] + cost  # cand[i, j] = f[g-1][i] + cost of segment [i, j)
        arg[g] = np.argmax(cand, axis=0)
        f[g] = cand[arg[g], np.arange(B + 1)]
    if not np.isfinite(f[k, B]):
        return None
    cuts: list[int] = []
    j = B
    for g in range(k, 1, -1):
        j = int(arg[g, j])
        cuts.append(j)
    return float(f[k, B]), sorted(cuts)


def build_masterscale(
    y: object,
    p: object,
    *,
    n_grades: int,
    min_count: float = 0,
    min_events: float = 0,
    objective: str = "likelihood",
    target_shares: object = None,
    prebins: int = 512,
    sample_weight: object = None,
    names: object = None,
) -> Masterscale:
    """Design a masterscale from data by exact dynamic programming.

    ``p`` is sorted and cut into ``prebins`` equal-mass pre-bins whose
    boundaries are observed values (ties merge pre-bins). Grades are runs of
    consecutive pre-bins, so the returned :class:`Masterscale` carries exact
    edges and its half-open ``assign`` reproduces the partition the optimizer
    scored. Complexity ``O(n_grades * prebins^2)`` after an ``O(n log n)``
    sort; at the default 512 pre-bins that is well under a second.

    Parameters
    ----------
    y, p : array_like
        Outcomes and calibrated probabilities.
    n_grades : int, keyword-only
        Number of grades.
    min_count, min_events : float, keyword-only
        Floors per grade on the (weighted) observation and event counts.
    objective : {"likelihood", "target_shares"}, keyword-only
        ``"likelihood"`` maximizes the grade-level binomial log-likelihood
        (equivalently, minimizes within-grade PD heterogeneity);
        ``"target_shares"`` minimizes the squared deviation of each grade's
        share of the sample from ``target_shares``.
    target_shares : sequence of float or None, keyword-only
        Required with ``objective="target_shares"``: one share per grade,
        best to worst, summing to 1.
    prebins : int, keyword-only
        Pre-bin count (an upper bound on the number of candidate edges).
    sample_weight : array_like or None, keyword-only
        Weights; counts become weighted sums.
    names : sequence of str or None, keyword-only
        Grade names best to worst; ``None`` gives ``"G1"``, ``"G2"``, ...

    Returns
    -------
    Masterscale
        With ``provenance`` recording the objective, its value, the pre-bin
        counts, the floors, and which floors bind at the optimum.

    Raises
    ------
    ValueError
        If no partition satisfies the floors (the message names the binding
        floor), or on an invalid ``objective``, ``target_shares``, or ``names``.
    """
    y_arr = validate_binary_y(y)
    p_arr = validate_scores(p, name="p")
    if len(y_arr) != len(p_arr):
        raise ValueError("y and p must have equal length")
    w_arr = validate_weights(sample_weight, len(p_arr))
    if n_grades < 1:
        raise ValueError("n_grades must be >= 1")
    if objective not in ("likelihood", "target_shares"):
        raise ValueError(f'objective must be "likelihood" or "target_shares", got {objective!r}')
    target: np.ndarray | None = None
    if objective == "target_shares":
        if target_shares is None:
            raise ValueError('objective="target_shares" requires target_shares')
        target = np.asarray(target_shares, dtype=np.float64).reshape(-1)
        if len(target) != n_grades or np.any(target < 0):
            raise ValueError(f"target_shares must hold {n_grades} non-negative shares")
        if abs(float(target.sum()) - 1.0) > 1e-9:
            raise ValueError("target_shares must sum to 1")
    labels = None if names is None else [str(x) for x in list(names)]  # type: ignore[call-overload]
    if labels is not None and len(labels) != n_grades:
        raise ValueError(f"names must have {n_grades} entries, got {len(labels)}")

    W, E, cuts = _prebin(p_arr, y_arr, w_arr, int(prebins))
    mc, me = float(min_count), float(min_events)
    solved = _solve(W, E, n_grades, objective, target, mc, me)
    if solved is None:
        binding = []
        if _solve(W, E, n_grades, objective, target, mc, 0.0) is not None:
            binding.append("min_events")
        if _solve(W, E, n_grades, objective, target, 0.0, me) is not None:
            binding.append("min_count")
        if not binding:
            binding = (
                ["min_count", "min_events"]
                if (mc or me)
                else ["n_grades (more grades than distinct pre-bins)"]
            )
        raise ValueError(
            f"no {n_grades}-grade partition satisfies the floors; binding: {', '.join(binding)} "
            f"(min_count={min_count}, min_events={min_events}, {len(W)} pre-bins)"
        )
    value, cut_pos = solved
    edges = [float(cuts[j - 1]) for j in cut_pos]
    bounds = list(zip([0, *cut_pos], [*cut_pos, len(W)], strict=True))
    seg_n = np.array([W[a:b].sum() for a, b in bounds])
    seg_e = np.array([E[a:b].sum() for a, b in bounds])
    binding_at_opt = []
    if mc and np.any(np.isclose(seg_n, mc)):
        binding_at_opt.append("min_count")
    if me and np.any(np.isclose(seg_e, me)):
        binding_at_opt.append("min_events")
    provenance = {
        "objective": objective,
        "objective_value": value,
        "prebins": int(prebins),
        "prebins_effective": int(len(W)),
        "n_grades": int(n_grades),
        "min_count": mc,
        "min_events": me,
        "target_shares": None if target is None else [float(x) for x in target],
        "binding": binding_at_opt,
    }
    return Masterscale(Masterscale.from_edges(edges, names=labels).bands, provenance=provenance)
