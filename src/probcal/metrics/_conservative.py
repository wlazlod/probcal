"""Pluto-Tasche one-period most-prudent PDs for ordered rating grades.

Low- and zero-default portfolios (the common case for the best few grades of
a retail or sovereign scorecard) leave the exact binomial/Jeffreys per-grade
tests in ``grade.py`` almost powerless: with zero defaults every posterior or
tail-probability reading is uninformative about the grade's true PD. Pluto &
Tasche (2005) address this by assuming rating-grade monotonicity -- the true
PD cannot decrease from a better grade to a worse one -- and using that
assumption to borrow information from worse grades' (higher-default) data
when bounding a given grade's PD. Theory and the coverage simulation:
``docs/concepts/conservatism.md``.
"""

import warnings
from dataclasses import dataclass

import numpy as np

from .._math import beta_ppf
from .._results import Interpretation, _ResultBase
from .._validation import validate_scores, validate_weights
from ._common import binary_y
from ._deprecation import UNSET, renamed_kwarg
from ._grades import aggregate, grade_labels, is_masterscale, resolve_order


@dataclass(frozen=True, eq=False)
class PlutoTascheResult(_ResultBase):
    """Pluto-Tasche one-period most-prudent PD per rating grade.

    Attributes
    ----------
    grades : tuple of str
        Grade labels, best to worst, in the order given to
        :func:`pluto_tasche` / :func:`pluto_tasche_from_arrays`.
    n : numpy.ndarray
        Own obligor count per grade (weighted sum if fitted from arrays with
        ``sample_weight``).
    d : numpy.ndarray
        Own default count per grade (weighted sum likewise).
    n_pooled : numpy.ndarray
        Obligor count pooled with all worse grades: ``n_pooled[i] = sum(n[i:])``.
    d_pooled : numpy.ndarray
        Default count pooled the same way.
    pd_upper : numpy.ndarray
        Most-prudent PD per grade: the one-sided Clopper-Pearson upper bound
        of the pooled default rate at ``confidence``.
    confidence : float
        Confidence level used for every grade's bound.
    monotonized : bool
        ``True`` only if ``pd_upper`` needed the running-maximum touch-up
        (see below) to stay non-decreasing best to worst. Pooled sets are
        nested (grade ``i``'s pooled set contains grade ``i + 1``'s), so
        for a portfolio whose observed per-grade default rates already
        respect rating order, ``pd_upper`` comes out non-decreasing on its
        own and this flag is ``False``; a noisy grade whose own rate
        exceeds the worse-grade pool it joins can still produce a real
        (not merely floating-point) local dip. ``pd_upper`` is always
        non-decreasing on return either way: the raw bound is replaced by
        its cumulative maximum best to worst (a prudent hull), which never
        *lowers* any grade's bound -- only raises a grade whose raw bound
        fell below a better grade's, never the reverse.
    """

    grades: tuple[str, ...]
    n: np.ndarray
    d: np.ndarray
    n_pooled: np.ndarray
    d_pooled: np.ndarray
    pd_upper: np.ndarray
    confidence: float
    monotonized: bool

    def interpret(self) -> Interpretation:
        """Read one audit sentence per grade: own counts, pooling, and the bound.

        Returns
        -------
        Interpretation
            ``method="PlutoTasche"``, one ``pd_upper.<grade>`` parameter and
            one audit sentence per grade.

        Examples
        --------
        >>> import numpy as np
        >>> from probcal.metrics import pluto_tasche
        >>> res = pluto_tasche(
        ...     np.array([100.0, 400.0, 300.0]),
        ...     np.array([0.0, 0.0, 0.0]),
        ...     confidence=0.9,
        ...     grades=("A", "B", "C"),
        ... )
        >>> msg = res.interpret().messages[0]
        >>> "grade A: 0 defaults among 100 obligors" in msg
        True
        >>> "most-prudent PD at 90% confidence = 0.29%" in msg
        True
        """
        param_names = tuple(f"pd_upper.{g}" for g in self.grades)
        param_values = tuple(float(v) for v in self.pd_upper)
        messages = tuple(
            f"grade {g}: {self.d[i]:g} defaults among {self.n[i]:g} obligors; "
            f"pooled with worse grades (n*={self.n_pooled[i]:g}, d*={self.d_pooled[i]:g}); "
            f"most-prudent PD at {self.confidence:.0%} confidence = {self.pd_upper[i]:.2%}"
            for i, g in enumerate(self.grades)
        )
        return Interpretation(
            method="PlutoTasche",
            param_names=param_names,
            param_values=param_values,
            messages=messages,
        )


def pluto_tasche(
    grade_n: object,
    grade_d: object,
    *,
    confidence: float = 0.9,
    grades: object = None,
) -> PlutoTascheResult:
    """Pluto & Tasche (2005) one-period most-prudent PD, from per-grade counts.

    For grade ``i`` (best to worst, in the order given), pool its own
    obligors and defaults with every worse grade's: ``n*_i = sum(n[i:])``,
    ``d*_i = sum(d[i:])``. The most-prudent PD is the one-sided
    Clopper-Pearson upper bound of the pooled rate,
    ``p`` solving ``I_p(d*_i + 1, n*_i - d*_i) = confidence``
    (``beta_ppf(confidence, d*_i + 1, n*_i - d*_i)``), i.e. the largest PD
    under which observing at most ``d*_i`` defaults in ``n*_i`` obligors
    still has probability ``>= 1 - confidence``. Pooling with worse grades is the
    rating-monotonicity assumption doing its work: a grade's own data alone
    is often uninformative (frequently zero defaults), but the assumption
    that its true PD cannot exceed a worse grade's lets that grade's
    defaults bound this one.

    Parameters
    ----------
    grade_n : array_like
        Obligor count per grade, best to worst. Non-integer (weighted)
        counts are accepted and pass directly into the Beta shape
        parameters below.
    grade_d : array_like
        Default count per grade, same order; ``grade_d[i] <= grade_n[i]``.
    confidence : float, keyword-only
        Confidence level in ``(0, 1)`` for every grade's upper bound.
    grades : sequence of str or None, keyword-only
        Grade labels, best to worst; ``None`` uses ``"1", "2", ..., "K"``.

    Returns
    -------
    PlutoTascheResult
        Per-grade counts, pooled counts, and most-prudent PDs.

    Raises
    ------
    ValueError
        If ``grade_n``/``grade_d`` are not equal-length 1-D arrays, contain
        negative values, have a default count exceeding the obligor count,
        ``confidence`` is not in ``(0, 1)``, ``grades`` does not match the
        count arrays' length, or a grade's pooled obligor count
        (``n*_i``) is zero.

    Examples
    --------
    >>> import numpy as np
    >>> from probcal.metrics import pluto_tasche
    >>> res = pluto_tasche(
    ...     np.array([100.0, 400.0, 300.0]),
    ...     np.array([0.0, 0.0, 0.0]),
    ...     confidence=0.9,
    ...     grades=("A", "B", "C"),
    ... )
    >>> res.grades
    ('A', 'B', 'C')
    >>> np.round(res.pd_upper, 4)
    array([0.0029, 0.0033, 0.0076])
    """
    n = np.asarray(grade_n, dtype=np.float64)
    d = np.asarray(grade_d, dtype=np.float64)
    if n.ndim != 1 or d.ndim != 1 or n.shape != d.shape:
        raise ValueError("grade_n and grade_d must be 1-D arrays of equal length")
    if len(n) == 0:
        raise ValueError("pluto_tasche requires at least one grade")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if np.any(n < 0.0) or np.any(d < 0.0):
        raise ValueError("grade_n and grade_d must be non-negative")
    if np.any(d > n):
        raise ValueError("grade_d cannot exceed grade_n in any grade")

    k = len(n)
    if grades is None:
        grade_labels = tuple(str(i + 1) for i in range(k))
    else:
        grade_labels = tuple(str(g) for g in np.asarray(grades).reshape(-1))
        if len(grade_labels) != k:
            raise ValueError("grades must have the same length as grade_n/grade_d")

    n_pooled = np.cumsum(n[::-1])[::-1]
    d_pooled = np.cumsum(d[::-1])[::-1]
    if np.any(n_pooled == 0.0):
        raise ValueError("pluto_tasche: grade pooled with worse grades has zero obligors (n* == 0)")

    pd_upper = np.empty(k, dtype=np.float64)
    for i in range(k):
        ns, ds = n_pooled[i], d_pooled[i]
        # n_pooled - d_pooled cannot go negative here: d <= n was validated
        # per grade above, and both are cumulative sums over the same
        # (worse-grades-first) order, so the inequality is preserved term
        # by term -- beta_ppf's second shape parameter stays > 0 (b == 0
        # only in the d* == n* case handled explicitly).
        pd_upper[i] = 1.0 if ds == ns else beta_ppf(confidence, ds + 1.0, ns - ds)

    # Pooled sets are nested (grade i's pooled set contains grade i + 1's),
    # so pd_upper comes out non-decreasing already whenever the observed
    # per-grade default rates respect rating order. A noisy grade whose own
    # rate exceeds the worse-grade pool it joins can still produce a real
    # local dip; the most-prudent reading of such a dip is to raise the
    # worse grade's bound up to the better grade's, never to lower the
    # better grade's bound to match -- so the touch-up is the cumulative
    # maximum best to worst (a prudent hull), which never reduces any
    # grade's bound.
    hull = np.maximum.accumulate(pd_upper)
    monotonized = bool(np.any(hull != pd_upper))

    return PlutoTascheResult(
        grades=grade_labels,
        n=n,
        d=d,
        n_pooled=n_pooled,
        d_pooled=d_pooled,
        pd_upper=hull,
        confidence=confidence,
        monotonized=monotonized,
    )


def pluto_tasche_from_arrays(
    grades: object,
    y: object,
    *,
    order: object = None,
    p: object = None,
    confidence: float = 0.9,
    sample_weight: object = None,
) -> PlutoTascheResult:
    """Pluto-Tasche most-prudent PD from observation-level grades and outcomes.

    Convenience wrapper around :func:`pluto_tasche`: aggregates ``y`` by
    ``grades`` into per-grade obligor/default counts (weighted sums when
    ``sample_weight`` is given) in the explicit ``order``, then applies the
    same pooling and bound.

    Parameters
    ----------
    grades : array_like or Masterscale
        Rating grade label per observation, or a :class:`probcal.Masterscale`
        that assigns them from ``p``.
    y : array_like
        Binary outcomes in ``{0, 1}``. Unlike most probcal metrics, an
        all-zero ``y`` is accepted -- Pluto-Tasche is built for exactly that
        case.
    order : sequence of str or None, keyword-only
        Explicit best-to-worst grade order; must match the unique labels in
        ``grades`` exactly (same set, same count, any order raises if it
        does not correspond to a permutation of the unique labels). Required
        for a label array; defaults to the masterscale's own order otherwise.
    p : array_like or None, keyword-only
        Predicted probabilities, required when ``grades`` is a
        ``Masterscale`` (labels are assigned from it); ignored otherwise.
    confidence : float, keyword-only
        Confidence level in ``(0, 1)`` for every grade's upper bound.
    sample_weight : array_like or None, keyword-only
        Optional positive weights, same length as ``y``; per-grade counts
        become weighted sums, i.e. weights are read literally as frequency
        counts (integer weights equal row duplication) -- an exception to
        the relative-weight convention of :mod:`probcal.metrics._common`.

    Returns
    -------
    PlutoTascheResult
        Per-grade counts, pooled counts, and most-prudent PDs.

    Raises
    ------
    ValueError
        If ``grades`` and ``y`` are not equal-length 1-D arrays, ``y``
        contains values outside ``{0, 1}``, or ``order`` does not match the
        unique labels in ``grades``.

    Examples
    --------
    >>> import numpy as np
    >>> from probcal.metrics import pluto_tasche_from_arrays
    >>> grades = np.array(["A"] * 100 + ["B"] * 400 + ["C"] * 300)
    >>> y = np.zeros(800)
    >>> res = pluto_tasche_from_arrays(grades, y, order=("A", "B", "C"))
    >>> res.n
    array([100., 400., 300.])
    """
    y_arr = binary_y(y, require_both_classes=False)
    if is_masterscale(grades):
        if p is None:
            raise ValueError(
                "p is required when grades is a Masterscale (labels are assigned from p)"
            )
        p_arr = validate_scores(p, name="p")
        if len(p_arr) != len(y_arr):
            raise ValueError("y and p must have equal length")
    else:
        if order is None:
            raise ValueError("order is required when grades is a label array")
        p_arr = np.empty(0)
    g_str, default_order = grade_labels(grades, p_arr, len(y_arr))
    w_arr = validate_weights(sample_weight, len(y_arr))
    order_t = resolve_order(g_str, order, default_order)
    n, d, _, _ = aggregate(g_str, order_t, y_arr, w=w_arr)

    return pluto_tasche(n, d, confidence=confidence, grades=order_t)


def jeffreys_upper_bands(
    y: object,
    p: object,
    grades: object,
    *,
    confidence: float = 0.9,
    order: object = None,
    level: object = UNSET,
) -> dict[str, tuple[float, float]]:
    """Jeffreys per-grade upper bounds as a contiguous masterscale band table.

    For grade ``i`` (best to worst, in ``order``): ``hi_i`` is the same
    one-sided Jeffreys posterior upper bound ``jeffreys_grade_test`` reports
    as its own-grade display interval, ``beta_ppf(confidence, k_i + 0.5,
    n_i - k_i + 0.5)`` under a ``Beta(k_i + 0.5, n_i - k_i + 0.5)`` posterior
    on grade ``i``'s own default rate; ``lo_i`` is the previous grade's
    ``hi`` (``0.0`` for the best grade), so the bands are contiguous by
    construction: ``(lo_0, hi_0), (hi_0, hi_1), (hi_1, hi_2), ...``. Unlike
    :func:`pluto_tasche`, each grade's bound uses only its own counts (no
    pooling across grades), so a zero-default grade still gets a strictly
    positive ``hi`` from the Jeffreys prior alone.

    The raw ``hi`` sequence need not come out non-decreasing (a noisy grade
    can have a smaller posterior upper bound than a better grade), which
    would make the bands overlap or invert. It is replaced by its running
    maximum best to worst (``np.maximum.accumulate``, the same prudent hull
    :func:`pluto_tasche` uses), which never lowers any grade's bound -- it
    only raises a worse grade's bound to a better grade's -- with a
    ``UserWarning`` emitted only when that adjustment changed a value. (0.3.x
    used a size-weighted PAVA fit, which could *lower* a grade's upper bound
    and give a grade a zero-width band.)

    The resulting ``{grade: (lo, hi)}`` table is exactly the shape
    :func:`probcal.thresholds.calibrated_bands_to_raw` consumes to translate
    a masterscale defined on calibrated PD into raw-score intervals.

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``. Unlike most probcal metrics (but like
        :func:`pluto_tasche_from_arrays`), an all-zero ``y`` is accepted --
        the low- and zero-default portfolio is this module's motivating
        case.
    p : array_like
        Predicted probabilities in ``[0, 1]`` (only used, alongside
        ``grades``, to determine the default best-to-worst ``order`` when
        ``order`` is not given).
    grades : array_like or Masterscale
        Rating grade label per observation, or a :class:`probcal.Masterscale`
        that assigns them from ``p`` and supplies the default ``order``.
    confidence : float, keyword-only
        Confidence level in ``(0, 1)`` for every grade's Jeffreys upper
        bound.
    order : sequence of str or None, keyword-only
        Explicit best-to-worst grade order; must match the unique labels in
        ``grades`` exactly. ``None`` (default) uses the masterscale's order,
        or orders label-array grades by their mean ``p``, ascending (lowest
        predicted PD first).
    level : float, keyword-only
        Deprecated spelling of ``confidence`` (removed in 0.4.0).

    Returns
    -------
    dict[str, tuple[float, float]]
        Mapping of grade label to ``(lo, hi)`` calibrated-probability bounds,
        contiguous and non-decreasing best to worst.

    Raises
    ------
    ValueError
        If ``y``/``p``/``grades`` are not equal-length 1-D arrays,
        ``confidence`` is not in ``(0, 1)``, or ``order`` does not match the unique grade
        labels.

    Examples
    --------
    >>> import numpy as np
    >>> from probcal.metrics import jeffreys_upper_bands
    >>> grades = np.array(["A"] * 100 + ["B"] * 100)
    >>> y = np.array([0.0] * 100 + [1.0] * 5 + [0.0] * 95)
    >>> p = np.array([0.01] * 100 + [0.05] * 100)
    >>> bands = jeffreys_upper_bands(y, p, grades, confidence=0.9)
    >>> bands["A"][0]
    0.0
    >>> bands["A"][1] < bands["B"][1]
    True
    """
    conf = renamed_kwarg("jeffreys_upper_bands", "level", "confidence", level, confidence, 0.9)
    y_arr = binary_y(y, require_both_classes=False)
    p_arr = validate_scores(p, name="p")
    if len(p_arr) != len(y_arr):
        raise ValueError("y and p must have equal length")
    g_arr, default_order = grade_labels(grades, p_arr, len(y_arr))
    if not 0.0 < float(conf) < 1.0:  # type: ignore[arg-type]
        raise ValueError("confidence must lie in (0, 1)")

    if order is None and default_order is None:
        sorted_labels = resolve_order(g_arr, None, None)
        _, _, mean_p, _ = aggregate(g_arr, sorted_labels, y_arr, p=p_arr)
        assert mean_p is not None
        order_t = tuple(sorted_labels[i] for i in np.argsort(mean_p, kind="stable"))
    else:
        order_t = resolve_order(g_arr, order, default_order)
    n, k, _, _ = aggregate(g_arr, order_t, y_arr)

    hi_raw = np.array(
        [beta_ppf(float(conf), k[i] + 0.5, n[i] - k[i] + 0.5) for i in range(len(order_t))]  # type: ignore[arg-type]
    )
    hi = np.maximum.accumulate(hi_raw)
    if np.any(hi != hi_raw):
        warnings.warn(
            "jeffreys_upper_bands: raised upper bounds to the running maximum to keep "
            "the bands non-decreasing best to worst",
            UserWarning,
            stacklevel=2,
        )

    lo = np.concatenate(([0.0], hi[:-1]))
    return {str(label): (float(lo[i]), float(hi[i])) for i, label in enumerate(order_t)}
