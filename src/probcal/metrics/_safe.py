"""Fixed-sample mixture-LR grade e-test (safe Hosmer-Lemeshow analogue).

A fixed-sample audit has no "past" batches to build a predictable plug-in
from (unlike ``probcal.monitor``, which learns its offset/shape plug-ins
from strictly earlier data), so the honest fixed-sample e-value is the
**mixture** likelihood ratio alone: average the Bernoulli likelihood-ratio
factor ``LR_i(sigma(z_i + delta) : p_i)`` (``monitor._processes.bern_log_lr``)
over a fixed, data-independent grid of offsets ``delta``, symmetrized to
``+/-`` the same way ``CalibrationMonitor`` symmetrizes its own mixture
grid. Averages of e-values are e-values, so each grade's mixture average is
itself a valid e-value for that grade's null; grades partition the sample
into disjoint observations, so the product of per-grade e-values is a valid
e-value for the joint null (independent factors). Theory and the full-size
type-I/power simulation table: ``docs/concepts/monitoring.md``.

Naming note: this is named "mixture-LR grade e-test (safe Hosmer-Lemeshow
analogue)", not "the safe Hosmer-Lemeshow test". Henzi, Puke, Dimitriadis &
Ziegel (2024), "A safe Hosmer-Lemeshow test" (*The New England Journal of
Statistics in Data Science* 2(2), 175-189), is cited here as the paper that
motivated building a fixed-sample e-value analogue of Hosmer-Lemeshow, not
as a description of what is implemented: their construction is not
reproduced, and no claim is made that this test matches its power or
optimality properties.
"""

from dataclasses import dataclass

import numpy as np

from .._math import expit, logit
from .._results import Interpretation, _ResultBase
from ..monitor._processes import bern_log_lr, logsumexp
from .scores import _prep


@dataclass(frozen=True)
class HlEResult(_ResultBase):
    """Mixture-LR grade e-test result (:func:`hl_e_test`).

    Attributes
    ----------
    e_value : float
        The test e-value, ``exp(log E)`` with ``log E = sum(log E_g)`` over
        grades -- a product of per-grade e-values, itself a valid e-value
        for the joint null.
    p_value : float
        ``min(1, 1 / e_value)`` -- a valid (if generally conservative)
        p-value derived from the e-value via Markov's inequality.
    grades : tuple of str
        Grade labels, sorted.
    e_grade : numpy.ndarray
        Per-grade e-value, aligned with ``grades``; ``e_value`` is their
        product.
    construction : str
        Always ``"mixture-lr"`` -- recorded so a serialized or logged
        result names its own construction unambiguously.

    Examples
    --------
    >>> import numpy as np
    >>> from probcal.metrics import hl_e_test
    >>> rng = np.random.default_rng(0)
    >>> p = np.full(200, 0.1)
    >>> y = (rng.random(200) < 0.1).astype(float)
    >>> grades = np.array(["A"] * 100 + ["B"] * 100)
    >>> res = hl_e_test(y, p, grades)
    >>> res.construction
    'mixture-lr'
    >>> bool(np.isclose(res.e_value, np.prod(res.e_grade), rtol=1e-9))
    True
    >>> res.p_value == min(1.0, 1.0 / res.e_value)
    True
    """

    e_value: float
    p_value: float
    grades: tuple[str, ...]
    e_grade: np.ndarray
    construction: str

    def interpret(self) -> Interpretation:
        """Read one e-value sentence per grade and the test-level conclusion.

        Returns
        -------
        Interpretation
            ``method="HlETest"``, parameters ``e_value``/``p_value``, one
            per-grade message plus a closing sentence on how to read the
            e-value/p-value pair.

        Examples
        --------
        >>> import numpy as np
        >>> from probcal.metrics import hl_e_test
        >>> rng = np.random.default_rng(0)
        >>> p = np.full(200, 0.1)
        >>> y = (rng.random(200) < 0.1).astype(float)
        >>> grades = np.array(["A"] * 100 + ["B"] * 100)
        >>> interp = hl_e_test(y, p, grades).interpret()
        >>> interp.method
        'HlETest'
        >>> "grade A: e =" in interp.messages[0]
        True
        """
        grade_messages = tuple(
            f"grade {g}: e = {e:.4g}" for g, e in zip(self.grades, self.e_grade, strict=True)
        )
        closing = (
            f"e = {self.e_value:.4g}; e >= 1/alpha rejects H0 (miscalibration) at level alpha "
            f"(Ville/Markov, single fixed-sample look); p = min(1, 1/e) = {self.p_value:.4g} "
            "is a valid p-value derived from the same evidence."
        )
        return Interpretation(
            method="HlETest",
            param_names=("e_value", "p_value"),
            param_values=(self.e_value, self.p_value),
            messages=grade_messages + (closing,),
        )


def hl_e_test(
    y: object,
    p: object,
    grades: object,
    *,
    mixture_grid: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0),
    sample_weight: object = None,
) -> HlEResult:
    """Fixed-sample mixture-LR grade e-test (safe Hosmer-Lemeshow analogue).

    For grade ``g``, with ``z_i = logit(p_i)``:

    ``log E_g = logsumexp_{delta in +/-mixture_grid}(sum_{i in g} log
    LR_i(sigma(z_i + delta) : p_i)) - log(2 * len(mixture_grid))``

    i.e. the log-mean Bernoulli log-likelihood-ratio (``monitor._processes
    .bern_log_lr``) of the grade's observations, averaged over the
    symmetrized offset grid -- the same mixture construction
    ``CalibrationMonitor``'s offset e-process uses, applied once per grade
    with no predictable (plug-in) component, since a fixed sample has no
    strictly-earlier data to learn one from. The test statistic is the
    product across grades, ``log E = sum_g log E_g``, ``e_value = exp(log
    E)``: grades partition the sample into disjoint observations, each
    grade's mixture average is an e-value for that grade's null (an average
    of e-values, each with conditional expectation 1 under H0), and the
    product of e-values over independent (here: disjoint-observation)
    factors is itself an e-value. ``p_value = min(1, 1 / e_value)`` follows
    from Markov's inequality and is a valid (generally conservative)
    p-value.

    Sample weights, when given, enter as exponents on the Bernoulli factors
    (passed straight into ``bern_log_lr``) -- consistent with how
    ``CalibrationMonitor`` and the rest of ``probcal.metrics`` treat
    weights, but note that non-integer weights break the interpretation of
    ``LR`` as a genuine likelihood ratio of independent Bernoulli draws
    (the same caveat ``docs/concepts/monitoring.md`` records for the
    monitor).

    Parameters
    ----------
    y : array_like
        Binary outcomes in ``{0, 1}``; both classes must be present overall
        (grade-level all-zero/all-one subsets are fine).
    p : array_like
        Assigned probabilities (the null) in ``[0, 1]``.
    grades : array_like
        Rating grade label per observation.
    mixture_grid : tuple of float, keyword-only
        Positive logit-scale offsets; symmetrized to ``+/-`` before
        averaging (matching ``CalibrationMonitor(mixture_grid=...)``).
    sample_weight : array_like or None, keyword-only
        Optional non-negative weights, same length as ``y``.

    Returns
    -------
    HlEResult
        Per-grade and combined e-values, the derived p-value, and the
        construction tag.

    Raises
    ------
    ValueError
        If ``y``/``p``/``grades`` are not equal-length 1-D arrays, ``y``
        is not binary or single-class, ``p`` lies outside ``[0, 1]``, or
        ``mixture_grid`` is empty or not 1-D.

    Examples
    --------
    >>> import numpy as np
    >>> from probcal.metrics import hl_e_test
    >>> rng = np.random.default_rng(1)
    >>> p = np.full(400, 0.05)
    >>> y = (rng.random(400) < 0.05).astype(float)
    >>> grades = np.array(["A"] * 200 + ["B"] * 200)
    >>> res = hl_e_test(y, p, grades)
    >>> res.grades
    ('A', 'B')
    >>> res.e_value > 0.0
    True
    """
    y_arr, p_arr, w_arr = _prep(y, p, sample_weight)
    g_arr = np.asarray(grades)
    if g_arr.ndim != 1 or len(g_arr) != len(y_arr):
        raise ValueError("grades must be a 1-D array matching y and p in length")
    grid = np.asarray(mixture_grid, dtype=np.float64)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("mixture_grid must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(grid)):
        raise ValueError("mixture_grid must contain only finite values")

    offsets = np.concatenate([grid, -grid])
    log_norm = float(np.log(2.0 * grid.size))
    z_arr = logit(p_arr)

    g_str = np.array([str(g) for g in g_arr])
    labels = tuple(str(label) for label in sorted(np.unique(g_str)))
    log_e_grade = np.empty(len(labels))
    for i, label in enumerate(labels):
        mask = g_str == label
        y_g, p_g, z_g, w_g = y_arr[mask], p_arr[mask], z_arr[mask], w_arr[mask]
        log_terms = np.array([bern_log_lr(y_g, p_g, expit(z_g + delta), w_g) for delta in offsets])
        log_e_grade[i] = logsumexp(log_terms) - log_norm

    log_e = float(np.sum(log_e_grade))
    e_grade = np.exp(log_e_grade)
    e_value = float(np.exp(log_e))
    p_value = min(1.0, 1.0 / e_value) if e_value > 0.0 else 1.0

    return HlEResult(
        e_value=e_value,
        p_value=p_value,
        grades=labels,
        e_grade=e_grade,
        construction="mixture-lr",
    )
