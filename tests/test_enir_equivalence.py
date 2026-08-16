"""Equivalence gate for the W8 heap-scheduled ENIR path rewrite (task-8-brief.md):
the new ``ENIRCalibrator._path_fit`` must reproduce the frozen v0.1.2 reference
(``tests/_enir_reference.py``) on ``path_lambdas_``, ``path_solutions_``, predictions,
and ``weights_``.

Accepted fp-window divergence (documented per the brief): the reference recomputes
*all* pair violation states at every breakpoint, while the engine refreshes only the
pairs touched by the most recent merge. The two can disagree only when an untouched
pair's gap drifts through ``(0, 1e-15]`` between breakpoints -- a measure-zero event
in exact arithmetic that the 20 seeded datasets below arbitrate empirically. Any
other divergence (wrong merge order, wrong level count, wrong weights) is a bug.
"""

import numpy as np

from _enir_reference import reference_aggregate, reference_path, reference_weights
from probcal.bayesian import ENIRCalibrator


def test_reference_matches_unmodified_class() -> None:
    """Sanity check for the frozen reference itself, run before the engine rewrite:
    ``reference_path``/``reference_weights`` must reproduce the current (v0.1.2)
    ``ENIRCalibrator`` fitted attributes exactly on one small seeded dataset."""
    rng = np.random.default_rng(0)
    n = 60
    s = rng.uniform(0.0, 1.0, n)
    y = (rng.random(n) < s).astype(float)

    cal = ENIRCalibrator().fit(s, y)

    s_u, y_u, w_u = reference_aggregate(s, y, np.ones(n))
    np.testing.assert_allclose(s_u, cal._x)
    lambdas, solutions = reference_path(y_u, w_u)
    np.testing.assert_allclose(np.asarray(lambdas), cal.path_lambdas_)
    np.testing.assert_allclose(np.asarray(solutions), cal.path_solutions_)
    weights = reference_weights(y_u, w_u, solutions)
    np.testing.assert_allclose(weights, cal.weights_)
