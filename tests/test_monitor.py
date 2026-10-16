"""Fast tests for probcal.monitor: e-processes, weights, persistence, diagnostics, validation.

Per-grade behaviour lives in test_monitor_grades.py, the drift onset in
test_monitor_onset.py, actions in test_monitor_actions.py and the simulation
gates in test_monitor_sim.py.
"""

import json
import warnings
from dataclasses import replace

import numpy as np
import pytest

from probcal._math import bern_log_lr, expit, logit
from probcal.datasets import make_pd_portfolio
from probcal.monitor import CalibrationMonitor
from probcal.monitor._diagnostics import residual_shape_lr, slope_ci
from probcal.monitor._processes import exponent_weights, plug_in_delta, plug_in_shape


def _batch(n=2000, shift=0.0, slope=1.0, seed=0, event_rate=0.05):
    d = make_pd_portfolio(n=n, event_rate=event_rate, random_state=seed)
    p = d.scores
    true = expit(slope * logit(p) + shift)
    rng = np.random.default_rng(seed + 1000)
    y = (rng.random(n) < true).astype(float)
    return y, p


def _small_batch(n: int = 300, shift: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    p = expit(rng.normal(-2.5, 1.0, n))
    y = (rng.random(n) < expit(logit(p) + shift)).astype(float)
    return y, p


def test_first_batch_plug_ins_are_identity() -> None:
    mon = CalibrationMonitor()
    y, p = _batch(seed=3)
    step = mon.update(y, p, label="b1")
    assert step.delta_hat == 0.0
    assert step.slope_hat == 1.0
    assert step.e_shape == 1.0  # identity plug-in -> every factor is exactly 1
    assert step.n == 2000 and step.label == "b1"


def test_mixture_hand_computed_on_tiny_batch() -> None:
    # Two observations at p = 0.5, y = (1, 0). For grid point d the batch
    # factor is q(1-q)/0.25 with q = sigma(d); the plug-in factor is 1.
    mon = CalibrationMonitor(mixture_grid=(0.5, 1.0))
    step = mon.update(np.array([1.0, 0.0]), np.array([0.5, 0.5]), label="t")
    deltas = np.array([-1.0, -0.5, 0.5, 1.0])
    q = expit(deltas)
    e_mix = float(np.mean(q * (1 - q) / 0.25))
    expected_off = 0.5 * (1.0 + e_mix)
    assert step.e_offset == pytest.approx(expected_off, rel=1e-12)


def test_alarm_and_reoffset_recommendation_under_level_drift() -> None:
    mon = CalibrationMonitor(alpha=0.05)
    fired_at = None
    for k in range(8):
        y, p = _batch(shift=0.8, seed=k)
        step = mon.update(y, p, label=f"m{k}")
        if step.alarm and fired_at is None:
            fired_at = k
    assert fired_at is not None and fired_at <= 4
    rep = mon.report()
    assert rep.alarm_at == f"m{fired_at}"
    assert rep.recommendation == "re-offset"
    assert any("shape" in r for r in rep.reasoning)
    lo, hi = mon.steps_[-1].delta_ci
    assert lo <= 0.8 <= hi  # CS covers the injected offset
    assert lo > 0.0  # and excludes zero after sustained drift


def test_no_alarm_on_calibrated_stream() -> None:
    mon = CalibrationMonitor(alpha=0.05)
    for k in range(12):
        y, p = _batch(shift=0.0, seed=500 + k)
        step = mon.update(y, p, label=f"m{k}")
    assert step.alarm is False
    assert step.p_anytime > 0.05
    rep = mon.report()
    assert rep.recommendation == "none"
    assert rep.onset_label is None
    action = mon.apply_recommendation()
    assert action.kind == "none"
    assert action.offset is None and action.composed is None and action.monitor is None
    assert action.window == ()


def test_predictability_future_batches_do_not_change_past() -> None:
    def run(later_shift: float) -> list[float]:
        mon = CalibrationMonitor()
        out = []
        for k in range(6):
            shift = 0.0 if k < 3 else later_shift
            y, p = _batch(shift=shift, seed=40 + k)
            out.append(mon.update(y, p, label=f"m{k}").e_global)
        return out

    a = run(0.0)
    b = run(2.0)
    assert a[:3] == b[:3]  # bit-identical prefixes
    assert a[3:] != b[3:]


def test_persistence_reproduces_trajectory_bit_for_bit(tmp_path) -> None:
    def grade_batch(k: int):
        y, p = _batch(n=600, shift=0.3, seed=60 + k)
        return y, p, np.array(["A", "B"])[np.arange(len(p)) % 2]

    batches = [grade_batch(k) for k in range(5)]
    mon = CalibrationMonitor(grades=("A", "B"))
    direct = [mon.update(y, p, grade=g, label=f"m{k}") for k, (y, p, g) in enumerate(batches)]

    resumed = []
    mon2 = CalibrationMonitor(grades=("A", "B"))
    for k, (y, p, g) in enumerate(batches):
        resumed.append(mon2.update(y, p, grade=g, label=f"m{k}"))
        path = tmp_path / f"state{k}.json"
        mon2.to_json(path)
        mon2 = CalibrationMonitor.from_json(path)
    assert direct == resumed  # every field, incl. grade CIs and log-e increments
    rep = mon2.report()
    assert list(rep.steps) == direct
    frame = rep.to_frame()  # a DataFrame with pandas, else a list of dicts
    assert len(frame) == 5


def test_monitor_and_action_json_is_strict() -> None:
    # MON-9: nan component e-values and the -inf initial running max used to be
    # written as NaN / -Infinity tokens, which strict JSON parsers reject.
    def _reject(token: str) -> None:
        raise AssertionError(f"non-strict JSON token {token}")

    mon = CalibrationMonitor(components=("offset",))
    assert np.isneginf(mon._max_log_global)
    json.loads(mon.to_json(), parse_constant=_reject)
    for k in range(6):
        y, p = _batch(n=1000, shift=0.8, seed=k)
        mon.update(y, p, label=f"m{k}")
    assert np.isnan(mon.steps_[-1].e_shape)
    json.loads(mon.to_json(), parse_constant=_reject)
    restored = CalibrationMonitor.from_json(mon.to_json())
    assert np.isnan(restored.steps_[-1].e_shape)
    assert restored.to_json() == mon.to_json()
    json.loads(mon.apply_recommendation().to_json(), parse_constant=_reject)

    # MON-14: the reasoning never prints nan for the unmonitored component.
    rep = mon.report()
    assert rep.alarm_at is not None
    assert not any("nan" in line for line in rep.reasoning)
    assert any("not monitored" in line for line in rep.reasoning)


# ---------------------------------------------------------------- weights


def test_exponent_weights_bounded_and_scale_free() -> None:
    w = np.random.default_rng(0).lognormal(0.0, 1.5, 500)
    u = exponent_weights(w)
    assert np.all((u > 0.0) & (u <= 1.0))
    assert np.allclose(exponent_weights(1e4 * w), u, rtol=1e-12)
    big = w >= w.mean()
    assert np.all(u[big] == 1.0)
    assert np.allclose(u[~big], w[~big] / w.mean())


def test_weights_do_not_warn_and_are_scale_free() -> None:
    y, p = _batch(seed=7)
    w = np.ones_like(p)
    w[:10] = 2.5
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a = CalibrationMonitor().update(y, p, sample_weight=w, label="w1")
    b = CalibrationMonitor().update(y, p, sample_weight=7.0 * w, label="w1")
    assert a.e_global == pytest.approx(b.e_global, rel=1e-12)
    assert a.e_global != CalibrationMonitor().update(y, p, label="w1").e_global


def test_constant_weights_reduce_to_the_unweighted_monitor() -> None:
    plain = CalibrationMonitor(grades=("A", "B", "C"), delta_ci_grid=(-2.0, 2.0, 41))
    weighted = CalibrationMonitor(grades=("A", "B", "C"), delta_ci_grid=(-2.0, 2.0, 41))
    for k in range(6):
        y, p = _small_batch(shift=0.5 if k >= 3 else 0.0, seed=k)
        g = np.array(["A", "B", "C"])[np.arange(len(p)) % 3]
        a = plain.update(y, p, grade=g, label=f"b{k}")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            b = weighted.update(y, p, sample_weight=np.full(len(p), 1e4), grade=g, label=f"b{k}")
        assert replace(b, n_events=a.n_events) == a  # n_events is the weighted count


def test_skewed_weight_batch_factor_is_an_e_value() -> None:
    # Under the null, E[prod_i LR_i^u_i] <= 1 for exponents u_i in (0, 1];
    # raw exponents w_i > 1 inflate it above 1.
    rng = np.random.default_rng(0)
    n, reps = 12, 40_000
    p = expit(rng.normal(-1.5, 0.8, n))
    q = expit(logit(p) + 0.7)
    w = rng.lognormal(0.0, 2.0, n) * 5.0
    u = exponent_weights(w)
    y = (rng.random((reps, n)) < p).astype(float)
    terms = y * (np.log(q) - np.log(p)) + (1.0 - y) * (np.log1p(-q) - np.log1p(-p))
    factor = np.exp(terms @ u)
    raw = np.exp(terms @ w)
    se = factor.std() / np.sqrt(reps)
    assert factor.mean() <= 1.0 + 3.0 * se
    assert raw.mean() > 1.5
    assert np.isclose(np.log(factor[0]), bern_log_lr(y[0], p, q, u))


def test_skewed_weight_null_alarm_rate_is_controlled() -> None:
    alarms = 0
    reps = 40
    for r in range(reps):
        rng = np.random.default_rng(r)
        mon = CalibrationMonitor(components=("offset",))
        for _ in range(10):
            p = expit(rng.normal(-3.0, 1.0, 200))
            y = (rng.random(200) < p).astype(float)
            s = mon.update(y, p, sample_weight=1e4 * rng.lognormal(0, 1.5, 200))
        alarms += s.alarm
    assert alarms / reps <= 0.05 + 0.05  # loose: alpha plus Monte Carlo slack


# ---------------------------------------------------------------- diagnostics


def test_residual_shape_lr_invariant_to_weight_scale() -> None:
    y, p = _small_batch(n=2000, shift=0.4, seed=50)
    z = logit(p)
    w = np.random.default_rng(1).uniform(0.5, 2.0, len(p))
    a = residual_shape_lr(z, y, w, 0.3)
    b = residual_shape_lr(z, y, 10.0 * w, 0.3)
    assert a == pytest.approx(b, rel=1e-9)


def test_residual_shape_lr_weighted_null_size_and_unit_weights() -> None:
    # Offset-only truth: the design-effect-corrected statistic exceeds the
    # chi-square(1) 5% bound about 5% of the time, also with skewed weights.
    rng = np.random.default_rng(3)
    hits = 0
    reps = 300
    for _ in range(reps):
        z = rng.normal(-2.0, 1.0, 400)
        y = (rng.random(400) < expit(z + 0.3)).astype(float)
        w = rng.lognormal(0.0, 1.0, 400)
        hits += residual_shape_lr(z, y, w, plug_in_delta(z, y, w)) > 3.841
    assert hits / reps < 0.09
    z = rng.normal(-2.0, 1.0, 400)
    y = (rng.random(400) < expit(z)).astype(float)
    ones = np.ones(400)
    c, a = plug_in_shape(z, y, ones)

    def ll(q: np.ndarray) -> float:
        return float(np.sum(y * np.log(q) + (1.0 - y) * np.log1p(-q)))

    plain = max(0.0, 2.0 * (ll(expit(c + a * z)) - ll(expit(z + 0.1))))
    assert residual_shape_lr(z, y, ones, 0.1) == plain


def test_slope_ci_constant_logits_is_uninformative_without_warnings() -> None:
    z = np.full(50, -2.0)
    y = (np.arange(50) % 5 == 0).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert slope_ci(z, y, np.ones(50)) == (-np.inf, np.inf)


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"components": ("offset", "nope")}, "components"),
        ({"alpha": 1.5}, "alpha"),
        ({"recommendation_window": "nope"}, "recommendation_window"),
        ({"grades": (1, 1.0)}, "duplicate"),
        *[({"plug_in_window": w}, "plug_in_window") for w in (0, -1, True, 1.5)],
        *[({"min_history": v}, "min_history") for v in (-1, 1.5, True)],
        *[
            ({"mixture_grid": g}, "mixture_grid")
            for g in [(), (0.0,), (-0.5, 1.0), (np.inf,), (np.nan,), (True,), ("0.5",)]
        ],
        *[
            ({"delta_ci_grid": g}, "delta_ci_grid")
            for g in [
                (3.0, -3.0, 241),
                (-3.0, 3.0, 0),
                (-3.0, 3.0, 1),
                (-3.0, 3.0, 2.5),
                (0.5, 3.0, 11),
                (-3.0, -0.5, 11),
                (-np.inf, 3.0, 11),
                (-3.0, 3.0),
                (1.0, 1.0, 5),
            ]
        ],
    ],
)
def test_constructor_validation(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        CalibrationMonitor(**kwargs)


def test_edge_grids_and_min_history_zero_are_accepted() -> None:
    CalibrationMonitor(mixture_grid=(0.5,), delta_ci_grid=(0.0, 2.0, 2))
    CalibrationMonitor(mixture_grid=(1, 2), delta_ci_grid=(-1, 0, 3))
    mon = CalibrationMonitor(min_history=0)
    y, p = _batch(n=500, seed=1)
    assert mon.update(y, p).delta_hat == 0.0  # empty past -> identity plug-in


@pytest.mark.parametrize(
    ("y", "p", "kwargs", "match"),
    [
        ([0.0, 2.0], [0.1, 0.2], {}, "0, 1"),
        ([], [], {}, "must not be empty"),  # used to surface as an IndexError
        ([0.0, 1.0], [0.1, 0.2], {"grade": ["A", "Z"]}, "not in the declared grades"),
    ],
)
def test_update_validation_leaves_no_step(y, p, kwargs, match) -> None:
    mon = CalibrationMonitor(grades=("A", "B"))
    with pytest.raises(ValueError, match=match):
        mon.update(np.array(y), np.array(p), **kwargs)
    assert mon.steps_ == []
    # a single-class batch is legal (a quiet month has no defaults)
    mon.update(np.zeros(50), np.full(50, 0.02), label="quiet")
