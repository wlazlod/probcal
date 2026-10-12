"""Regression tests for the 0.3.5 monitor fixes.

Weighted e-processes, legacy grade-key renaming, drift-onset placement,
element-wise grade keys, weight-scale invariance of the MoC count offset,
and constructor/update validation.
"""

import json
import warnings

import numpy as np
import pytest

from probcal._math import bern_log_lr, beta_ppf, expit, logit
from probcal.monitor import CalibrationMonitor, moc_offset_from_counts
from probcal.monitor._diagnostics import residual_shape_lr, slope_ci
from probcal.monitor._monitor import grade_key, grade_keys
from probcal.monitor._onset import estimate_onset
from probcal.monitor._processes import exponent_weights, plug_in_delta, plug_in_shape


def _batch(n: int = 300, shift: float = 0.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = expit(rng.normal(-2.5, 1.0, n))
    y = (rng.random(n) < expit(logit(p) + shift)).astype(float)
    return y, p


def _grades(n: int) -> np.ndarray:
    return np.array(["A", "B", "C"])[np.arange(n) % 3]


# ---------------------------------------------------------------- weights


def test_exponent_weights_unit_and_constant_weights_are_exactly_one() -> None:
    for w in (np.ones(17), np.full(200, 0.1), np.full(3, 1e6)):
        u = exponent_weights(w)
        assert np.array_equal(u, np.ones_like(w))
        assert u.dtype == np.float64


def test_exponent_weights_bounded_and_scale_free() -> None:
    w = np.random.default_rng(0).lognormal(0.0, 1.5, 500)
    u = exponent_weights(w)
    assert np.all((u > 0.0) & (u <= 1.0))
    assert np.allclose(exponent_weights(1e4 * w), u, rtol=1e-12)
    big = w >= w.mean()
    assert np.all(u[big] == 1.0)
    assert np.allclose(u[~big], w[~big] / w.mean())


@pytest.mark.parametrize("scale", [10.0, 0.1, 1e4])
def test_constant_weights_reduce_to_the_unweighted_monitor(scale: float) -> None:
    plain = CalibrationMonitor(grades=("A", "B", "C"), delta_ci_grid=(-2.0, 2.0, 41))
    weighted = CalibrationMonitor(grades=("A", "B", "C"), delta_ci_grid=(-2.0, 2.0, 41))
    for k in range(6):
        y, p = _batch(shift=0.5 if k >= 3 else 0.0, seed=k)
        g = _grades(len(p))
        a = plain.update(y, p, grade=g, label=f"b{k}")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            b = weighted.update(y, p, sample_weight=np.full(len(p), scale), grade=g, label=f"b{k}")
        assert a.e_global == b.e_global
        assert a.e_offset == b.e_offset
        assert a.e_shape == b.e_shape
        assert a.e_grades == b.e_grades
        assert a.delta_ci == b.delta_ci
        assert a.grade_delta_ci == b.grade_delta_ci
        assert a.log_e_increment == b.log_e_increment


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
def test_skewed_weight_batch_factor_is_an_e_value(sigma: float) -> None:
    # Under the null, E[prod_i LR_i^u_i] <= 1 for exponents u_i in (0, 1];
    # raw exponents w_i > 1 inflate it above 1.
    rng = np.random.default_rng(0)
    n, reps = 12, 40_000
    p = expit(rng.normal(-1.5, 0.8, n))
    q = expit(logit(p) + 0.7)
    w = rng.lognormal(0.0, sigma, n) * 5.0
    u = exponent_weights(w)
    y = (rng.random((reps, n)) < p).astype(float)
    terms = y * (np.log(q) - np.log(p)) + (1.0 - y) * (np.log1p(-q) - np.log1p(-p))
    factor = np.exp(terms @ u)
    raw = np.exp(terms @ w)
    se = factor.std() / np.sqrt(reps)
    assert factor.mean() <= 1.0 + 3.0 * se
    assert raw.mean() > 1.5
    # same quantity as the processes compute
    assert np.isclose(np.log(factor[0]), bern_log_lr(y[0], p, q, u))


@pytest.mark.parametrize("weights", ["constant", "skewed"])
def test_weighted_null_alarm_rate_is_controlled(weights: str) -> None:
    alarms = 0
    reps = 40
    for r in range(reps):
        rng = np.random.default_rng(r)
        mon = CalibrationMonitor(components=("offset",))
        for _ in range(10):
            p = expit(rng.normal(-3.0, 1.0, 200))
            y = (rng.random(200) < p).astype(float)
            w = np.full(200, 10.0) if weights == "constant" else 1e4 * rng.lognormal(0, 1.5, 200)
            s = mon.update(y, p, sample_weight=w)
        alarms += s.alarm
    assert alarms / reps <= 0.05 + 0.05  # loose: alpha plus Monte Carlo slack


# ---------------------------------------------------------------- legacy re-keying


@pytest.mark.parametrize("version", ["0.3.4", "0.3.5", "0.3.5.dev0", "0.4.0"])
def test_current_payload_string_labels_are_not_rekeyed(version: str) -> None:
    mon = CalibrationMonitor()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(3):
            y, p = _batch(seed=10 + k)
            mon.update(y, p, grade=np.array(["1.0", "2.0"])[np.arange(len(p)) % 2])
    d = json.loads(mon.to_json())
    d["probcal_version"] = version
    loaded = CalibrationMonitor.from_dict(d)
    assert set(loaded.steps_[-1].e_grades) == {"1.0", "2.0"}
    assert loaded.steps_ == mon.steps_
    y, p = _batch(seed=20)
    g = np.array(["1.0", "2.0"])[np.arange(len(p)) % 2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert loaded.update(y, p, grade=g) == mon.update(y, p, grade=g)


def test_pre_0_3_4_payload_float_keys_are_still_rekeyed() -> None:
    mon = CalibrationMonitor(grades=(1.0, 2.0))
    y, p = _batch(seed=30)
    g = ((np.arange(len(p)) % 2) + 1).astype(float)
    mon.update(y, p, grade=g)
    d = json.loads(mon.to_json())
    d["probcal_version"] = "0.3.3"
    st = d["state"]
    st["grade_procs"] = {f"{k}.0": v for k, v in st["grade_procs"].items()}
    loaded = CalibrationMonitor.from_dict(d)
    assert set(loaded._grade_procs) == {"1", "2"}


# ---------------------------------------------------------------- onset


def test_estimate_onset_ties_resolve_to_earliest_and_no_evidence_is_whole_history() -> None:
    assert estimate_onset(np.array([0.0, 3.0, 2.0])) == 0
    assert estimate_onset(np.array([-1.0, 0.0, 3.0])) == 1
    assert estimate_onset(np.array([0.0, 0.0, 0.0])) == 0
    assert estimate_onset(np.array([-1.0, -0.5])) == 0


def _drift_from_start(**kw: object) -> CalibrationMonitor:
    mon = CalibrationMonitor(**kw)  # type: ignore[arg-type]
    for k in range(6):
        y, p = _batch(n=2000, shift=0.6, seed=40 + k)
        mon.update(y, p, label=f"m{k}")
    return mon


def test_onset_never_after_alarm() -> None:
    mon = _drift_from_start()
    rep = mon.report()
    labels = [s.label for s in mon.steps_]
    assert rep.alarm_at is not None and rep.onset_label is not None
    assert labels.index(rep.onset_label) <= labels.index(rep.alarm_at)


def test_no_plug_in_evidence_falls_back_to_whole_history() -> None:
    # min_history so large the plug-ins never engage: every increment is 0
    # and the alarm comes from the fixed-grid mixture alone.
    mon = _drift_from_start(min_history=100, components=("offset",))
    assert all(s.log_e_increment == 0.0 for s in mon.steps_)
    rep = mon.report()
    assert rep.alarm_at == "m0"
    assert rep.onset_label == "m0"
    action = mon.apply_recommendation()
    assert action.window == tuple(s.label for s in mon.steps_)
    assert action.audit["onset_label"] == rep.onset_label


# ---------------------------------------------------------------- grade keys


@pytest.mark.parametrize(
    "labels",
    [
        [1.0, "A"],
        np.array([0.1, 0.2, 0.1], dtype=np.float32),
        np.array([1e20, -3.0, 2.5, np.nan]),
        np.array([2**64 - 1, 3], dtype=np.uint64),
        np.array([1, 2, 1], dtype=np.int8),
        np.array([True, False]),
        np.array([b"A", b"B"]),
        np.array(["1.0", "x"]),
        (1, 2.0, "B"),
    ],
)
def test_grade_keys_match_grade_key_element_wise(labels: object) -> None:
    keys = grade_keys(labels)
    assert keys.tolist() == [grade_key(x) for x in list(labels)]  # type: ignore[call-overload]
    assert all(type(k) is str for k in keys.tolist())


def test_mixed_and_float32_declared_grades_accept_their_own_labels() -> None:
    mon = CalibrationMonitor(grades=(1.0, "A"))
    mon.update([0.0, 1.0], [0.1, 0.2], grade=[1.0, "A"])
    g32 = np.array([0.1, 0.2], dtype=np.float32)
    mon = CalibrationMonitor(grades=tuple(g32))
    mon.update([0.0, 1.0], [0.1, 0.2], grade=g32)
    assert set(mon.steps_[-1].e_grades) == {grade_key(x) for x in g32}


def test_grade_keys_rejects_non_1d() -> None:
    with pytest.raises(ValueError, match="1-D"):
        grade_keys(np.zeros((2, 2)))


# ---------------------------------------------------------------- MoC from counts


def test_moc_offset_from_counts_invariant_to_weight_scale() -> None:
    y = np.array([0.0] * 970 + [1.0] * 30)
    p = np.full(1000, 0.02)
    ref = moc_offset_from_counts(y, p, level=0.9).post_mean_
    assert ref == pytest.approx(beta_ppf(0.9, 30.5, 970.5))
    for s in (0.01, 1.0, 100.0):
        got = moc_offset_from_counts(y, p, level=0.9, sample_weight=np.full(1000, s))
        assert got.post_mean_ == pytest.approx(ref, rel=1e-10)
    unit = moc_offset_from_counts(y, p, level=0.9, sample_weight=np.ones(1000))
    assert unit.post_mean_ == ref


def test_residual_shape_lr_invariant_to_weight_scale() -> None:
    y, p = _batch(n=2000, shift=0.4, seed=50)
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


class _Scale:
    def __init__(self, fp: str) -> None:
        self._fp = fp

    def fingerprint(self) -> str:
        return self._fp

    def assign(self, p: np.ndarray) -> np.ndarray:
        return np.where(p < 0.1, "G1", "G2")


def test_masterscale_fingerprint_recorded_only_for_accepted_batches() -> None:
    mon = CalibrationMonitor(grades=("Z",))
    y, p = _batch(seed=60)
    with pytest.raises(ValueError, match="not in the declared grades"):
        mon.update(y, p, grade=_Scale("a" * 64))
    assert mon.masterscale_fingerprint_ is None
    assert mon.steps_ == []

    mon = CalibrationMonitor(grades=("G1", "G2"))
    with pytest.raises(ValueError, match="equal length"):
        mon.update(y, p[:-1], grade=_Scale("a" * 64))
    assert mon.masterscale_fingerprint_ is None
    mon.update(y, p, grade=_Scale("b" * 64))
    assert mon.masterscale_fingerprint_ == "b" * 64
    with pytest.raises(ValueError, match="different masterscale"):
        mon.update(y, p, grade=_Scale("c" * 64))
    assert mon.masterscale_fingerprint_ == "b" * 64


@pytest.mark.parametrize("grid", [(), (0.0,), (-0.5, 1.0), (np.inf,), (np.nan,), (True,), ("0.5",)])
def test_bad_mixture_grid_rejected_at_construction(grid: tuple) -> None:
    with pytest.raises(ValueError, match="mixture_grid"):
        CalibrationMonitor(mixture_grid=grid)


@pytest.mark.parametrize(
    "grid",
    [
        (3.0, -3.0, 241),
        (-3.0, 3.0, 0),
        (-3.0, 3.0, 1),
        (-3.0, 3.0, 2.5),
        (0.5, 3.0, 11),
        (-3.0, -0.5, 11),
        (-np.inf, 3.0, 11),
        (-3.0, 3.0),
        (1.0, 1.0, 5),
    ],
)
def test_bad_delta_ci_grid_rejected_at_construction(grid: tuple) -> None:
    with pytest.raises(ValueError, match="delta_ci_grid"):
        CalibrationMonitor(delta_ci_grid=grid)  # type: ignore[arg-type]


def test_valid_grids_still_accepted() -> None:
    CalibrationMonitor(mixture_grid=(0.5,), delta_ci_grid=(0.0, 2.0, 2))
    CalibrationMonitor(mixture_grid=(1, 2), delta_ci_grid=(-1, 0, 3))
