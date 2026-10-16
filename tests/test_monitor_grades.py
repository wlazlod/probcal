"""Per-grade e-processes, grade keys, and grade confidence sequences."""

import warnings

import numpy as np
import pytest

from probcal._math import expit, logit
from probcal.datasets import make_pd_portfolio
from probcal.monitor import CalibrationMonitor
from probcal.monitor._monitor import grade_key, grade_keys


def _grade_batch(
    n: int = 1500, shift_a: float = 0.0, shift_b: float = 0.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d_a = make_pd_portfolio(n=n, random_state=seed)
    d_b = make_pd_portfolio(n=n, random_state=seed + 1)
    p = np.concatenate([d_a.scores, d_b.scores])
    true = np.concatenate([expit(logit(d_a.scores) + shift_a), expit(logit(d_b.scores) + shift_b)])
    rng = np.random.default_rng(seed + 5000)
    y = (rng.random(len(p)) < true).astype(float)
    g = np.array(["A"] * n + ["B"] * n)
    return y, p, g


def _batch(n=1000, shift=0.0, seed=0):
    d = make_pd_portfolio(n=n, random_state=seed)
    rng = np.random.default_rng(seed + 1000)
    y = (rng.random(n) < expit(logit(d.scores) + shift)).astype(float)
    return y, d.scores


def _grades(n, labels=("A", "B")):
    return np.asarray(labels)[np.arange(n) % len(labels)]


@pytest.mark.slow
def test_two_grade_drift_confidence_sequence_coverage() -> None:
    n_runs = 20
    hits_a = hits_b = 0
    for r in range(n_runs):
        mon = CalibrationMonitor(alpha=0.05, delta_ci_grid=(-2.0, 2.0, 81), grades=("A", "B"))
        step = None
        for k in range(6):
            y, p, g = _grade_batch(n=1500, shift_a=0.6, shift_b=0.0, seed=1000 * r + k)
            step = mon.update(y, p, grade=g, label=f"m{k}")
        lo_a, hi_a = step.grade_delta_ci["A"]
        lo_b, hi_b = step.grade_delta_ci["B"]
        hits_a += lo_a <= 0.6 <= hi_a
        hits_b += lo_b <= 0.0 <= hi_b
    assert hits_a / n_runs >= 0.9
    assert hits_b / n_runs >= 0.9


def test_declared_grades_drive_the_alarm_on_single_grade_drift() -> None:
    mon = CalibrationMonitor(alpha=0.05, grades=("A", "B"))
    for k in range(8):
        y_a, p_a = _batch(n=1000, shift=1.2, seed=200 + k)
        y_b, p_b = _batch(n=1000, shift=0.0, seed=300 + k)
        g = np.array(["A"] * 1000 + ["B"] * 1000)
        step = mon.update(np.concatenate([y_a, y_b]), np.concatenate([p_a, p_b]), grade=g)
    assert step.alarm
    assert step.e_grades["A"] > step.e_grades["B"]
    assert "A" in mon.report().grade_table


def test_declared_grades_are_in_the_mixture_from_the_first_batch() -> None:
    # MON-6: the mixture weights are fixed by the declared universe.
    mon = CalibrationMonitor(grades=("A", "B", "C"))
    for k in range(5):
        y, p = _batch(seed=40 + k)
        labels = ("A", "B", "C") if k >= 2 else ("A", "B")  # C appears mid-stream
        step = mon.update(y, p, grade=_grades(len(p), labels), label=f"m{k}")
        assert set(step.e_grades) == {"A", "B", "C"}
        if k < 2:
            assert step.e_grades["C"] == 1.0  # no data yet: e = 1, weight fixed
        grade_mean = np.mean([step.e_grades[g] for g in ("A", "B", "C")])
        expected = np.mean([step.e_offset, step.e_shape, grade_mean])
        assert step.e_global == pytest.approx(expected, rel=1e-12)


def test_undeclared_grades_do_not_enter_the_global_mixture() -> None:
    # MON-6: grades first seen mid-stream used to re-weight the mixture.
    plain, graded = CalibrationMonitor(), CalibrationMonitor()
    with pytest.warns(UserWarning, match="not part of the global alarm"):
        for k in range(5):
            y, p = _batch(seed=10 + k)
            a = plain.update(y, p, label=f"m{k}")
            b = graded.update(y, p, grade=_grades(len(p)) if k >= 2 else None, label=f"m{k}")
            assert a.e_global == b.e_global
            assert a.alarm == b.alarm
    assert set(graded.steps_[-1].e_grades) == {"A", "B"}  # still tracked and reported


def test_undeclared_grade_warning_fires_once_and_survives_persistence() -> None:
    mon = CalibrationMonitor()
    y, p = _batch(seed=3)
    with pytest.warns(UserWarning, match="grades=None"):
        mon.update(y, p, grade=_grades(len(p)))
    mon2 = CalibrationMonitor.from_dict(mon.to_dict())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mon2.update(y, p, grade=_grades(len(p)))


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


def test_grade_keys_rejects_non_1d() -> None:
    with pytest.raises(ValueError, match="1-D"):
        grade_keys(np.zeros((2, 2)))


def test_int_and_float_grade_labels_share_one_process() -> None:
    mon = CalibrationMonitor(grades=(1, 2))
    y, p = _batch(seed=80)
    g_int = (np.arange(len(p)) % 2) + 1
    mon.update(y, p, grade=g_int)
    step = mon.update(y, p, grade=g_int.astype(float))  # 1.0 is grade "1"
    assert set(step.e_grades) == {"1", "2"}
    assert set(step.grade_delta_ci) == {"1", "2"}


def test_mixed_and_float32_declared_grades_accept_their_own_labels() -> None:
    mon = CalibrationMonitor(grades=(1.0, "A"))
    mon.update([0.0, 1.0], [0.1, 0.2], grade=[1.0, "A"])
    g32 = np.array([0.1, 0.2], dtype=np.float32)
    mon = CalibrationMonitor(grades=tuple(g32))
    mon.update([0.0, 1.0], [0.1, 0.2], grade=g32)
    assert set(mon.steps_[-1].e_grades) == {grade_key(x) for x in g32}


class _Scale:
    def __init__(self, fp: str) -> None:
        self._fp = fp

    def fingerprint(self) -> str:
        return self._fp

    def assign(self, p: np.ndarray) -> np.ndarray:
        return np.where(p < 0.1, "G1", "G2")


def test_masterscale_fingerprint_recorded_only_for_accepted_batches() -> None:
    mon = CalibrationMonitor(grades=("Z",))
    y, p = _batch(n=300, seed=60)
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


# ---------------------------------------------------------------- string labels


def test_float_like_string_labels_round_trip_unchanged() -> None:
    mon = CalibrationMonitor()
    labels = np.array(["1.0", "2.0"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(3):
            y, p = _batch(n=300, seed=10 + k)
            mon.update(y, p, grade=labels[np.arange(len(p)) % 2])
    loaded = CalibrationMonitor.from_json(mon.to_json())
    assert set(loaded.steps_[-1].e_grades) == {"1.0", "2.0"}
    assert loaded.steps_ == mon.steps_
    y, p = _batch(n=300, seed=20)
    g = labels[np.arange(len(p)) % 2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert loaded.update(y, p, grade=g) == mon.update(y, p, grade=g)
