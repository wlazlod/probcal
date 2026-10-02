"""Regression tests for the 0.4.0 monitor fixes (MON-* review items)."""

import json
import warnings

import numpy as np
import pytest

from probcal import BetaCalibrator, CalibratedModel, make_pd_portfolio
from probcal._math import bern_log_lr, expit, logit
from probcal.monitor import CalibrationMonitor


def _batch(n=1000, shift=0.0, seed=0, event_rate=0.05):
    d = make_pd_portfolio(n=n, event_rate=event_rate, random_state=seed)
    p = d.scores
    rng = np.random.default_rng(seed + 1000)
    y = (rng.random(n) < expit(logit(p) + shift)).astype(float)
    return y, p


def _grades(n, labels=("A", "B")):
    return np.asarray(labels)[np.arange(n) % len(labels)]


# ---------------------------------------------------------------- MON-7 / MON-8


@pytest.mark.parametrize("window", [0, -1, True, 1.5])
def test_plug_in_window_must_be_none_or_positive(window) -> None:
    # plug_in_window=0 used to mean "all history" (slice(-0, None)).
    with pytest.raises(ValueError, match="plug_in_window"):
        CalibrationMonitor(plug_in_window=window)


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_min_history_must_be_a_non_negative_integer(value) -> None:
    with pytest.raises(ValueError, match="min_history"):
        CalibrationMonitor(min_history=value)


def test_min_history_zero_is_legal() -> None:
    mon = CalibrationMonitor(min_history=0)
    y, p = _batch(seed=1)
    assert mon.update(y, p).delta_hat == 0.0  # empty past -> identity plug-in


def test_empty_batch_raises_a_clear_error() -> None:
    # used to surface as an IndexError from w_arr[0]
    mon = CalibrationMonitor()
    with pytest.raises(ValueError, match="batch is empty"):
        mon.update(np.array([]), np.array([]))
    assert mon.steps_ == []


# ---------------------------------------------------------------- MON-6: fixed mixture


def _old_e_global(step, n_components=2) -> float:
    """The pre-0.4.0 rule: every grade seen so far joined the average."""
    parts = [step.e_offset, step.e_shape][:n_components]
    if step.e_grades:
        parts.append(float(np.mean(list(step.e_grades.values()))))
    return float(np.mean(parts))


def test_undeclared_grades_do_not_enter_the_global_mixture() -> None:
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


@pytest.mark.slow
def test_null_stream_e_global_never_jumps_when_grades_appear_mid_stream() -> None:
    # Under H0, the global e-value at the batch where grades first appear
    # must be exactly what the same stream without grade arrays gives. The
    # pre-0.4.0 rule (a new component entering at e = 1) instead pulled a
    # below-1 global value up toward 1 with no evidence at all.
    runs, appear = 40, 3
    jumped_before = 0
    for r in range(runs):
        plain, graded = CalibrationMonitor(), CalibrationMonitor()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for k in range(6):
                y, p = _batch(n=800, seed=10_000 * (r + 1) + k)
                g = _grades(len(p)) if k >= appear else None
                a = plain.update(y, p, label=f"m{k}")
                b = graded.update(y, p, grade=g, label=f"m{k}")
                assert b.e_global == a.e_global, (r, k)
                if k == appear and _old_e_global(b) > b.e_global and b.e_global < 1.0:
                    jumped_before += 1
    # the old rule would have jumped in most null runs (global value < 1)
    assert jumped_before >= runs // 2, jumped_before


def test_declared_grades_are_in_the_mixture_from_the_first_batch() -> None:
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


def test_declared_grade_component_without_any_grade_array() -> None:
    # The component is part of the mixture whether or not grade arrays come.
    mon = CalibrationMonitor(grades=("A", "B"))
    y, p = _batch(seed=5)
    step = mon.update(y, p)
    assert step.e_global == pytest.approx(np.mean([step.e_offset, step.e_shape, 1.0]))


def test_label_outside_declared_grades_raises() -> None:
    # used to be added silently, re-weighting the mixture mid-stream
    mon = CalibrationMonitor(grades=("A", "B"))
    y, p = _batch(seed=6)
    with pytest.raises(ValueError, match="not in the declared grades"):
        mon.update(y, p, grade=_grades(len(p), ("A", "B", "Z")))
    assert mon.steps_ == []


def test_declared_grades_drive_the_alarm_on_single_grade_drift() -> None:
    mon = CalibrationMonitor(alpha=0.05, grades=("A", "B"))
    for k in range(8):
        y_a, p_a = _batch(n=1000, shift=1.2, seed=200 + k)
        y_b, p_b = _batch(n=1000, shift=0.0, seed=300 + k)
        g = np.array(["A"] * 1000 + ["B"] * 1000)
        step = mon.update(np.concatenate([y_a, y_b]), np.concatenate([p_a, p_b]), grade=g)
    assert step.alarm
    assert step.e_grades["A"] > step.e_grades["B"]


# ---------------------------------------------------------------- MON-10: onset series


def test_onset_increment_sums_only_active_components() -> None:
    mon = CalibrationMonitor(components=("offset",))
    for k in range(5):
        y, p = _batch(shift=0.5, seed=60 + k)
        step = mon.update(y, p, label=f"m{k}")
        d = step.delta_hat
        expected = 0.0 if d == 0.0 else bern_log_lr(y, p, expit(logit(p) + d), np.ones_like(p))
        assert step.log_e_increment == pytest.approx(expected, abs=1e-12)


def test_onset_increment_includes_grades_only_when_declared() -> None:
    plain, undeclared, declared = (
        CalibrationMonitor(),
        CalibrationMonitor(),
        CalibrationMonitor(grades=("A", "B")),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for k in range(4):
            y, p = _batch(shift=0.6, seed=70 + k)
            g = _grades(len(p))
            a = plain.update(y, p)
            b = undeclared.update(y, p, grade=g)
            c = declared.update(y, p, grade=g)
            assert b.log_e_increment == a.log_e_increment
            if k >= 1:  # plug-ins engaged: grade plug-ins add their own evidence
                assert c.log_e_increment != a.log_e_increment


# ---------------------------------------------------------------- MON-11 / MON-13: grade keys


def test_int_and_float_grade_labels_share_one_process() -> None:
    mon = CalibrationMonitor(grades=(1, 2))
    y, p = _batch(seed=80)
    g_int = (np.arange(len(p)) % 2) + 1
    mon.update(y, p, grade=g_int)
    step = mon.update(y, p, grade=g_int.astype(float))  # 1.0 is grade "1"
    assert set(step.e_grades) == {"1", "2"}
    assert set(step.grade_delta_ci) == {"1", "2"}


def test_duplicate_declared_grades_after_normalization_raise() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        CalibrationMonitor(grades=(1, 1.0))


def test_grade_keys_are_plain_str() -> None:
    mon = CalibrationMonitor(grades=("A", "B"))
    y, p = _batch(seed=81)
    step = mon.update(y, p, grade=_grades(len(p)))
    assert all(type(k) is str for k in step.e_grades)
    assert all(type(k) is str for k in step.grade_delta_ci)
    assert all(type(k) is str for k in mon.report().grade_table)


def test_pre_0_4_float_grade_keys_are_renamed_on_load() -> None:
    mon = CalibrationMonitor(grades=(1.0, 2.0))
    y, p = _batch(seed=82)
    g = ((np.arange(len(p)) % 2) + 1).astype(float)
    mon.update(y, p, grade=g)
    d = json.loads(mon.to_json())

    def legacy(k: str) -> str:
        return {"1": "1.0", "2": "2.0"}.get(k, k)

    # rewrite as a 0.3.x writer keyed them: str(1.0) == "1.0"
    st = d["state"]
    d["probcal_version"] = "0.3.3"
    st["grade_procs"] = {legacy(k): v for k, v in st["grade_procs"].items()}
    st["grade_cs_log"] = {legacy(k): v for k, v in st["grade_cs_log"].items()}
    st["grade_cs_max"] = {legacy(k): v for k, v in st["grade_cs_max"].items()}
    st["g"] = [[legacy(x) for x in a] for a in st["g"]]
    for s in st["steps"]:
        s["e_grades"] = {legacy(k): v for k, v in s["e_grades"].items()}
        s["grade_delta_ci"] = {legacy(k): v for k, v in s["grade_delta_ci"].items()}

    loaded = CalibrationMonitor.from_dict(d)
    assert loaded.steps_ == mon.steps_
    a = mon.update(y, p, grade=g)
    b = loaded.update(y, p, grade=g)
    assert a == b


# ---------------------------------------------------------------- MON-14: reasoning


def test_reasoning_never_prints_nan_for_an_unmonitored_component() -> None:
    mon = CalibrationMonitor(components=("offset",))
    for k in range(6):
        y, p = _batch(n=2000, shift=0.8, seed=90 + k)
        mon.update(y, p, label=f"m{k}")
    rep = mon.report()
    assert rep.alarm_at is not None
    assert not any("nan" in line for line in rep.reasoning)
    assert any("not monitored" in line for line in rep.reasoning)


# ---------------------------------------------------------------- MON-5: CalibratedModel target


class _StubModel:
    def fit(self, X, y):  # noqa: ARG002
        return self

    def predict_proba(self, X):
        s = np.asarray(X)[:, 0]
        return np.column_stack([1.0 - s, s])

    def get_params(self):
        return {"stub": True}


def _drifted_monitor() -> CalibrationMonitor:
    mon = CalibrationMonitor(alpha=0.05)
    for k in range(6):
        y, p = _batch(n=1000, shift=0.8, seed=k)
        mon.update(y, p, label=f"m{k}")
    return mon


def test_apply_recommendation_on_a_calibrated_model_loaded_from_json() -> None:
    # offset_to() needs the calibration scores, which JSON does not carry:
    # this used to raise RuntimeError for any model loaded from disk.
    d = make_pd_portfolio(n=600, random_state=3)
    wrapped = CalibratedModel(_StubModel(), BetaCalibrator(), flow="prefit").fit(
        d.scores.reshape(-1, 1), d.y
    )
    loaded = CalibratedModel.from_json(wrapped.to_json(), model=_StubModel())
    action = _drifted_monitor().apply_recommendation(target=loaded)
    assert action.kind == "re-offset"
    X = d.scores.reshape(-1, 1)
    expected = action.offset.transform(loaded.predict_proba(X))
    np.testing.assert_allclose(action.composed.predict_proba(X), expected, atol=1e-12)
    assert loaded.offsets_ == []  # target untouched
    assert action.audit["new_target_fingerprint"] == action.composed.fingerprint()


# ---------------------------------------------------------------- MON-9: strict JSON


def test_monitor_and_action_json_is_strict() -> None:
    # nan component e-values and the -inf initial running max used to be
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
