"""Tests for probcal.monitor._actions (margin-of-conservatism offsets, apply_recommendation)."""

from dataclasses import replace

import numpy as np
import pytest

from probcal import BetaCalibrator, CalibratedModel, Chain, make_pd_portfolio
from probcal._math import beta_ppf, expit, logit
from probcal.monitor import AppliedAction, CalibrationMonitor, moc_offset, moc_offset_from_counts
from probcal.monitor._onset import estimate_onset


def _batch(n=1000, shift=0.0, slope=1.0, seed=0, event_rate=0.05):
    d = make_pd_portfolio(n=n, event_rate=event_rate, random_state=seed)
    p = d.scores
    true = expit(slope * logit(p) + shift)
    rng = np.random.default_rng(seed + 1000)
    y = (rng.random(n) < true).astype(float)
    return y, p


@pytest.fixture(scope="module")
def drifted() -> CalibrationMonitor:
    """Shared drifted monitor; moc_offset / apply_recommendation never mutate it."""
    mon = CalibrationMonitor(alpha=0.05)
    for k in range(6):
        y, p = _batch(shift=0.8, seed=k)
        mon.update(y, p, label=f"m{k}")
    return mon


class _StubModel:
    """Deterministic sklearn-free model over a single score column."""

    def fit(self, X, y):  # noqa: ARG002
        return self

    def predict_proba(self, X):
        s = np.asarray(X)[:, 0]
        return np.column_stack([1.0 - s, s])

    def get_params(self):
        return {"stub": True}


def _wrapped(n: int = 400) -> tuple[CalibratedModel, np.ndarray]:
    d = make_pd_portfolio(n=n, random_state=7)
    X = d.scores.reshape(-1, 1)
    return CalibratedModel(_StubModel(), BetaCalibrator(), flow="prefit").fit(X, d.y), X


# ---------------------------------------------------------------- moc_offset


def test_moc_offset_default_uses_last_step_delta_ci(drifted) -> None:
    # The CS's upper end is the margin of conservatism: at least the
    # predictable plug-in estimate, fitted on the last batch's probabilities.
    off = moc_offset(drifted)
    assert off.delta_ == pytest.approx(drifted.steps_[-1].delta_ci[1])
    assert off.delta_ >= drifted.steps_[-1].delta_hat
    assert off.pre_mean_ == pytest.approx(float(np.mean(expit(drifted._z[-1]))))

    # A report carries no batch: same delta, placeholder pre-mean.
    from_report = moc_offset(drifted.report())
    assert from_report.delta_ == pytest.approx(off.delta_)
    assert from_report.pre_mean_ == pytest.approx(0.5)


def test_moc_offset_level_recomputes_surviving_set(drifted) -> None:
    off_level = moc_offset(drifted, level=1.0 - drifted.alpha)
    # Same alpha as the monitor's own -> same threshold -> same result.
    assert off_level.delta_ == pytest.approx(moc_offset(drifted).delta_)

    threshold = -np.log(1.0 - 0.5)
    surviving = drifted._cs.grid[drifted._cs.max < threshold]
    assert moc_offset(drifted, level=0.5).delta_ == pytest.approx(float(surviving.max()))


def test_action_validation(drifted) -> None:
    with pytest.raises(TypeError, match="CalibrationMonitor"):
        moc_offset(drifted.report(), level=0.8)
    with pytest.raises(ValueError, match="level"):
        moc_offset(drifted, level=1.5)
    with pytest.raises(ValueError, match="no batches"):
        moc_offset(CalibrationMonitor())
    with pytest.raises(TypeError):
        moc_offset("not a monitor")
    with pytest.raises(ValueError, match="level"):
        moc_offset_from_counts(np.array([0.0, 1.0]), np.array([0.1, 0.2]), level=0.0)
    with pytest.raises(TypeError, match="Chain"):
        drifted.apply_recommendation(target=object())


def test_moc_offset_grid_too_narrow_raises() -> None:
    # The drift runs off a too-narrow delta_ci_grid: no CS at the monitor's
    # own level, and none at a stricter level either.
    for alpha, level in ((0.05, None), (0.4, 0.999999)):
        mon = CalibrationMonitor(alpha=alpha, delta_ci_grid=(-0.1, 0.1, 5))
        for k in range(4):
            y, p = _batch(shift=2.0, seed=k)
            mon.update(y, p, label=f"m{k}")
        if level is None:
            assert mon.steps_[-1].delta_ci is None
        with pytest.raises(ValueError, match="delta_ci_grid"):
            moc_offset(mon, level=level)


# ---------------------------------------------------------------- moc_offset_from_counts


def test_moc_offset_from_counts_jeffreys_quantile_and_weight_scale() -> None:
    y = np.array([0.0] * 970 + [1.0] * 30)
    p = np.full(1000, 0.02)
    ref = moc_offset_from_counts(y, p, level=0.9)
    assert ref.post_mean_ == pytest.approx(beta_ppf(0.9, 30.5, 970.5))
    np.testing.assert_allclose(ref.transform(p).mean(), ref.post_mean_, atol=1e-10)
    for s in (0.01, 1.0, 100.0):
        got = moc_offset_from_counts(y, p, level=0.9, sample_weight=np.full(1000, s))
        assert got.post_mean_ == pytest.approx(ref.post_mean_, rel=1e-10)


def test_moc_offset_from_counts_weighted_uses_kish_counts() -> None:
    y = np.array([0.0, 0.0, 1.0, 0.0])
    p = np.array([0.05, 0.05, 0.05, 0.05])
    w = np.array([2.0, 3.0, 1.0, 4.0])
    off = moc_offset_from_counts(y, p, level=0.9, sample_weight=w)
    # Kish effective counts: n_eff = sum(w)^2 / sum(w^2), events scaled alike
    kish = float(np.sum(w)) / float(np.sum(w * w))
    k = float(np.sum(w * y)) * kish
    n = float(np.sum(w)) * kish
    q = beta_ppf(0.9, k + 0.5, n - k + 0.5)
    assert off.post_mean_ == pytest.approx(q, abs=1e-9)


# ---------------------------------------------------------------- apply_recommendation


def test_apply_recommendation_re_fit_carries_no_offset() -> None:
    mon = CalibrationMonitor(alpha=0.05)
    for k in range(12):
        y, p = _batch(shift=0.0, slope=0.7, seed=100 + k)
        mon.update(y, p, label=f"m{k}")
    assert mon.report().recommendation == "re-fit"

    action = mon.apply_recommendation()
    assert action.kind == "re-fit"
    assert action.offset is None
    assert action.composed is None
    assert action.monitor is None
    assert action.window  # non-empty: the suggested re-fit window


def test_apply_recommendation_window_uses_onset_index_not_label_lookup() -> None:
    # Regression: labels are documented as opaque and may repeat. Deriving
    # the onset index by looking a batch up BY LABEL (the previous
    # implementation) silently returns the FIRST match and can point at
    # the wrong index; the window must come from estimate_onset's index
    # directly (CalibrationMonitor._onset_index), exactly as report() does.
    mon = CalibrationMonitor(alpha=0.05)
    for k in range(4):
        y, p = _batch(n=2000, shift=0.0, seed=k)
        mon.update(y, p, label="m")  # every batch shares the same label
    for k in range(6):
        y, p = _batch(n=2000, shift=0.8, seed=100 + k)
        mon.update(y, p, label="m")
    assert mon.report().alarm_at == "m"

    # Pin the evidence trail so the onset is unambiguously the first drifted
    # batch (index 4), up to and including the first alarm.
    forced = [0.0, -1.0, -1.0, -1.0, 20.0, 50.0, 80.0, 60.0, 80.0, 80.0]
    mon.steps_ = [replace(s, log_e_increment=v) for s, v in zip(mon.steps_, forced, strict=True)]
    alarm_idx = next(i for i, s in enumerate(mon.steps_) if s.alarm)
    assert alarm_idx >= 4
    onset_idx = estimate_onset(np.array(forced[: alarm_idx + 1]))
    assert onset_idx == 4
    action = mon.apply_recommendation()
    assert len(action.window) == len(mon.steps_) - onset_idx
    assert len(action.window) != len(mon.steps_)  # a label lookup would find index 0


def test_apply_recommendation_audit_round_trips_and_leaves_monitor_untouched(drifted) -> None:
    before = drifted.to_dict()
    action = drifted.apply_recommendation()
    assert drifted.to_dict() == before

    restored = AppliedAction.from_json(action.to_json())
    assert restored.audit == action.audit
    assert restored.fingerprint() == action.fingerprint()


def test_apply_recommendation_composes_chain_target(drifted) -> None:
    d = make_pd_portfolio(n=400, random_state=7)
    chain = Chain([BetaCalibrator().fit(d.scores, d.y)])

    action = drifted.apply_recommendation(target=chain)
    assert isinstance(action.composed, Chain)
    assert action.composed.offsets_[-1].delta_ == pytest.approx(action.offset.delta_)
    assert chain.offsets_ == ()  # the original target is untouched

    restored = AppliedAction.from_json(action.to_json())
    assert isinstance(restored.composed, Chain)
    assert restored.fingerprint() == action.fingerprint()
    assert restored.composed.fingerprint() == action.composed.fingerprint()


def test_apply_recommendation_composes_calibrated_model_target(drifted) -> None:
    wrapped, X = _wrapped()
    action = drifted.apply_recommendation(target=wrapped)
    assert len(action.composed.offsets_) == 1
    assert action.composed.offsets_[-1].delta_ == pytest.approx(action.offset.delta_)
    assert wrapped.offsets_ == []  # the original target is untouched (deep-copied)
    js = action.to_json()

    # Lazy load: only a model *reference* was serialized (CalibratedModel.to_dict),
    # never the model itself -- composed comes back with model_ unattached.
    lazy = AppliedAction.from_json(js)
    assert isinstance(lazy.composed, CalibratedModel)
    assert lazy.composed.model_ is None

    # model= reattaches the same model class: fingerprints agree exactly,
    # and prediction is possible again.
    restored = AppliedAction.from_json(js, model=_StubModel())
    assert restored.fingerprint() == action.fingerprint()
    assert restored.composed.fingerprint() == action.composed.fingerprint()
    np.testing.assert_array_equal(
        restored.composed.predict_proba(X), action.composed.predict_proba(X)
    )


def test_apply_recommendation_on_a_calibrated_model_loaded_from_json(drifted) -> None:
    # MON-5: offset_to() needs the calibration scores, which JSON does not carry:
    # this used to raise RuntimeError for any model loaded from disk.
    wrapped, X = _wrapped(600)
    loaded = CalibratedModel.from_json(wrapped.to_json(), model=_StubModel())
    action = drifted.apply_recommendation(target=loaded)
    assert action.kind == "re-offset"
    expected = action.offset.transform(loaded.predict_proba(X))
    np.testing.assert_allclose(action.composed.predict_proba(X), expected, atol=1e-12)
    assert loaded.offsets_ == []  # target untouched
    assert action.audit["new_target_fingerprint"] == action.composed.fingerprint()


def test_apply_recommendation_closes_the_drift_loop() -> None:
    """End-to-end: alarm -> apply_recommendation() -> feed the corrected
    stream into the fresh monitor -> no further alarm."""
    mon = CalibrationMonitor(alpha=0.05)
    alarm_batch = None
    batches = []
    for k in range(20):
        shift = 0.5 if k >= 10 else 0.0
        y, p = _batch(n=2000, shift=shift, seed=k)
        batches.append((y, p))
        step = mon.update(y, p, label=f"m{k}")
        if step.alarm and alarm_batch is None:
            alarm_batch = k
    assert alarm_batch is not None and alarm_batch >= 10

    action = mon.apply_recommendation()
    assert action.kind == "re-offset"

    for k in range(alarm_batch + 1, 20):
        y, p = batches[k]
        step = action.monitor.update(y, action.offset.transform(p), label=f"m{k}")
        assert not step.alarm
