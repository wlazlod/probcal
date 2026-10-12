"""Tests for probcal.sklearn.SklearnOffset. Skipped without sklearn."""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.pipeline import Pipeline  # noqa: E402

import probcal  # noqa: E402
from probcal import BetaCalibrator, Chain, LogitOffset, make_pd_portfolio  # noqa: E402
from probcal.sklearn import SklearnCalibrator, SklearnOffset  # noqa: E402


def test_fit_transform_matches_the_core_offset():
    p = make_pd_portfolio(n=400, random_state=7).scores
    core = LogitOffset(target_mean=0.03).fit(p)
    step = SklearnOffset(target_mean=0.03).fit(p.reshape(-1, 1))
    assert step.offset_.delta_ == core.delta_
    np.testing.assert_array_equal(step.transform(p.reshape(-1, 1))[:, 0], core.transform(p))
    proba = step.predict_proba(p.reshape(-1, 1))
    assert proba.shape == (len(p), 2)
    np.testing.assert_array_equal(proba[:, 1], core.transform(p))
    d = step.to_dict()  # serialization delegates to the inner offset
    assert d["class"] == "LogitOffset"
    assert probcal._registry.load(d).delta_ == step.offset_.delta_


def test_two_column_input_and_positive_column():
    p = make_pd_portfolio(n=400, random_state=7).scores
    m = np.column_stack([1.0 - p, p])
    a = SklearnOffset(delta=0.2).fit(m)
    b = SklearnOffset(delta=0.2, positive_column=0).fit(m[:, ::-1])
    np.testing.assert_array_equal(a.transform(m), b.transform(m[:, ::-1]))


def test_pipeline_equals_the_fitted_chain_bit_for_bit():
    d = make_pd_portfolio(n=400, random_state=7)
    q = make_pd_portfolio(n=150, random_state=8).scores
    pipe = Pipeline([("cal", SklearnCalibrator()), ("off", SklearnOffset(target_mean=0.03))])
    pipe.fit(d.scores.reshape(-1, 1), d.y)
    chain = Chain([BetaCalibrator(), LogitOffset(target_mean=0.03)]).fit(d.scores, d.y)
    np.testing.assert_array_equal(
        pipe.predict_proba(q.reshape(-1, 1))[:, 1], chain.predict_proba(q)
    )


def test_chain_prototype_as_the_final_pipeline_step():
    # pipeline end-to-end with the chain as the final step
    d = make_pd_portfolio(n=400, random_state=7)
    q = make_pd_portfolio(n=150, random_state=8).scores
    est = SklearnCalibrator(calibrator=Chain([BetaCalibrator(), LogitOffset(target_mean=0.03)]))
    pipe = Pipeline([("cal", est)]).fit(d.scores.reshape(-1, 1), d.y)
    fitted = pipe.named_steps["cal"].calibrator_
    assert isinstance(fitted, Chain) and fitted.fitted_
    manual = Chain([BetaCalibrator(), LogitOffset(target_mean=0.03)]).fit(d.scores, d.y)
    np.testing.assert_array_equal(
        pipe.predict_proba(q.reshape(-1, 1))[:, 1], manual.predict_proba(q)
    )
    # the prototype's own stages were not mutated by fitting the pipeline
    assert est.calibrator.fitted_ is False


def test_sample_weight_matches_the_weighted_core_offset():
    p = make_pd_portfolio(n=400, random_state=7).scores
    rng = np.random.default_rng(5)
    w = rng.uniform(0.5, 2.0, size=len(p))
    core = LogitOffset(target_mean=0.03).fit(p, sample_weight=w)
    step = SklearnOffset(target_mean=0.03).fit(p.reshape(-1, 1), sample_weight=w)
    assert step.offset_.delta_ == core.delta_
    # zero weight drops the row
    w = np.ones_like(p)
    w[::10] = 0.0
    kept = p[w > 0.0]
    dropped = SklearnOffset(target_mean=0.03).fit(kept.reshape(-1, 1))
    zeroed = SklearnOffset(target_mean=0.03).fit(p.reshape(-1, 1), sample_weight=w)
    assert zeroed.offset_.delta_ == dropped.offset_.delta_
