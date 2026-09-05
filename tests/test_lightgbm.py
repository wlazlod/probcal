"""End-to-end: a real LightGBM predict_proba matrix through the calibration pipeline."""

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("lightgbm")

from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from probcal import make_pd_portfolio  # noqa: E402
from probcal.sklearn import SklearnCalibrator, SklearnOffset  # noqa: E402


def test_lightgbm_predict_proba_matrix_through_the_pipeline():
    rng = np.random.default_rng(0)
    d = make_pd_portfolio(n=2000, random_state=7)
    X = np.column_stack([d.scores, rng.normal(size=len(d.scores))])
    model = LGBMClassifier(n_estimators=20, random_state=0).fit(X[:1000], d.y[:1000])

    P_cal = model.predict_proba(X[1000:1600])  # the (n, 2) matrix, no wrapper
    P_new = model.predict_proba(X[1600:])
    target_mean = float(d.y.mean())
    pipe = Pipeline([("cal", SklearnCalibrator()), ("off", SklearnOffset(target_mean=target_mean))])
    pipe.fit(P_cal, d.y[1000:1600])

    out = pipe.predict_proba(P_new)
    assert out.shape == (len(P_new), 2)
    assert np.all((out >= 0.0) & (out <= 1.0))
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
