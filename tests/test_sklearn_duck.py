"""Bare-core duck typing against sklearn >= 1.6.

These tests use no adapter: a plain probcal calibrator is handed straight to
``sklearn.base.clone``, ``sklearn.utils.get_tags``, ``check_is_fitted`` and
``cross_val_score``. The numpy-only import guard that keeps the hooks free of
a module-level sklearn import lives in ``tests/test_import_footprint.py``
(this module is skipped entirely when sklearn is absent).
"""

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn", minversion="1.6")

from sklearn.base import clone  # noqa: E402
from sklearn.exceptions import NotFittedError  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402
from sklearn.utils import get_tags  # noqa: E402
from sklearn.utils.validation import check_is_fitted  # noqa: E402

from probcal import BetaCalibrator, Chain, LogitOffset  # noqa: E402
from probcal._registry import SERIALIZABLE  # noqa: E402
from probcal.base import BaseCalibrator  # noqa: E402

CALIBRATOR_CLASSES = sorted(
    (c for c in SERIALIZABLE.values() if isinstance(c, type) and issubclass(c, BaseCalibrator)),
    key=lambda c: c.__name__,
)


def _data(n: int = 300, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    s = rng.uniform(0.05, 0.95, n)
    y = rng.binomial(1, s).astype(float)
    return s, y


def test_get_tags_reports_requires_fit() -> None:
    tags = get_tags(BetaCalibrator())
    assert tags.requires_fit is True
    assert tags.estimator_type is None
    assert tags.target_tags.required is True
    assert tags.input_tags.two_d_array is False


@pytest.mark.parametrize("cls", CALIBRATOR_CLASSES, ids=lambda c: c.__name__)
def test_clone_every_registered_calibrator(cls: type) -> None:
    original = cls()
    copy = clone(original)
    assert type(copy) is cls
    assert copy.get_params() == original.get_params()
    assert copy.__sklearn_is_fitted__() is False


def test_clone_of_fitted_calibrator_is_unfitted() -> None:
    s, y = _data()
    fitted = BetaCalibrator(variant="ab").fit(s, y)
    copy = clone(fitted)
    assert copy.get_params() == {"variant": "ab"}
    assert copy.fitted_ is False
    with pytest.raises(RuntimeError, match="not fitted"):
        copy.predict_proba(s)


def test_check_is_fitted_before_and_after_fit() -> None:
    s, y = _data()
    cal = BetaCalibrator()
    with pytest.raises(NotFittedError):
        check_is_fitted(cal)
    cal.fit(s, y)
    check_is_fitted(cal)


def test_chain_and_offset_report_fitted() -> None:
    # Chain and LogitOffset carry the is-fitted hook, but not the tag hook:
    # a Chain is built from already-fitted stages and has no ``fit`` method,
    # which sklearn's check_is_fitted requires of an estimator instance.
    s, y = _data()
    cal = BetaCalibrator().fit(s, y)
    offset = LogitOffset(delta=0.2)
    assert offset.__sklearn_is_fitted__() is False
    offset.fit(cal.predict_proba(s))
    assert offset.__sklearn_is_fitted__() is True
    assert Chain([cal, offset]).__sklearn_is_fitted__() is True


def test_cross_val_score_with_custom_scorer() -> None:
    s, y = _data()

    def scorer(est: BaseCalibrator, X: np.ndarray, y_true: np.ndarray) -> float:
        p = est.predict_proba(X)
        assert p.ndim == 1
        return -log_loss(y_true, p, labels=[0.0, 1.0])

    scores = cross_val_score(BetaCalibrator(), s.reshape(-1, 1), y, cv=3, scoring=scorer)
    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))
