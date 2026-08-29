"""Golden-file guarantee: every 0.x release reads schema 1 (spec W5).

The JSONs in tests/golden/ were written by the release that introduced
serialization and are committed; this suite proves the current build still
loads them and reproduces their predictions. Tolerance is allclose at
1e-12, not bit-equality: goldens must survive numpy/platform drift
(DECISIONS 73). Regeneration (tests/golden/_generate.py) is legitimate only
alongside a schema bump with a converter and a DECISIONS entry.
"""

import json
import pathlib

import numpy as np
import pytest

from probcal import CalibratedModel
from probcal._registry import SERIALIZABLE, load
from probcal._serialize import check_schema

_GOLDEN_DIR = pathlib.Path(__file__).parent / "golden"
_GOLDEN_FILES = sorted(_GOLDEN_DIR.glob("*.json"))


class _StubModel:
    """Mirrors tests/golden/_generate.py — keep the two definitions identical."""

    def fit(self, X, y):  # noqa: ARG002
        return self

    def predict_proba(self, X):
        s = np.asarray(X)[:, 0]
        return np.column_stack([1.0 - s, s])

    def get_params(self):
        return {"stub": True}


def test_every_registered_class_has_a_golden() -> None:
    names = {p.stem for p in _GOLDEN_FILES}
    assert names == set(SERIALIZABLE), (
        "golden coverage drifted from the registry; regenerate ONLY with a "
        "schema bump + converter + DECISIONS entry"
    )


@pytest.mark.parametrize("path", _GOLDEN_FILES, ids=lambda p: p.stem)
def test_golden_loads_and_reproduces(path: pathlib.Path) -> None:
    payload = json.loads(path.read_text())
    d = payload["object"]
    check_schema(d)
    q = np.asarray(payload["query"], dtype=np.float64)
    if path.stem == "CalibrationMonitor":
        obj = load(d)
        result = np.asarray([s.e_global for s in obj.steps_])
    elif path.stem == "CalibratedModel":
        obj = CalibratedModel.from_dict(d, model=_StubModel())
        result = obj.predict_proba(q.reshape(-1, 1))
    elif path.stem == "AppliedAction":
        obj = load(d)
        result = np.asarray([obj.offset.delta_] if obj.offset is not None else [0.0])
    elif path.stem == "SegmentedCalibrator":
        obj = load(d)
        segments = np.asarray(["a", "b", "c"])[np.arange(len(q)) % 3]
        result = obj.predict_proba(q, segments=segments)
    else:
        obj = load(d)
        result = obj.transform(q) if path.stem == "LogitOffset" else obj.predict_proba(q)
    expected = np.asarray(payload["expected"], dtype=np.float64)
    np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-12)
