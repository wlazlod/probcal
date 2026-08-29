"""Regenerate the golden serialization files (run manually, outputs committed).

Usage: ``uv run python tests/golden/_generate.py``

One JSON per registered class: ``{"object": to_dict(), "query": [...],
"expected": [...]}`` with fixed seeds identical to tests/test_serialize.py's
battery. These files are the enforcement of the compatibility promise: every
0.x release must load them (schema 1) and reproduce ``expected`` on
``query`` to within 1e-12. Regenerate ONLY when the schema version bumps —
that bump requires a converter and a changelog entry.
"""

import json
import pathlib

import numpy as np

from probcal import BetaCalibrator, make_pd_portfolio
from probcal._registry import SERIALIZABLE

# Smaller than the battery's fixtures on purpose: the heavy classes (ENIR
# path solutions, CVAP's fold IVAPs) serialize O(n) state and these files
# are committed.
_D = make_pd_portfolio(n=400, random_state=7)
_Q = make_pd_portfolio(n=150, random_state=8).scores


class _StubModel:
    """Deterministic sklearn-free model over a single score column.

    Mirrored in tests/test_golden.py — keep the two definitions identical.
    """

    def fit(self, X, y):  # noqa: ARG002
        return self

    def predict_proba(self, X):
        s = np.asarray(X)[:, 0]
        return np.column_stack([1.0 - s, s])

    def get_params(self):
        return {"stub": True}


def _monitor_batches():
    out = []
    for k in range(3):
        d = make_pd_portfolio(n=300, event_rate=0.1, random_state=50 + k)
        rng = np.random.default_rng(150 + k)
        y = (rng.random(300) < d.scores).astype(float)
        out.append((y, d.scores))
    return out


def _fitted(name: str):
    cls = SERIALIZABLE[name]
    if name == "CalibrationMonitor":
        mon = cls(delta_ci_grid=(-2.0, 2.0, 41))
        for k, (y, p) in enumerate(_monitor_batches()):
            mon.update(y, p, label=f"g{k}")
        return mon
    if name == "Chain":
        from probcal import Chain, LogitOffset

        cal = BetaCalibrator().fit(_D.scores, _D.y)
        off = LogitOffset(delta=0.2).fit(cal.predict_proba(_D.scores))
        return Chain([cal, off])
    if name == "LogitOffset":
        return cls(delta=0.3).fit(_D.scores)
    if name == "SegmentedCalibrator":
        segments = np.array(["a", "b", "c"])[np.arange(400) % 3]
        return cls().fit(_D.scores, _D.y, segments=segments)
    if name == "CalibratedModel":
        wrapped = cls(_StubModel(), BetaCalibrator(), flow="prefit").fit(
            _D.scores.reshape(-1, 1), _D.y
        )
        wrapped.offset_to(delta=0.15)
        return wrapped
    if name == "AppliedAction":
        from probcal._math import expit, logit

        mon = SERIALIZABLE["CalibrationMonitor"](delta_ci_grid=(-2.0, 2.0, 41))
        for k, (y, p) in enumerate(_monitor_batches()):
            mon.update(y, p, label=f"g{k}")
        for k in range(3):
            d = make_pd_portfolio(n=300, event_rate=0.1, random_state=50 + k)
            p = d.scores
            rng = np.random.default_rng(160 + k)
            y = (rng.random(300) < expit(logit(p) + 0.8)).astype(float)
            mon.update(y, p, label=f"drift{k}")
        return mon.apply_recommendation(target=None)
    return cls().fit(_D.scores, _D.y)


def _predict(obj, q):
    if type(obj).__name__ == "CalibrationMonitor":
        return np.asarray([s.e_global for s in obj.steps_])
    if type(obj).__name__ == "LogitOffset":
        return obj.transform(q)
    if type(obj).__name__ == "SegmentedCalibrator":
        segments = np.array(["a", "b", "c"])[np.arange(len(q)) % 3]
        return obj.predict_proba(q, segments=segments)
    if type(obj).__name__ == "CalibratedModel":
        return obj.predict_proba(q.reshape(-1, 1))
    if type(obj).__name__ == "AppliedAction":
        return np.asarray([obj.offset.delta_] if obj.offset is not None else [0.0])
    return obj.predict_proba(q)


def main() -> None:
    out_dir = pathlib.Path(__file__).parent
    for name in sorted(SERIALIZABLE):
        obj = _fitted(name)
        payload = {
            "object": obj.to_dict(),
            "query": _Q.tolist(),
            "expected": _predict(obj, _Q).tolist(),
        }
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
