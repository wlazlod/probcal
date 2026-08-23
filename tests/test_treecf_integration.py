"""Joint smoke test with treecf (spec W11 P4). Skipped without treecf.

Correctness is always re-verified through sklearn's own predict — never
trusted from treecf's report. These tests surfaced a treecf boundary-routing
defect (sklearn casts inputs to float32 before comparing against float64
thresholds; treecf routed in float64, so an x_cf placed exactly on a split
threshold could flip trees), fixed in treecf 0.2.3 (treecf#21) — and the
assertions below keep going through the model, not the report, regardless.
"""

import numpy as np
import pytest

pytest.importorskip("treecf")
pytest.importorskip("sklearn")

from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402
from treecf import Explainer, Target  # noqa: E402

from probcal import BetaCalibrator, Chain, IsotonicCalibrator, LogitOffset  # noqa: E402
from probcal._math import expit  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(3)
    n = 3000
    X = np.column_stack([rng.normal(size=n), rng.normal(size=n), rng.uniform(0, 1, n)])
    z = 1.3 * X[:, 0] - 0.9 * X[:, 1] + 0.8 * X[:, 2] - 2.2
    y = (rng.random(n) < expit(z)).astype(int)
    model = GradientBoostingClassifier(n_estimators=60, max_depth=3, random_state=0).fit(X, y)
    scores = model.predict_proba(X)[:, 1]
    cal = BetaCalibrator().fit(scores, y.astype(float))
    return model, X, y, scores, cal


def _pick_high_risk(model, X, cal, threshold=0.05):
    p = cal.predict_proba(model.predict_proba(X)[:, 1])
    idx = int(np.argmax(p))
    return X[idx], p[idx]


def _require_treecf_0_2_4() -> None:
    # Fail, not skip: a stale environment silently dropping these assertions
    # would defeat the loop-close this module pins.
    import treecf

    if tuple(int(p) for p in treecf.__version__.split(".")[:3]) < (0, 2, 4):
        pytest.fail("probcal[treecf] requires treecf>=0.2.4 — update the pin")


def test_counterfactual_hits_calibrated_target(setup) -> None:
    model, X, y, scores, cal = setup
    x0, p0 = _pick_high_risk(model, X, cal)
    assert p0 > 0.05
    exp = Explainer(model=model, background=X[:500])
    res = exp.explain(
        x0, target=Target.calibrated(cal, op="<=", value=0.05), seed=0, backend="exact"
    )
    assert hasattr(res, "x_cf"), f"expected a counterfactual, got {type(res).__name__}"
    x_cf = np.asarray(res.x_cf, dtype=np.float64)
    p_cf = cal.predict_proba(model.predict_proba(x_cf.reshape(1, -1))[:, 1])
    assert p_cf[0] <= 0.05 + 1e-9  # closed interval per the P3 contract


def test_score_calibrated_readout(setup) -> None:
    _require_treecf_0_2_4()
    model, X, y, scores, cal = setup
    x0, _ = _pick_high_risk(model, X, cal)
    exp = Explainer(model=model, background=X[:500])
    res = exp.explain(
        x0, target=Target.calibrated(cal, op="<=", value=0.05), seed=0, backend="exact"
    )
    assert hasattr(res, "x_cf")
    # The read-out is presentational; the raw interval decided. It must agree
    # with the independent model -> calibrator recompute exactly.
    x_cf = np.asarray(res.x_cf, dtype=np.float64)
    p_cf = cal.predict_proba(model.predict_proba(x_cf.reshape(1, -1))[:, 1])
    assert res.score_calibrated is not None
    assert res.score_calibrated == pytest.approx(p_cf[0], abs=1e-12)


def test_certificate_provenance_loop(setup) -> None:
    _require_treecf_0_2_4()
    model, X, y, scores, cal = setup
    off = LogitOffset(delta=0.5).fit(cal.predict_proba(scores))
    chain = Chain([cal, off])
    x0, _ = _pick_high_risk(model, X, cal)
    exp = Explainer(model=model, background=X[:500])
    target = Target.calibrated(chain, op="<=", value=0.05, buffer_logit=0.05)
    res = exp.explain(x0, target=target, seed=0, backend="exact")
    assert hasattr(res, "x_cf")

    cert = exp.certificate(x0, res, target)
    block = cert["target"]["calibrator"]
    assert block["fingerprint"] == chain.fingerprint()
    assert block["buffer_logit"] == 0.05

    # The self-contained pair: certificate + calibrator JSON. A validator
    # rebuilds the chain from its serialized form and closes the loop.
    rebuilt = Chain.from_json(chain.to_json())
    report = exp.check_certificate(cert, calibrator=rebuilt)
    assert report["calibrator_match"] is True
    assert report["mismatches"] == []

    other = Chain([BetaCalibrator().fit(scores[:1500], y[:1500].astype(float)), off])
    report_other = exp.check_certificate(cert, calibrator=other)
    assert report_other["calibrator_match"] is False

    # No-argument call keeps the 0.2.3 report shape: no calibrator_match key.
    assert "calibrator_match" not in exp.check_certificate(cert)


def test_chain_target_after_macro_shift(setup) -> None:
    model, X, y, scores, cal = setup
    off = LogitOffset(delta=0.5).fit(cal.predict_proba(scores))
    chain = Chain([cal, off])
    x0, _ = _pick_high_risk(model, X, cal)
    exp = Explainer(model=model, background=X[:500])
    res = exp.explain(
        x0, target=Target.calibrated(chain, op="<=", value=0.05), seed=0, backend="exact"
    )
    assert hasattr(res, "x_cf")
    x_cf = np.asarray(res.x_cf, dtype=np.float64)
    p_cf = chain.predict_proba(model.predict_proba(x_cf.reshape(1, -1))[:, 1])
    assert p_cf[0] <= 0.05 + 1e-9


def test_isotonic_plateau_target_lands_on_block_boundary(setup) -> None:
    model, X, y, scores, cal = setup
    iso = IsotonicCalibrator().fit(scores, y.astype(float))
    levels = np.asarray(iso.block_mean_)
    inner = levels[(levels > levels.min()) & (levels < levels.max())]
    t = float(inner[len(inner) // 2])
    x0, _ = _pick_high_risk(model, X, iso)
    exp = Explainer(model=model, background=X[:500])
    res = exp.explain(x0, target=Target.calibrated(iso, op="<=", value=t), seed=0, backend="exact")
    assert hasattr(res, "x_cf")
    x_cf = np.asarray(res.x_cf, dtype=np.float64)
    p_cf = iso.predict_proba(model.predict_proba(x_cf.reshape(1, -1))[:, 1])
    assert p_cf[0] <= t + 1e-12  # the plateau level itself qualifies (closed)


def test_bands_grade_recourse(setup) -> None:
    model, X, y, scores, cal = setup
    x0, _ = _pick_high_risk(model, X, cal)
    exp = Explainer(model=model, background=X[:500])
    bands = {"A": (0.0, 0.02), "B": (0.02, 0.08)}
    out = exp.explain(x0, target=Target.bands(bands, space="calibrated", calibrator=cal), seed=0)
    assert set(out) == {"A", "B"}
