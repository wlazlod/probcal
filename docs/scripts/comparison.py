"""W13 comparison benchmark: probcal vs sklearn / netcal / betacal.

Usage: ``uv run python docs/scripts/comparison.py [--fast]``
Requires ``probcal[bench]`` (pins recorded in the output header). Datasets
download from OpenML through scikit-learn's ``fetch_openml`` cache.

Protocol per dataset: 50/25/25 train/calibration/test split (seeded);
HistGradientBoostingClassifier base model trained once; every calibrator
fits on the calibration split's scores and is evaluated on the test split
with ``probcal.metrics.evaluate`` bootstrap CIs (log loss, ECE-sweep, ICI),
the Jeffreys per-grade backtest pass rate (six fixed PD bands), and wall
fit time. The output table is pasted into ``docs/benchmarks/comparison.md``.
"""

import sys
import time
import warnings

import numpy as np

from probcal import (
    BetaCalibrator,
    CalibratorSelector,
    SplineCalibrator,
    VennAbersCalibrator,
)
from probcal.metrics import evaluate, jeffreys_grade_test

FAST = "--fast" in sys.argv
N_BOOT = 100 if FAST else 200
TEST_CAP = 8000 if FAST else 20000

DATASETS = [
    # (openml name, version, positive label) — event rates ~1.5% to 30%
    ("Satellite", 1, "Anomaly"),
    ("mammography", 1, "1"),
    ("bank-marketing", 1, "2"),
    ("adult", 2, ">50K"),
    ("credit-g", 1, "bad"),
]

_GRADE_EDGES = np.array([0.0, 0.005, 0.01, 0.02, 0.05, 0.15, 1.0])
_GRADE_LABELS = np.array(["A", "B", "C", "D", "E", "F"])


def _load(name: str, version: int, pos: str):
    from sklearn.datasets import fetch_openml

    data = fetch_openml(name=name, version=version, as_frame=True, parser="auto")
    df = data.frame
    y = (data.target.astype(str) == pos).to_numpy(dtype=float)
    X = df.drop(columns=[data.target_names[0]])
    num = X.select_dtypes("number")
    cat = X.select_dtypes(exclude="number")
    parts = [num.to_numpy(float)] if len(num.columns) else []
    for col in cat.columns:
        codes = cat[col].astype("category").cat.codes.to_numpy(float)
        parts.append(codes[:, None])
    return np.hstack(parts), y


def _grade_pass_rate(y, p) -> float:
    grades = _GRADE_LABELS[np.clip(np.searchsorted(_GRADE_EDGES, p, side="right") - 1, 0, 5)]
    res = jeffreys_grade_test(y, p, grades)
    lights = np.asarray(res.p_value) > 0.05
    return float(np.mean(lights))


def _methods(seed: int):
    """name -> fit(s, y) returning predict(s_new) -> calibrated p."""
    out: dict[str, object] = {}

    def probcal_method(proto):
        def fit(s, y):
            cal = type(proto)(**proto.get_params()).fit(s, y)
            return cal.predict_proba

        return fit

    out["probcal beta (abm)"] = probcal_method(BetaCalibrator())
    out["probcal spline"] = probcal_method(SplineCalibrator())
    out["probcal IVAP"] = probcal_method(VennAbersCalibrator())
    out["probcal selector"] = probcal_method(CalibratorSelector())

    def sk_sigmoid(s, y):
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(max_iter=1000)
        z = np.log(np.clip(s, 1e-12, 1 - 1e-12) / np.clip(1 - s, 1e-12, 1))
        lr.fit(z[:, None], y)
        return lambda s_new: lr.predict_proba(
            np.log(np.clip(s_new, 1e-12, 1 - 1e-12) / np.clip(1 - s_new, 1e-12, 1))[:, None]
        )[:, 1]

    def sk_isotonic(s, y):
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(s, y)
        return lambda s_new: np.asarray(iso.predict(s_new))

    out["sklearn sigmoid"] = sk_sigmoid
    out["sklearn isotonic"] = sk_isotonic

    try:
        from netcal.binning import BBQ, ENIR
        from netcal.scaling import BetaCalibration as NetcalBeta

        def netcal_method(ctor):
            def fit(s, y):
                m = ctor()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.fit(s.astype(np.float64), y.astype(int))
                return lambda s_new: np.clip(
                    np.asarray(m.transform(s_new.astype(np.float64)), dtype=float), 0.0, 1.0
                )

            return fit

        out["netcal BBQ"] = netcal_method(BBQ)
        out["netcal ENIR"] = netcal_method(ENIR)
        out["netcal beta"] = netcal_method(NetcalBeta)
    except ImportError:
        print("netcal unavailable — its rows are skipped")

    try:
        from betacal import BetaCalibration as BetacalBeta

        def betacal_fit(s, y):
            m = BetacalBeta(parameters="abm")
            m.fit(s[:, None], y)
            return lambda s_new: np.asarray(m.predict(s_new), dtype=float)

        out["betacal (abm)"] = betacal_fit
    except ImportError:
        print("betacal unavailable — its row is skipped")

    return out


def run_dataset(name: str, version: int, pos: str, seed: int = 0) -> list[dict]:
    from sklearn.ensemble import HistGradientBoostingClassifier

    X, y = _load(name, version, pos)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    X, y = X[order], y[order]
    n = len(y)
    i_tr, i_ca = int(0.5 * n), int(0.75 * n)
    model = HistGradientBoostingClassifier(random_state=0).fit(X[:i_tr], y[:i_tr])
    s_cal, y_cal = model.predict_proba(X[i_tr:i_ca])[:, 1], y[i_tr:i_ca]
    s_test, y_test = model.predict_proba(X[i_ca:])[:, 1], y[i_ca:]
    if len(y_test) > TEST_CAP:
        s_test, y_test = s_test[:TEST_CAP], y_test[:TEST_CAP]

    rows = []
    for method, fit in _methods(seed).items():
        t0 = time.perf_counter()
        try:
            predict = fit(np.clip(s_cal, 1e-12, 1 - 1e-12), y_cal)
        except Exception as exc:
            rows.append({"dataset": name, "method": method, "error": f"{type(exc).__name__}"})
            continue
        fit_s = time.perf_counter() - t0
        p = np.clip(np.asarray(predict(np.clip(s_test, 1e-12, 1 - 1e-12)), float), 0.0, 1.0)
        rep = evaluate(y_test, p, n_boot=N_BOOT, metrics=("log_loss", "ece_sweep", "ici"))
        vals = dict(zip(rep.names, rep.values, strict=True))
        los = dict(zip(rep.names, rep.ci_low, strict=True))
        his = dict(zip(rep.names, rep.ci_high, strict=True))
        rows.append(
            {
                "dataset": f"{name} ({y.mean():.1%})",
                "method": method,
                "log_loss": (vals["log_loss"], los["log_loss"], his["log_loss"]),
                "ece_sweep": (vals["ece_sweep"], los["ece_sweep"], his["ece_sweep"]),
                "ici": (vals["ici"], los["ici"], his["ici"]),
                "grade_pass": _grade_pass_rate(y_test, p),
                "fit_s": fit_s,
            }
        )
    return rows


def main() -> None:
    import betacal as _bc  # noqa: F401 - version pins recorded below
    import netcal as _nc
    import sklearn as _sk

    print(
        f"pins: scikit-learn {_sk.__version__}, netcal {_nc.__version__}, "
        f"betacal {getattr(_bc, '__version__', 'unknown')}, n_boot={N_BOOT}"
    )
    all_rows: list[dict] = []
    for name, version, pos in DATASETS:
        print(f"\n### {name}", flush=True)
        try:
            rows = run_dataset(name, version, pos)
        except Exception as exc:
            print(f"| (dataset unavailable: {type(exc).__name__}: {exc}) |")
            continue
        all_rows.extend(rows)
        print("| method | log loss | ECE-sweep | ICI | grade pass | fit s |")
        print("|---|---|---|---|---|---|")
        for r in rows:
            if "error" in r:
                print(f"| {r['method']} | fit failed: {r['error']} | | | | |")
                continue

            def ci(t):
                return f"{t[0]:.4f} [{t[1]:.4f}, {t[2]:.4f}]"

            print(
                f"| {r['method']} | {ci(r['log_loss'])} | {ci(r['ece_sweep'])} | "
                f"{ci(r['ici'])} | {r['grade_pass']:.0%} | {r['fit_s']:.2f} |"
            )


if __name__ == "__main__":
    main()
