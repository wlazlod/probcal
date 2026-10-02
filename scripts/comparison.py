"""Comparison benchmark: probcal vs sklearn / netcal / betacal.

Usage: ``uv run python docs/scripts/comparison.py [--fast] [--readme]``
Requires ``probcal[bench]`` (pins recorded in the output header). Datasets
download from OpenML through scikit-learn's ``fetch_openml`` cache.

Protocol per dataset: 50/25/25 train/calibration/test split (seeded); one
base model trained once on the train split. The five general datasets use
``HistGradientBoostingClassifier``; the credit-card dataset uses a scorecard
(quantile-binned one-hot features, logistic regression, logit rounded to
integer points at PDO=20), so its scores carry the ties a deployed scorecard
has. Every calibrator fits on the calibration split's scores and is evaluated
on the test split with ``probcal.metrics.evaluate`` bootstrap CIs (log loss,
ECE-sweep, ICI), the Jeffreys per-grade backtest over a fixed PD-band
masterscale (grades assigned from each method's own calibrated PD), and wall
fit time. Score ties and grade sizes are reported per dataset. ``--readme``
runs the credit-card dataset only and appends the README-sized table. The
output is pasted into ``docs/benchmarks/comparison.md``.
"""

import contextlib
import io
import sys
import time
import warnings

import numpy as np

from probcal import (
    BetaCalibrator,
    CalibratorSelector,
    IsotonicCalibrator,
    Masterscale,
    PlattCalibrator,
    SplineCalibrator,
    VennAbersCalibrator,
)
from probcal.metrics import evaluate, jeffreys_grade_test

FAST = "--fast" in sys.argv
README = "--readme" in sys.argv
N_BOOT = 100 if FAST else 200
TEST_CAP = 8000 if FAST else 20000

CREDIT = "default-of-credit-card-clients"

DATASETS = [
    # (openml name, version, positive label, base model) — event rates ~1.5% to 30%
    (CREDIT, 1, "1", "scorecard"),
    ("Satellite", 1, "Anomaly", "hgb"),
    ("mammography", 1, "1", "hgb"),
    ("bank-marketing", 1, "2", "hgb"),
    ("adult", 2, ">50K", "hgb"),
    ("credit-g", 1, "bad", "hgb"),
]

_DEFAULT_SCALE = (np.array([0.0, 0.005, 0.01, 0.02, 0.05, 0.15, 1.0]), list("ABCDEF"))
_MASTERSCALES = {
    CREDIT: (np.array([0.0, 0.03, 0.06, 0.10, 0.15, 0.25, 0.40, 0.60, 1.0]), list("ABCDEFGH")),
}

_PDO = 20.0  # scorecard points to double the odds
_PDO_FACTOR = _PDO / np.log(2.0)

README_METHODS = (
    "probcal Platt",
    "probcal beta (abm)",
    "probcal isotonic",
    "sklearn sigmoid",
    "sklearn isotonic",
    "netcal beta",
    "netcal BBQ",
)


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


def _fit_base(X_tr, y_tr, base: str):
    """Train the base model once; return ``score(X) -> s in (0, 1)``."""
    if base == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(random_state=0).fit(X_tr, y_tr)
        return lambda X: model.predict_proba(X)[:, 1]
    if base == "scorecard":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import KBinsDiscretizer

        pipe = make_pipeline(
            KBinsDiscretizer(n_bins=5, encode="onehot", strategy="quantile", subsample=None),
            LogisticRegression(max_iter=2000),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # degenerate bins on low-cardinality columns
            pipe.fit(X_tr, y_tr)

        def score(X):
            points = np.round(_PDO_FACTOR * pipe.decision_function(X))
            return 1.0 / (1.0 + np.exp(-points / _PDO_FACTOR))

        return score
    raise ValueError(f"unknown base model {base!r}")


def _tie_stats(s) -> dict:
    _, counts = np.unique(s, return_counts=True)
    return {
        "n": int(len(s)),
        "distinct": int(len(counts)),
        "largest_block": int(counts.max()),
        "median_block": float(np.median(counts)),
    }


def _grade_stats(y, p, edges, labels) -> dict:
    """Jeffreys backtest over the masterscale; grades from the calibrated PD itself."""
    ms = Masterscale({lab: (float(edges[i]), float(edges[i + 1])) for i, lab in enumerate(labels)})
    res = jeffreys_grade_test(y, p, ms)
    tab = ms.table(y, p)
    sizes = [int(n) if n > 0 else None for n in tab.n]
    passed = int(np.sum(np.asarray(res.p_value) > 0.05))
    return {
        "passed": passed,
        "total": len(res.grades),
        "sizes": sizes,
        "n_small": int(np.sum(np.asarray(res.n) < 30)),
        "n_zero": int(np.sum(np.asarray(res.k) == 0)),
    }


def _methods():
    """name -> fit(s, y) returning predict(s_new) -> calibrated p."""
    out: dict[str, object] = {}

    def probcal_method(proto):
        def fit(s, y):
            cal = type(proto)(**proto.get_params()).fit(s, y)
            return cal.predict_proba

        return fit

    out["probcal Platt"] = probcal_method(PlattCalibrator())
    out["probcal beta (abm)"] = probcal_method(BetaCalibrator())
    out["probcal isotonic"] = probcal_method(IsotonicCalibrator())
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
                with (
                    warnings.catch_warnings(),
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
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


def run_dataset(name: str, version: int, pos: str, base: str, seed: int = 0):
    from sklearn.metrics import roc_auc_score

    X, y = _load(name, version, pos)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    X, y = X[order], y[order]
    n = len(y)
    i_tr, i_ca = int(0.5 * n), int(0.75 * n)
    score = _fit_base(X[:i_tr], y[:i_tr], base)
    s_cal, y_cal = score(X[i_tr:i_ca]), y[i_tr:i_ca]
    s_test, y_test = score(X[i_ca:]), y[i_ca:]
    if len(y_test) > TEST_CAP:
        s_test, y_test = s_test[:TEST_CAP], y_test[:TEST_CAP]
    edges, labels = _MASTERSCALES.get(name, _DEFAULT_SCALE)

    diag = {
        "event_rate": float(y.mean()),
        "n": n,
        "n_test": int(len(y_test)),
        "base": base,
        "auc": float(roc_auc_score(y_test, s_test)),
        "ties_cal": _tie_stats(s_cal),
        "labels": labels,
        "edges": edges,
    }

    rows = []
    for method, fit in _methods().items():
        t0 = time.perf_counter()
        try:
            predict = fit(np.clip(s_cal, 1e-12, 1 - 1e-12), y_cal)
            fit_s = time.perf_counter() - t0
            p = np.clip(np.asarray(predict(np.clip(s_test, 1e-12, 1 - 1e-12)), float), 0.0, 1.0)
            rep = evaluate(y_test, p, n_boot=N_BOOT, metrics=("log_loss", "ece_sweep", "ici"))
            grades = _grade_stats(y_test, p, edges, labels)
        except Exception as exc:
            rows.append({"method": method, "error": f"{type(exc).__name__}"})
            continue
        vals = dict(zip(rep.names, rep.values, strict=True))
        los = dict(zip(rep.names, rep.ci_low, strict=True))
        his = dict(zip(rep.names, rep.ci_high, strict=True))
        rows.append(
            {
                "method": method,
                "log_loss": (vals["log_loss"], los["log_loss"], his["log_loss"]),
                "ece_sweep": (vals["ece_sweep"], los["ece_sweep"], his["ece_sweep"]),
                "ici": (vals["ici"], los["ici"], his["ici"]),
                "grades": grades,
                "levels": int(len(np.unique(p))),
                "fit_s": fit_s,
            }
        )
    return rows, diag


def _ci(t) -> str:
    if not (np.isfinite(t[1]) and np.isfinite(t[2])):
        return f"{t[0]:.4f} (CI undefined)"
    return f"{t[0]:.4f} [{t[1]:.4f}, {t[2]:.4f}]"


def _sizes(g: dict) -> str:
    return "/".join("–" if s is None else str(s) for s in g["sizes"])


def _print_dataset(name: str, rows: list[dict], diag: dict) -> None:
    t = diag["ties_cal"]
    print(f"\n### {name} ({diag['event_rate']:.1%} event rate, n={diag['n']:,})\n")
    print(
        f"base: {diag['base']}, test AUC {diag['auc']:.3f}; scores: {t['distinct']:,} distinct "
        f"on n_cal={t['n']:,} ({t['distinct'] / t['n']:.1%}); largest tie block "
        f"{t['largest_block']}, median block {t['median_block']:.0f}; n_test={diag['n_test']:,}\n"
    )
    print("| method | log loss | ECE-sweep | ICI | grade pass | fit s |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        if "error" in r:
            print(f"| {r['method']} | fit failed: {r['error']} | | | | |")
            continue
        g = r["grades"]
        print(
            f"| {r['method']} | {_ci(r['log_loss'])} | {_ci(r['ece_sweep'])} | "
            f"{_ci(r['ici'])} | {g['passed']}/{g['total']} | {r['fit_s']:.2f} |"
        )
    labels = "/".join(diag["labels"])
    print(f"\n| method | output levels | grade sizes ({labels}) | n<30 | zero-default | pass |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        if "error" in r:
            continue
        g = r["grades"]
        print(
            f"| {r['method']} | {r['levels']} | {_sizes(g)} | {g['n_small']} | "
            f"{g['n_zero']} | {g['passed']}/{g['total']} |"
        )


def _print_readme(rows: list[dict], diag: dict) -> None:
    t = diag["ties_cal"]
    print("\n### README block\n")
    print(
        f"scores: {t['distinct']:,} distinct on n_cal={t['n']:,}; largest tie block "
        f"{t['largest_block']}; test AUC {diag['auc']:.3f}; n_test={diag['n_test']:,}"
    )
    by_name = {r["method"]: r for r in rows}
    sizes_a = {
        m: by_name[m]["grades"]["sizes"][0]
        for m in README_METHODS
        if m in by_name and "error" not in by_name[m]
    }
    print(f"grade A size per README method: {sizes_a}\n")
    print("| method | log loss | ICI | grade pass | n<30 | zero-default | fit s |")
    print("|---|---|---|---|---|---|---|")
    for m in README_METHODS:
        r = by_name.get(m)
        if r is None:
            continue
        if "error" in r:
            print(f"| {m} | fit failed: {r['error']} | | | | | |")
            continue
        g = r["grades"]
        print(
            f"| {m} | {_ci(r['log_loss'])} | {_ci(r['ici'])} | {g['passed']}/{g['total']} | "
            f"{g['n_small']} | {g['n_zero']} | {r['fit_s']:.2f} |"
        )


def main() -> None:
    import betacal as _bc  # noqa: F401 - version pins recorded below
    import netcal as _nc
    import pandas as _pd
    import sklearn as _sk

    print(
        f"pins: scikit-learn {_sk.__version__}, netcal {_nc.__version__}, "
        f"betacal {getattr(_bc, '__version__', 'unknown')}, pandas {_pd.__version__}, "
        f"n_boot={N_BOOT}"
    )
    datasets = [d for d in DATASETS if d[0] == CREDIT] if README else DATASETS
    for name, version, pos, base in datasets:
        print(f"(running {name})", file=sys.stderr, flush=True)
        try:
            rows, diag = run_dataset(name, version, pos, base)
        except Exception as exc:
            print(f"\n### {name}\n| (dataset unavailable: {type(exc).__name__}: {exc}) |")
            continue
        _print_dataset(name, rows, diag)
        if README and name == CREDIT:
            _print_readme(rows, diag)


if __name__ == "__main__":
    main()
