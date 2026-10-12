"""Regression tests: masterscale edges near 0/1, result equality, faceted plots, figure hygiene."""

import copy
import pickle

import numpy as np
import pytest

from probcal import Masterscale, build_masterscale, calibration_belt, make_pd_portfolio
from probcal._results import BeltResult, _values_equal

# ------------------------------------------------------------ result equality


def test_values_equal_handles_mappings_of_arrays() -> None:
    a = {0.8: (np.array([0.1, np.nan]), np.array([0.2, 0.3]))}
    b = {0.8: (np.array([0.1, np.nan]), np.array([0.2, 0.3]))}
    c = {0.8: (np.array([0.1, np.nan]), np.array([0.2, 0.4]))}
    assert _values_equal(a, b)
    assert not _values_equal(a, c)
    assert not _values_equal(a, {0.9: a[0.8]})


@pytest.fixture(scope="module")
def belt() -> BeltResult:
    d = make_pd_portfolio(n=2000, random_state=0)
    return calibration_belt(d.y, d.scores)


def test_belt_result_equality_does_not_raise(belt: BeltResult) -> None:
    d = make_pd_portfolio(n=2000, random_state=0)
    assert belt == calibration_belt(d.y, d.scores)
    other = calibration_belt(d.y, d.scores, confidence=(0.5, 0.9))
    assert belt != other


def test_belt_bands_are_read_only(belt: BeltResult) -> None:
    with pytest.raises(TypeError):
        belt.bands[0.5] = (belt.grid_p, belt.grid_p)  # type: ignore[index]
    assert set(belt.bands) == set(belt.levels)
    assert isinstance(dict(belt.bands), dict)


def test_belt_result_round_trips_through_pickle_and_copy(belt: BeltResult) -> None:
    for clone in (pickle.loads(pickle.dumps(belt)), copy.deepcopy(belt), copy.copy(belt)):
        assert clone == belt
        with pytest.raises(TypeError):
            clone.bands[0.5] = None  # type: ignore[index]


# ------------------------------------------------------------- masterscale


def test_assign_does_not_clip_probabilities_across_a_close_edge() -> None:
    low = Masterscale({"A": (0.0, 1e-13), "B": (1e-13, 1.0)})
    assert low.assign([0.0]).tolist() == ["A"]
    high = Masterscale({"A": (0.0, 1 - 1e-13), "B": (1 - 1e-13, 1.0)})
    assert high.assign([1.0]).tolist() == ["B"]


def test_assign_still_validates_probabilities() -> None:
    ms = Masterscale.from_edges([0.5])
    for bad in ([np.nan], [1.5], [-0.1]):
        with pytest.raises(ValueError):
            ms.assign(bad)
    # A two-column predict_proba matrix is still accepted.
    assert ms.assign(np.array([[0.9, 0.1], [0.2, 0.8]])).tolist() == ["G1", "G2"]


def test_table_mean_pd_uses_unclipped_probabilities() -> None:
    ms = Masterscale.from_edges([0.5])
    t = ms.table([0.0, 1.0, 1.0], [0.0, 1.0, 1.0])
    assert t.mean_pd.tolist() == [0.0, 1.0]


def test_build_masterscale_with_mass_at_one_has_no_near_one_edge() -> None:
    rng = np.random.default_rng(0)
    p = np.r_[rng.uniform(0.01, 0.2, 700), np.ones(300)]
    y = (rng.random(1000) < p).astype(float)
    ms = build_masterscale(y, p, n_grades=3)
    assert ms.edges[-1] == 1.0
    assert np.all(ms.edges[1:-1] < 0.2)
    assert ms.table(y, p).n.sum() == 1000


def test_build_masterscale_edges_are_observed_unclipped_values() -> None:
    p = np.r_[np.zeros(200), np.full(300, 0.3), np.full(500, 0.6)]
    y = (np.arange(1000) % 7 == 0).astype(float)
    ms = build_masterscale(y, p, n_grades=3)
    assert ms.edges.tolist() == [0.0, 0.3, 0.6, 1.0]
    assert ms.table(y, p).n.tolist() == [200.0, 300.0, 500.0]


def test_negative_zero_bound_shares_fingerprint() -> None:
    a = Masterscale({"A": (-0.0, 0.5), "B": (0.5, 1.0)})
    b = Masterscale({"A": (0.0, 0.5), "B": (0.5, 1.0)})
    assert a == b
    assert hash(a) == hash(b)
    assert a.fingerprint() == b.fingerprint()
    assert a.to_json() == b.to_json()


def test_interpret_states_half_open_band_inversion() -> None:
    text = Masterscale.from_edges([0.1]).interpret().messages[-1]
    assert "closed intervals" not in text
    assert "half-open" in text


# ------------------------------------------------------------------- plots

plt = pytest.importorskip("matplotlib.pyplot")

from probcal.curves import corp_reliability, reliability_binned  # noqa: E402
from probcal.metrics import evaluate  # noqa: E402
from probcal.plots import (  # noqa: E402
    plot_attributes,
    plot_belt,
    plot_comparison,
    plot_corp,
    plot_reliability,
)


@pytest.fixture(scope="module")
def portfolio():
    return make_pd_portfolio(n=3000, random_state=0)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _box(ax) -> str:
    return ax.texts[0].get_text()


def test_faceted_rejects_pooled_metric_report(portfolio) -> None:
    seg = np.where(portfolio.scores < 0.05, "lo", "hi")
    rep = evaluate(portfolio.y, portfolio.scores, n_boot=0)
    before = plt.get_fignums()
    with pytest.raises(ValueError, match="GroupedMetricReport"):
        plot_reliability(None, y=portfolio.y, p=portfolio.scores, by=seg, stats=rep)
    assert plt.get_fignums() == before


def test_faceted_uses_grouped_report_per_panel(portfolio) -> None:
    seg = np.where(portfolio.scores < 0.05, "lo", "hi")
    grouped = evaluate(portfolio.y, portfolio.scores, n_boot=0, by=seg)
    fig = plot_reliability(None, y=portfolio.y, p=portfolio.scores, by=seg, stats=grouped)
    titles = [ax.get_title() for ax in fig.axes]
    assert titles == ["pooled", "hi", "lo"]
    reports = dict(zip(titles, (grouped.pooled, *grouped.reports), strict=True))
    for ax in fig.axes:
        rep = reports[ax.get_title()]
        slope = rep.values[rep.names.index("slope")]
        assert f"slope = {slope:.3f}" in _box(ax)
    assert len({_box(ax) for ax in fig.axes}) == 3


def test_faceted_grouped_report_must_match_groups(portfolio) -> None:
    seg = np.where(portfolio.scores < 0.05, "lo", "hi")
    other = np.where(portfolio.scores < 0.05, "a", "b")
    grouped = evaluate(portfolio.y, portfolio.scores, n_boot=0, by=other)
    with pytest.raises(ValueError, match="groups"):
        plot_reliability(None, y=portfolio.y, p=portfolio.scores, by=seg, stats=grouped)


def test_grouped_report_without_by_raises(portfolio) -> None:
    seg = np.where(portfolio.scores < 0.05, "lo", "hi")
    grouped = evaluate(portfolio.y, portfolio.scores, n_boot=0, by=seg)
    curve = reliability_binned(portfolio.y, portfolio.scores)
    with pytest.raises(ValueError, match="by="):
        plot_reliability(curve, y=portfolio.y, p=portfolio.scores, stats=grouped)


def test_faceted_stats_true_is_per_group(portfolio) -> None:
    seg = np.where(portfolio.scores < 0.05, "lo", "hi")
    fig = plot_reliability(None, y=portfolio.y, p=portfolio.scores, by=seg, stats=True)
    assert len({_box(ax) for ax in fig.axes}) == 3


def test_faceted_rejects_labels_that_display_alike(portfolio) -> None:
    by = np.empty(len(portfolio.y), dtype=object)
    by[:] = [1 if i % 2 else "1" for i in range(len(portfolio.y))]
    with pytest.raises(ValueError, match="both display as"):
        evaluate(portfolio.y, portfolio.scores, n_boot=0, by=by)
    before = plt.get_fignums()
    with pytest.raises(ValueError, match="both display as"):
        plot_reliability(None, y=portfolio.y, p=portfolio.scores, by=by)
    assert plt.get_fignums() == before


def test_faceted_groups_by_raw_value(portfolio) -> None:
    by = np.arange(len(portfolio.y)) % 2
    fig = plot_reliability(None, y=portfolio.y, p=portfolio.scores, by=by, annotate=False)
    assert [ax.get_title() for ax in fig.axes] == ["pooled", "0", "1"]


def test_late_error_closes_the_created_figure(portfolio) -> None:
    curve = reliability_binned(portfolio.y, portfolio.scores)
    before = plt.get_fignums()
    with pytest.raises(ValueError):
        plot_reliability(curve, y=portfolio.y * 2, p=portfolio.scores, stats=True)
    assert plt.get_fignums() == before


def test_late_error_keeps_a_caller_supplied_figure(portfolio) -> None:
    curve = reliability_binned(portfolio.y, portfolio.scores)
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        plot_reliability(curve, y=portfolio.y * 2, p=portfolio.scores, stats=True, ax=ax)
    assert plt.fignum_exists(fig.number)


def test_scale_is_validated_everywhere(portfolio) -> None:
    y, p = portfolio.y, portfolio.scores
    curve = reliability_binned(y, p)
    calls = [
        lambda: plot_reliability(curve, scale="logits"),
        lambda: plot_reliability(None, y=y, p=p, by=p < 0.05, scale="Logit"),
        lambda: plot_belt(calibration_belt(y, p), scale="log"),
        lambda: plot_comparison(curve, curve, scale="prob"),
        lambda: plot_corp(corp_reliability(y, p, n_resamples=20), scale="logits"),
        lambda: plot_attributes(y, p, scale="logits"),
    ]
    before = plt.get_fignums()
    for call in calls:
        with pytest.raises(ValueError, match="scale must be"):
            call()
    assert plt.get_fignums() == before


def test_valid_scales_still_draw(portfolio) -> None:
    curve = reliability_binned(portfolio.y, portfolio.scores)
    for scale in ("probability", "logit"):
        ax = plot_reliability(curve, scale=scale)
        assert ax.get_xlabel().startswith("predicted probability")


def test_report_explains_merged_jeffreys_bands(tmp_path) -> None:
    from probcal.report import validation_report

    rng = np.random.default_rng(0)
    p = np.r_[np.full(400, 0.01), np.full(400, 0.05), np.full(400, 0.2)]
    grades = np.repeat(np.array(["A", "B", "C"]), 400)
    # Grade A defaults far above its PD, grade B never: B's own bound falls below A's.
    y = np.r_[(np.arange(400) < 40).astype(float), np.zeros(400), (rng.random(400) < 0.2) * 1.0]
    with pytest.warns(UserWarning, match="merged"):
        validation_report(y, p, grades=grades, n_boot=10, format="markdown", path=tmp_path / "r.md")
    text = (tmp_path / "r.md").read_text()
    assert "A+B" in text
    assert "Merged bands (A+B)" in text
