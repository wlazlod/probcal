"""Protocol conformance across every inverse-capable object.

The contract a recourse engine (treecf's ``_SupportsIntervalInverse``)
relies on: keyword ``interval_inverse(lo, hi, space="logit",
buffer_logit=b)``; ``space="logit"`` bounds refer to the logit of the
object's *input* probability; ``lo=0``/``hi=1`` map to the full raw range
(∓inf on the logit scale); a buffered-empty interval raises
``UnattainableTargetError``; non-monotone fits say so and refuse inversion.
"""

import numpy as np
import pytest

from probcal import (
    BBQCalibrator,
    BetaCalibrator,
    CalibratedModel,
    CalibratorSelector,
    CenteredIsotonicCalibrator,
    Chain,
    HistogramBinningCalibrator,
    IsotonicCalibrator,
    LogitOffset,
    PlattCalibrator,
    ScalingBinningCalibrator,
    SegmentedCalibrator,
    SplineCalibrator,
    TemperatureCalibrator,
    UnattainableTargetError,
    VennAbersCalibrator,
    make_pd_portfolio,
)
from probcal._math import expit

_D = make_pd_portfolio(n=2500, event_rate=0.1, random_state=23)


class _StubModel:
    def predict_proba(self, X):
        s = np.asarray(X)[:, 0]
        return np.column_stack([1.0 - s, s])


def _cases() -> dict[str, object]:
    cases: dict[str, object] = {}
    for cls in (
        PlattCalibrator,
        TemperatureCalibrator,
        BetaCalibrator,
        IsotonicCalibrator,
        CenteredIsotonicCalibrator,
        HistogramBinningCalibrator,
        ScalingBinningCalibrator,
        BBQCalibrator,
        SplineCalibrator,
        VennAbersCalibrator,
        CalibratorSelector,
    ):
        cases[cls.__name__] = cls().fit(_D.scores, _D.y)
    cases["LogitOffset"] = LogitOffset(delta=0.3).fit(_D.scores)
    seg_labels = np.array(["a", "b", "c"])[np.arange(len(_D.scores)) % 3]
    cases["SegmentedCalibrator"] = SegmentedCalibrator().fit(_D.scores, _D.y, segments=seg_labels)
    beta = BetaCalibrator().fit(_D.scores, _D.y)
    off = LogitOffset(delta=0.2).fit(beta.predict_proba(_D.scores))
    cases["Chain"] = Chain([beta, off])
    wrapped = CalibratedModel(_StubModel(), BetaCalibrator(), flow="prefit").fit(
        _D.scores.reshape(-1, 1), _D.y
    )
    cases["CalibratedModel"] = wrapped
    return cases


_CASES = _cases()


def _skip_if_not_monotone(obj):
    if not getattr(obj, "is_monotone_", True):
        with pytest.raises((NotImplementedError, UnattainableTargetError)):
            obj.interval_inverse(0.0, 0.5, space="logit", buffer_logit=0.0)
        pytest.skip("non-monotone fit: refusal verified, inversion not applicable")


@pytest.mark.parametrize("name", sorted(_CASES))
def test_interval_inverse_keyword_protocol(name: str) -> None:
    obj = _CASES[name]
    _skip_if_not_monotone(obj)
    lo_z, hi_z = obj.interval_inverse(0.0, 0.5, space="logit", buffer_logit=0.1)
    assert np.isneginf(lo_z)  # lo = 0 -> the full lower raw range
    assert np.isfinite(hi_z) or np.isposinf(hi_z)
    lo2, hi2 = obj.interval_inverse(0.05, 1.0, space="logit", buffer_logit=0.0)
    assert np.isposinf(hi2)  # hi = 1 -> the full upper raw range
    # space="logit" bounds are the logit of the object's input probability.
    lo_p, hi_p = obj.interval_inverse(0.05, 0.3, space="probability")
    lo_z, hi_z = obj.interval_inverse(0.05, 0.3, space="logit")
    np.testing.assert_allclose(expit(np.array([lo_z, hi_z])), [lo_p, hi_p], atol=1e-9)
    # A buffer that empties the interval is refused, never clamped.
    with pytest.raises(UnattainableTargetError):
        obj.interval_inverse(0.09, 0.1, space="logit", buffer_logit=3.0)


# ---------------------------------------------------------------- plateau contract


@pytest.mark.parametrize(
    "make",
    [
        lambda: IsotonicCalibrator().fit(_D.scores, _D.y),
        lambda: HistogramBinningCalibrator(n_bins=6).fit(_D.scores, _D.y),
    ],
    ids=["isotonic", "histogram"],
)
def test_plateau_generalized_inverse_contract(make) -> None:
    # The contract treecf relies on (concepts/inverse-maps.md), for STEP
    # calibrators: for a target value equal to a plateau level, the left
    # inverse is the left edge of the first region attaining it and the
    # right inverse is the boundary after the last region within it —
    # closed intervals, so the whole plateau qualifies; targets beyond the
    # outer levels are one-sided (0/1 raw, ∓inf in logit space) or refused
    # when the interval misses the output range entirely.
    cal = make()
    assert cal.is_monotone_
    if isinstance(cal, HistogramBinningCalibrator):
        levels = np.asarray(cal.bin_rate_)
    else:
        levels = np.asarray(cal.block_mean_)
    inner = levels[(levels > levels.min()) & (levels < levels.max())]
    t = float(inner[len(inner) // 2])
    lo_s, hi_s = cal.interval_inverse(t, t)
    assert lo_s < hi_s  # the whole plateau, not a point
    p = cal.predict_proba(np.array([lo_s, 0.5 * (lo_s + hi_s)]))
    np.testing.assert_allclose(p, [t, t], atol=1e-12)  # closed on the left, level held
    # One-sided beyond the outer levels: full raw range on that side.
    lo_z, hi_z = cal.interval_inverse(0.0, float(levels.max()), space="logit")
    assert np.isneginf(lo_z) and np.isposinf(hi_z)
    # Entirely above the output range: refused, never clamped.
    above = 0.5 * (float(levels.max()) + 1.0)
    with pytest.raises(UnattainableTargetError):
        cal.interval_inverse(above, 1.0)


def test_centered_isotonic_is_pointwise_between_plateaus() -> None:
    # CIR interpolates linearly between block centers: unless adjacent block
    # means tie, a block level is attained at a single point, so the
    # degenerate-interval preimage is a point — the contract a recourse
    # engine gets from CIR differs from the step calibrators and is pinned
    # here (concepts/inverse-maps.md).
    cal = CenteredIsotonicCalibrator().fit(_D.scores, _D.y)
    levels = np.asarray(cal.block_mean_)
    inner = levels[(levels > levels.min()) & (levels < levels.max())]
    t = float(inner[len(inner) // 2])
    lo_s, hi_s = cal.interval_inverse(t, t)
    assert hi_s - lo_s <= 1e-6  # a point, not a plateau
    np.testing.assert_allclose(cal.predict_proba(np.array([0.5 * (lo_s + hi_s)])), [t], atol=1e-9)
