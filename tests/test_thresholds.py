"""Threshold translation: raw intervals partition the raw axis like Masterscale.assign."""

import numpy as np
import pytest

from probcal import (
    BetaCalibrator,
    HistogramBinningCalibrator,
    IsotonicCalibrator,
    Masterscale,
    PlattCalibrator,
    calibrated_bands_to_raw,
    calibrated_interval_to_raw,
    make_pd_portfolio,
)

PORT = make_pd_portfolio(n=6000, random_state=11)


def _raw_counts(raw: dict, s: np.ndarray, top: str) -> dict:
    out = {}
    for g, (lo, hi) in raw.items():
        inside = (s >= lo) & ((s <= hi) if g == top else (s < hi))
        out[g] = int(np.sum(inside))
    return out


def _scale_through_plateaus(cal, s: np.ndarray) -> Masterscale:
    """Edges placed exactly on fitted calibrated levels (the overlap case)."""
    levels = np.unique(cal.predict_proba(s))
    picks = levels[np.unique(np.linspace(1, len(levels) - 2, 4).astype(int))]
    return Masterscale.from_edges(picks)


@pytest.mark.parametrize(
    "cal",
    [IsotonicCalibrator(), HistogramBinningCalibrator(n_bins=5)],
    ids=["isotonic", "histogram"],
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_raw_interval_counts_equal_assign_counts(cal, seed) -> None:
    # OFF-1: closed raw intervals double-counted the plateau sitting on a
    # shared calibrated edge; half-open raw intervals must partition.
    d = make_pd_portfolio(n=4000, random_state=seed)
    cal.fit(d.scores, d.y)
    assert cal.is_monotone_  # n_bins=5 keeps the histogram rates monotone here
    for ms in (_scale_through_plateaus(cal, d.scores), Masterscale.from_edges([0.01, 0.02, 0.04])):
        raw = calibrated_bands_to_raw(cal, ms)
        labels = ms.assign(cal.predict_proba(d.scores))
        expected = {g: int(np.sum(labels == g)) for g in ms.names}
        assert _raw_counts(raw, d.scores, ms.names[-1]) == expected
        assert sum(expected.values()) == len(d.scores)


def test_adjacent_raw_bands_share_their_edge() -> None:
    cal = IsotonicCalibrator().fit(PORT.scores, PORT.y)
    ms = _scale_through_plateaus(cal, PORT.scores)
    raw = list(calibrated_bands_to_raw(cal, ms).values())
    for (_, hi0), (lo1, _) in zip(raw, raw[1:], strict=False):
        assert hi0 == lo1


def test_affine_calibrator_matches_closed_inversion_and_logit_space() -> None:
    cal = PlattCalibrator().fit(PORT.scores, PORT.y)
    ms = Masterscale.from_edges([0.02, 0.05])
    raw = calibrated_bands_to_raw(cal, ms)
    for g, (lo, hi) in ms.bands.items():
        closed = calibrated_interval_to_raw(cal, lo, hi)
        np.testing.assert_allclose(raw[g], closed, rtol=1e-12)
    raw_z = calibrated_bands_to_raw(cal, ms, space="logit")
    assert raw_z[ms.names[0]][0] == -np.inf and raw_z[ms.names[-1]][1] == np.inf


def test_plain_dict_bands_and_buffer() -> None:
    cal = BetaCalibrator().fit(PORT.scores, PORT.y)
    bands = {"A": (0.0, 0.02), "B": (0.02, 1.0)}
    raw = calibrated_bands_to_raw(cal, bands, buffer_logit=0.1)
    assert raw["A"][1] < raw["B"][0]  # the buffer opens a gap between bands
