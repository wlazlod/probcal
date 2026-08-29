"""SegmentedCalibrator: empirical-Bayes shrunken per-segment logit offsets.

Theory and the DerSimonian-Laird method-of-moments derivation:
``docs/concepts/segmented.md``.

References
----------
DerSimonian & Laird (1986) — random-effects meta-analysis method-of-moments
heterogeneity estimator, reused here across segments instead of studies.
"""

import warnings

import numpy as np

from ._math import expit, logit
from ._registry import SERIALIZABLE, register
from ._results import Interpretation
from ._validation import validate_scores
from .base import BaseCalibrator
from .chain import Chain
from .offset import LogitOffset, estimate_offset
from .parametric import BetaCalibrator


def _coerce_segments(raw: object, n: int) -> np.ndarray:
    """1-D string array of segment labels, length-checked against ``n``."""
    arr = np.asarray(raw)
    if arr.ndim != 1:
        raise ValueError(f"segments must be a 1-D array, got shape {arr.shape}")
    if arr.shape[0] != n:
        raise ValueError(f"segments must have length {n}, got {arr.shape[0]}")
    return arr.astype(str)


@register
class SegmentedCalibrator(BaseCalibrator):
    """Per-segment logit offsets on top of a shared base map, empirical-Bayes shrunk.

    Fits one shared ``base`` calibrator on all data, then an offset-only
    logistic MLE (:func:`probcal.offset.estimate_offset`) of each segment's
    residual log-odds shift against the base map's predictions. Segments
    with few observations have a noisy, high-variance ``delta_hat``; rather
    than use it directly (no pooling — overfits small segments) or discard
    it (complete pooling — ignores real heterogeneity), each segment's
    estimate is shrunk toward the across-segment mean by the classic
    empirical-Bayes/random-effects factor ``tau2 / (tau2 + se**2)``, where
    ``tau2`` is the between-segment heterogeneity variance estimated by the
    DerSimonian-Laird (1986) method of moments. A small, noisy segment
    (large ``se``) shrinks toward 0 (the base map); a large, precise segment
    (small ``se``) keeps most of its own estimate.

    Segments with only one outcome class have no offset MLE
    (``estimate_offset`` raises); they are recorded as ``delta_hat=0.0``,
    ``se=inf`` — fully shrunk, since an infinite-variance estimate carries
    no weight in the DerSimonian-Laird pooling and ``tau2 / (tau2 + inf)
    = 0``.

    ``fit`` and ``predict_proba`` add a keyword-only ``segments`` argument
    on top of the base signature (``segments=None`` degrades to a single
    segment ``"__all__"`` at fit time, and to the plain base map — no
    segment-specific offset — at predict time), so the zero-argument
    protocol calls (``SegmentedCalibrator().fit(s, y)``,
    ``cal.predict_proba(s)``) still work. Because :class:`~probcal.chain.Chain`
    has no ``segments=`` slot, ``Chain([seg, ...])`` always predicts through
    ``seg``'s global map (``segments=None``, ``delta=0``) — the per-segment
    shift is not baked into a ``Chain``; use ``SegmentedCalibrator`` directly
    (with ``segments=``) when the per-segment offset must apply.

    Segment labels are compared as strings (``_coerce_segments`` calls
    ``.astype(str)``): fit-time labels ``0``, ``1`` (int) become ``"0"``,
    ``"1"``, but predict-time labels ``0.0``, ``1.0`` (float) become
    ``"0.0"``, ``"1.0"`` — a mismatch that never raises (every label
    looks "unseen") and, under ``unseen="global"``, silently falls back
    to the base map for every row. Pass ``segments`` with the *same*
    representation (e.g. cast to ``str`` yourself) at fit and predict
    time. When every row of a ``predict_proba``/inverse call is unseen
    and ``unseen="global"``, a ``UserWarning`` is raised naming the
    fitted ``segments_`` — a partial overlap (some rows match, some are
    genuinely new segments) stays silent.

    Parameters
    ----------
    base : BaseCalibrator or None
        Unfitted calibrator cloned and fitted on the pooled data at
        :meth:`fit` time (``type(base)(**base.get_params())``, the
        ``wrapper.py`` clone pattern). ``None`` (default) uses
        :class:`~probcal.parametric.BetaCalibrator`.
    unseen : {"global", "raise"}, keyword-only
        Policy for a segment label at predict/inverse time that was not
        seen at fit time. ``"global"`` (default) applies ``delta=0`` (the
        base map); ``"raise"`` raises ``ValueError``.

    Attributes
    ----------
    base_ : BaseCalibrator
        The fitted clone of ``base`` (or a fitted ``BetaCalibrator()``).
    segments_ : tuple of str
        Segment labels seen at fit time, sorted.
    n_, events_ : numpy.ndarray
        Per-segment observation count and weighted event count, aligned
        with ``segments_``.
    delta_hat_, se_ : numpy.ndarray
        Per-segment offset MLE and its Fisher standard error (``se=inf``
        for single-class segments).
    tau2_ : float
        Between-segment heterogeneity variance (DerSimonian-Laird
        method-of-moments estimate); 0.0 when fewer than two segments have
        a finite ``se`` (complete pooling).
    shrink_ : numpy.ndarray
        Per-segment shrinkage factor ``tau2 / (tau2 + se**2)`` in ``[0, 1)``.
    delta_tilde_ : numpy.ndarray
        Per-segment shrunk offset, ``delta_hat * shrink``: the offset
        actually applied at predict time.
    is_monotone_ : bool
        ``base_.is_monotone_``.

    Examples
    --------
    >>> import numpy as np
    >>> from probcal import SegmentedCalibrator, make_pd_portfolio
    >>> d = make_pd_portfolio(n=900, random_state=0)
    >>> segments = np.array(["a", "b", "c"])[np.arange(900) % 3]
    >>> cal = SegmentedCalibrator().fit(d.scores, d.y, segments=segments)
    >>> cal.segments_
    ('a', 'b', 'c')
    >>> p_global = cal.predict_proba(d.scores)  # segments=None: the base map
    >>> p_seg = cal.predict_proba(d.scores, segments=segments)
    >>> p_global.shape == p_seg.shape == d.scores.shape
    True
    """

    _STATE_ATTRS = (
        "base_",
        "segments_",
        "delta_hat_",
        "se_",
        "delta_tilde_",
        "n_",
        "events_",
        "shrink_",
        "tau2_",
    )

    def __init__(self, base: BaseCalibrator | None = None, *, unseen: str = "global") -> None:
        self.base = base
        self.unseen = unseen

    # ------------------------------------------------------------- fitting

    def fit(
        self,
        s: object,
        y: object,
        sample_weight: object = None,
        *,
        segments: object = None,
    ) -> "SegmentedCalibrator":
        """Fit the shared base map, then per-segment shrunk offsets.

        Parameters
        ----------
        s : array_like
            Raw scores/probabilities in ``[0, 1]``.
        y : array_like
            Binary outcomes in ``{0, 1}``; both classes must be present
            overall (individual segments may be single-class — see class
            docstring).
        sample_weight : array_like or None
            Positive observation weights.
        segments : array_like or None, keyword-only
            Segment label per observation, same length as ``s``. ``None``
            (default) fits a single segment ``"__all__"``.

        Returns
        -------
        SegmentedCalibrator
            The fitted calibrator.
        """
        self._segments_arg = segments
        try:
            return super().fit(s, y, sample_weight)  # type: ignore[return-value]
        finally:
            self.__dict__.pop("_segments_arg", None)

    def _fit(self, s: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        if self.unseen not in ("global", "raise"):
            raise ValueError(f"unseen must be 'global' or 'raise', got {self.unseen!r}")
        raw_segments = self.__dict__.get("_segments_arg")
        if raw_segments is None:
            seg_arr = np.full(s.shape[0], "__all__")
        else:
            seg_arr = _coerce_segments(raw_segments, s.shape[0])

        base_obj = self.base if self.base is not None else BetaCalibrator()
        self.base_ = type(base_obj)(**base_obj.get_params()).fit(s, y, sample_weight=w)
        self.is_monotone_ = self.base_.is_monotone_
        p0 = self.base_.predict_proba(s)

        labels = tuple(sorted(set(seg_arr.tolist())))
        delta_hat = np.empty(len(labels))
        se = np.empty(len(labels))
        n_arr = np.empty(len(labels), dtype=np.int64)
        events_arr = np.empty(len(labels))
        for i, g in enumerate(labels):
            m = seg_arr == g
            y_g, p_g, w_g = y[m], p0[m], w[m]
            try:
                est = estimate_offset(y_g, p_g, sample_weight=w_g)
                delta_hat[i], se[i], n_arr[i], events_arr[i] = est.delta, est.se, est.n, est.events
            except ValueError:
                delta_hat[i] = 0.0
                se[i] = np.inf
                n_arr[i] = int(m.sum())
                events_arr[i] = float(np.sum(w_g * y_g))

        finite = np.isfinite(se)
        g_finite = int(finite.sum())
        if g_finite < 2:
            tau2 = 0.0
        else:
            se_f, delta_f = se[finite], delta_hat[finite]
            wts = 1.0 / se_f**2
            delta_bar = float(np.average(delta_f, weights=wts))
            q_stat = float(np.sum(wts * (delta_f - delta_bar) ** 2))
            sw, sw2 = float(np.sum(wts)), float(np.sum(wts**2))
            denom = sw - sw2 / sw
            tau2 = max(0.0, (q_stat - (g_finite - 1)) / denom) if denom > 0.0 else 0.0

        shrink = tau2 / (tau2 + se**2)

        self.segments_ = labels
        self.delta_hat_ = delta_hat
        self.se_ = se
        self.delta_tilde_ = delta_hat * shrink
        self.shrink_ = shrink
        self.n_ = n_arr
        self.events_ = events_arr
        self.tau2_ = float(tau2)

    # ------------------------------------------------------------- prediction

    def _lookup_deltas(self, labels: np.ndarray, *, stacklevel: int = 3) -> np.ndarray:
        delta_map = dict(zip(self.segments_, self.delta_tilde_.tolist(), strict=True))
        out = np.zeros(len(labels), dtype=np.float64)
        unseen_found: set[str] = set()
        n_seen = 0
        for i, g in enumerate(labels):
            if g in delta_map:
                out[i] = delta_map[g]
                n_seen += 1
            elif self.unseen == "raise":
                unseen_found.add(str(g))
        if unseen_found:
            raise ValueError(
                f"unseen segment(s) {sorted(unseen_found)!r} not in fitted segments_ "
                f"{self.segments_}; unseen='raise'"
            )
        if self.unseen == "global" and len(labels) > 0 and n_seen == 0:
            warnings.warn(
                f"SegmentedCalibrator: none of the {len(labels)} segment labels passed at "
                f"predict time were seen at fit time ({self.segments_}); the global map "
                "(delta=0) is applied to every row — check the label representation",
                UserWarning,
                # 3 frames out of here is predict_proba's caller; the inverse
                # paths reach this through _segment_chain_or_base and pass 4.
                stacklevel=stacklevel,
            )
        return out

    def predict_proba(self, s: object, *, segments: object = None) -> np.ndarray:
        """Calibrated probabilities, with an optional per-observation segment offset.

        Parameters
        ----------
        s : array_like
            Raw scores/probabilities in ``[0, 1]``.
        segments : array_like or None, keyword-only
            Segment label per observation, same length as ``s``. ``None``
            (default) returns the plain base map (``delta=0``); a label not
            in :attr:`segments_` is handled per ``unseen``.

        Returns
        -------
        numpy.ndarray of shape (n,)
            Calibrated probabilities.
        """
        self._check_fitted()
        s_arr = validate_scores(s)
        p0 = self.base_.predict_proba(s_arr)
        if segments is None:
            return p0
        deltas = self._lookup_deltas(_coerce_segments(segments, s_arr.shape[0]))
        return expit(logit(p0) + deltas)

    def _predict(self, s: np.ndarray) -> np.ndarray:
        return self.base_.predict_proba(s)

    # ------------------------------------------------------------- protocol

    @property
    def affine_logit_coeffs_(self) -> tuple[float, float] | None:
        """``(a, b + delta_tilde)`` only for a single fitted segment; else ``None``.

        With more than one segment the map is segment-dependent (there is
        no single affine map on the logit scale that fits every segment),
        so :meth:`point_inverse` (which relies on this property) is
        unavailable then — use :meth:`interval_inverse` with ``segment=``.
        """
        self._check_fitted()
        if len(self.segments_) != 1:
            return None
        coeffs = self.base_.affine_logit_coeffs_
        if coeffs is None:
            return None
        a, b = coeffs
        return (a, b + float(self.delta_tilde_[0]))

    def _segment_chain_or_base(self, segment: object) -> BaseCalibrator | Chain:
        if segment is None:
            return self.base_
        delta = float(self._lookup_deltas(_coerce_segments([segment], 1), stacklevel=4)[0])
        if delta == 0.0:
            return self.base_
        return Chain([self.base_, LogitOffset(delta=delta).fit(np.array([0.5]))])

    def interval_inverse(
        self,
        lo: float,
        hi: float,
        *,
        space: str = "probability",
        buffer_logit: float = 0.0,
        segment: object = None,
    ) -> tuple[float, float]:
        """Preimage of a calibrated interval, optionally for one fitted segment.

        ``segment=None`` (default) uses the global map (``delta=0``),
        matching :meth:`predict_proba`'s ``segments=None`` convention;
        otherwise the preimage is through ``base_`` composed with that
        segment's ``delta_tilde`` (``Chain([base_, LogitOffset(delta=...)])``).

        Parameters
        ----------
        lo, hi : float
            Calibrated-probability bounds.
        space : {"probability", "logit"}, keyword-only
            Scale of the returned raw bounds.
        buffer_logit : float, keyword-only
            Logit-space shrinkage applied before inverting.
        segment : str or None, keyword-only
            Segment label to invert through; ``None`` for the global map.

        Returns
        -------
        tuple of float
            ``(raw_lo, raw_hi)`` on the requested scale.
        """
        self._check_fitted()
        target = self._segment_chain_or_base(segment)
        return target.interval_inverse(lo, hi, space=space, buffer_logit=buffer_logit)

    def point_inverse(
        self, p: object, *, space: str = "probability", segment: object = None
    ) -> np.ndarray:
        """Exact preimage of calibrated probabilities, optionally for one segment.

        Same ``segment=`` convention as :meth:`interval_inverse`: inverts
        through ``base_`` (composed with the segment's ``delta_tilde`` when
        ``segment`` is given), so this works for any number of fitted
        segments as long as ``base_`` itself has an exact point inverse
        (``base_.affine_logit_coeffs_`` is not ``None``) — unlike
        :attr:`affine_logit_coeffs_` on ``self``, which is only defined for
        a single fitted segment.

        Parameters
        ----------
        p : array_like
            Calibrated probabilities strictly inside ``(0, 1)``.
        space : {"probability", "logit"}, keyword-only
            Scale of the returned raw values.
        segment : str or None, keyword-only
            Segment label to invert through; ``None`` for the global map.

        Returns
        -------
        numpy.ndarray
            Raw scores (or logits) whose calibrated probability equals ``p``.
        """
        self._check_fitted()
        target = self._segment_chain_or_base(segment)
        return target.point_inverse(p, space=space)

    def interpret(self) -> Interpretation:
        """Per-segment shrinkage table plus the fitted heterogeneity variance."""
        self._check_fitted()
        param_names = ("tau2", *(f"delta.{g}" for g in self.segments_))
        param_values = (self.tau2_, *(float(d) for d in self.delta_tilde_))
        messages = [
            f"tau2 = {self.tau2_:.4f}: between-segment heterogeneity variance "
            "(DerSimonian-Laird method of moments on the per-segment offset MLEs); "
            "tau2 = 0 means complete pooling (every delta_tilde = 0)",
        ]
        for g, n, ev, dh, se, dt, sh in zip(
            self.segments_,
            self.n_,
            self.events_,
            self.delta_hat_,
            self.se_,
            self.delta_tilde_,
            self.shrink_,
            strict=True,
        ):
            se_str = "inf" if not np.isfinite(se) else f"{se:.4f}"
            messages.append(
                f"segment {g!r}: n={int(n)}, events={ev:.1f}, delta_hat={dh:+.4f}, "
                f"se={se_str}, delta_tilde={dt:+.4f}, shrink={sh:.3f}"
            )
        if self.unseen == "global":
            messages.append(
                f"unseen segments at predict/inverse time use delta=0 (unseen={self.unseen!r})"
            )
        else:
            messages.append(
                f"unseen segments at predict/inverse time raise ValueError (unseen={self.unseen!r})"
            )
        return Interpretation(
            method="SegmentedCalibrator",
            param_names=param_names,
            param_values=param_values,
            messages=tuple(messages),
        )

    # ------------------------------------------------------------- serialization

    def _params_for_dict(self) -> dict[str, object]:
        base_spec = None
        if self.base is not None:
            base_spec = {"class": type(self.base).__name__, "params": self.base.get_params()}
        return {"base": base_spec, "unseen": self.unseen}

    @classmethod
    def _params_from_dict(cls, params: dict[str, object]) -> dict[str, object]:
        base_spec = params.get("base")
        if base_spec is None:
            base_obj = None
        else:
            base_cls = SERIALIZABLE[base_spec["class"]]  # type: ignore[index]
            base_obj = base_cls(**base_spec["params"])  # type: ignore[index]
        return {"base": base_obj, "unseen": params.get("unseen", "global")}

    def _set_state(self, state: dict[str, object]) -> None:
        super()._set_state(state)
        self.segments_ = tuple(self.segments_)
