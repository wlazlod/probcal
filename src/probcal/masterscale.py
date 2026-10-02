"""A masterscale as one value object: the grade ladder on the calibrated scale.

A masterscale is the fixed ladder of PD bands that defines rating grades.
probcal consumes it in two shapes -- a ``{name: (lo, hi)}`` dict for
translation and a per-observation label array for testing and monitoring --
and :class:`Masterscale` is the single place both shapes come from.

Boundary convention, stated once: :meth:`Masterscale.assign` is half-open,
``lo <= p < hi``, with the top band closed at its upper edge, so every ``p``
in ``[edges[0], edges[-1]]`` belongs to exactly one grade. Band inversion
(:func:`probcal.thresholds.calibrated_bands_to_raw`) follows the same rule on
the raw axis: band ``[lo, hi)`` maps to ``[inf{s: g(s) >= lo}, inf{s: g(s) >=
hi})``, top band closed, so the raw intervals partition the raw axis and
counting raw scores per interval reproduces :meth:`Masterscale.assign`
counts exactly — also for step calibrators (isotonic, histogram binning),
whose plateaus give a shared edge positive raw-score mass.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

import numpy as np

from ._registry import register
from ._results import Interpretation, _ResultBase
from ._serialize import JsonIO, canonical_json, check_payload, decode_value, encode_value, envelope
from ._validation import validate_binary_y, validate_scores, validate_weights


def _freeze(v: object) -> object:
    """Deep-immutable copy: dicts become read-only mappings, lists tuples."""
    if isinstance(v, Mapping):
        return MappingProxyType({str(k): _freeze(x) for k, x in v.items()})
    if isinstance(v, (list, tuple)):
        return tuple(_freeze(x) for x in v)
    if isinstance(v, np.ndarray):
        return tuple(_freeze(x) for x in v.tolist())
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    return v


def _thaw(v: object) -> object:
    """Inverse of :func:`_freeze`: a fresh, mutable, JSON-shaped copy."""
    if isinstance(v, Mapping):
        return {k: _thaw(x) for k, x in v.items()}
    if isinstance(v, tuple):
        return [_thaw(x) for x in v]
    return v


def _masterscale_from_state(
    bands: dict[str, tuple[float, float]], provenance: dict | None
) -> "Masterscale":
    """Pickle/deepcopy reconstructor (see :meth:`Masterscale.__reduce__`)."""
    return Masterscale(bands, provenance=provenance)


@dataclass(frozen=True, eq=False, repr=False)
class GradeTable(_ResultBase):
    """Per-grade counts against a masterscale: the standard grade table.

    Attributes
    ----------
    grades : tuple of str
        Every grade of the masterscale, best to worst (empty grades included).
    lo, hi : numpy.ndarray
        Band bounds per grade.
    n : numpy.ndarray
        Observation count per grade (weighted sum when ``sample_weight`` is given).
    events : numpy.ndarray
        Event count per grade (weighted).
    mean_pd : numpy.ndarray
        Mean assigned probability per grade; ``nan`` for an empty grade.
    observed_rate : numpy.ndarray
        ``events / n``; ``nan`` for an empty grade.
    """

    grades: tuple[str, ...]
    lo: np.ndarray
    hi: np.ndarray
    n: np.ndarray
    events: np.ndarray
    mean_pd: np.ndarray
    observed_rate: np.ndarray

    _TABLE: ClassVar[tuple[str, ...]] = (
        "grade",
        "lo",
        "hi",
        "n",
        "events",
        "mean_pd",
        "observed_rate",
    )

    def _rows(self) -> list[tuple[object, ...]]:
        return list(
            zip(
                self.grades,
                self.lo,
                self.hi,
                self.n,
                self.events,
                self.mean_pd,
                self.observed_rate,
                strict=True,
            )
        )


@register
class Masterscale(JsonIO):
    """Frozen grade ladder on the calibrated-probability scale.

    Parameters
    ----------
    bands : dict[str, tuple[float, float]]
        Grade name to ``(lo, hi)`` calibrated-probability bounds. Bands may be
        given in any order; they are stored sorted by ``lo``. Every bound must
        lie in ``[0, 1]``, each band needs ``lo < hi``, consecutive bands must
        share their edge exactly (gaps and overlaps both raise), and names must
        be unique.
    provenance : dict or None, keyword-only
        Optional record of how the scale was built (set by
        :func:`probcal.thresholds.build_masterscale`); ``None`` for a
        hand-built scale. Deep-frozen on construction (later changes to the
        dict passed in do not reach the scale), serialized with the object,
        and shown by :meth:`interpret`. Values must be JSON-encodable.

    Attributes
    ----------
    names : tuple of str
        Grade names, best (lowest PD) to worst.
    edges : numpy.ndarray
        The ``n_grades + 1`` band edges, ascending.
    n_grades : int
        Number of grades.

    Notes
    -----
    Assignment is half-open, ``lo <= p < hi``, with the top band closed at its
    upper edge. Values outside ``[edges[0], edges[-1]]`` raise; a masterscale
    is expected to cover ``[0, 1]`` (pass ``lo`` and ``hi`` to
    :meth:`from_edges` explicitly if yours does not, or accept the error).
    Band inversion through :func:`probcal.thresholds.calibrated_bands_to_raw`
    uses the same half-open rule on the raw axis, so raw-interval counts and
    :meth:`assign` counts agree exactly.

    Equality and hashing cover the bands *and* the provenance, so
    ``a == b`` exactly when ``a.fingerprint() == b.fingerprint()``; compare
    ``a.bands == b.bands`` for "same ladder, however built". Instances are
    immutable and support :func:`copy.deepcopy` and pickling.

    Examples
    --------
    >>> ms = Masterscale.from_edges([0.01, 0.05], names=["A", "B", "C"])
    >>> ms.assign([0.005, 0.01, 0.05, 1.0]).tolist()
    ['A', 'B', 'C', 'C']
    """

    __slots__ = ("_bands", "_edges", "_names", "_provenance")
    _bands: tuple[tuple[str, tuple[float, float]], ...]
    _edges: np.ndarray
    _names: tuple[str, ...]
    _provenance: "MappingProxyType[str, object] | None"

    def __init__(
        self, bands: dict[str, tuple[float, float]], *, provenance: dict | None = None
    ) -> None:
        if not bands:
            raise ValueError("a masterscale needs at least one band")
        items: list[tuple[float, float, str]] = []
        for name, bounds in bands.items():
            if len(bounds) != 2:
                raise ValueError(f"band {name!r}: expected (lo, hi), got {bounds!r}")
            lo, hi = float(bounds[0]), float(bounds[1])
            in_range = np.isfinite(lo) and np.isfinite(hi) and 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0
            if not in_range:
                raise ValueError(f"band {name!r}: bounds must lie in [0, 1], got ({lo}, {hi})")
            if not lo < hi:
                raise ValueError(f"band {name!r}: lo must be < hi, got ({lo}, {hi})")
            items.append((lo, hi, str(name)))
        items.sort(key=lambda t: t[0])
        names = tuple(t[2] for t in items)
        if len(set(names)) != len(names):
            raise ValueError(f"grade names must be unique, got {names}")
        for (_lo0, hi0, n0), (lo1, _hi1, n1) in zip(items, items[1:], strict=False):
            if hi0 < lo1:
                raise ValueError(f"gap between band {n0!r} (hi={hi0}) and band {n1!r} (lo={lo1})")
            if hi0 > lo1:
                raise ValueError(f"band {n1!r} (lo={lo1}) overlaps band {n0!r} (hi={hi0})")
        object.__setattr__(self, "_bands", tuple((n, (lo, hi)) for lo, hi, n in items))
        object.__setattr__(self, "_edges", np.array([t[0] for t in items] + [items[-1][1]]))
        object.__setattr__(self, "_names", names)
        frozen = None if provenance is None else _freeze(dict(provenance))
        object.__setattr__(self, "_provenance", frozen)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Masterscale is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Masterscale is immutable")

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (_masterscale_from_state, (self.bands, self.provenance))

    # ------------------------------------------------------------ constructors

    @classmethod
    def from_edges(
        cls,
        edges: object,
        names: object = None,
        lo: float = 0.0,
        hi: float = 1.0,
        *,
        provenance: dict | None = None,
    ) -> "Masterscale":
        """Build from interior edges: ``len(edges) + 1`` grades on ``[lo, hi]``.

        Parameters
        ----------
        edges : array_like
            Interior band edges (any order; sorted here).
        names : sequence of str or None
            One name per grade, best to worst; ``None`` gives ``"G1"``, ``"G2"``, ...
        lo, hi : float
            Outer bounds, ``0.0`` and ``1.0`` by default.
        provenance : dict or None, keyword-only
            Passed to the constructor (see :class:`Masterscale`).
        """
        inner = sorted(float(e) for e in np.asarray(edges, dtype=np.float64).reshape(-1))
        full = [float(lo), *inner, float(hi)]
        n = len(full) - 1
        if names is None:
            labels = [f"G{i + 1}" for i in range(n)]
        else:
            labels = [str(x) for x in list(names)]  # type: ignore[call-overload]
        if len(labels) != n:
            raise ValueError(f"{n} grades from {len(inner)} edges, but {len(labels)} names given")
        if len(set(labels)) != n:
            raise ValueError(f"grade names must be unique, got {tuple(labels)}")
        return cls({labels[i]: (full[i], full[i + 1]) for i in range(n)}, provenance=provenance)

    # ---------------------------------------------------------------- readers

    @property
    def bands(self) -> dict[str, tuple[float, float]]:
        """``{name: (lo, hi)}`` best to worst: the shape every band consumer accepts."""
        return dict(self._bands)

    @property
    def edges(self) -> np.ndarray:
        """The ``n_grades + 1`` band edges, ascending (a copy)."""
        return self._edges.copy()

    @property
    def names(self) -> tuple[str, ...]:
        """Grade names, best to worst."""
        return self._names

    @property
    def n_grades(self) -> int:
        """Number of grades."""
        return len(self._names)

    @property
    def provenance(self) -> dict | None:
        """How the scale was built (``None`` for a hand-built scale); a fresh copy."""
        return None if self._provenance is None else _thaw(self._provenance)  # type: ignore[return-value]

    def index(self, p: object) -> np.ndarray:
        """Zero-based grade index per observation, ``lo <= p < hi``, top band closed.

        ``p`` goes through :func:`probcal._validation.validate_scores`, so a
        two-column ``predict_proba`` matrix is accepted.

        Raises
        ------
        ValueError
            If any ``p`` lies outside ``[edges[0], edges[-1]]``.
        """
        p_arr = validate_scores(p, name="p")
        lo, hi = self._edges[0], self._edges[-1]
        bad = (p_arr < lo) | (p_arr > hi)
        if np.any(bad):
            raise ValueError(
                f"{int(bad.sum())} value(s) of p lie outside the masterscale range "
                f"[{lo}, {hi}] (first offender {p_arr[bad][0]!r}); a masterscale is expected "
                "to cover [0, 1]"
            )
        return np.searchsorted(self._edges[1:-1], p_arr, side="right").astype(np.int64)

    def assign(self, p: object) -> np.ndarray:
        """Grade name per observation (see :meth:`index` for the rule)."""
        return np.asarray(self._names, dtype=object)[self.index(p)].astype(str)

    # ----------------------------------------------------------------- table

    def table(self, y: object, p: object, sample_weight: object = None) -> GradeTable:
        """Per-grade counts of ``y`` against ``p`` assigned through this scale.

        Every grade is listed, including empty ones (``n == 0``, rates ``nan``);
        weights, when given, turn counts into weighted sums.
        """
        y_arr = validate_binary_y(y)
        p_arr = validate_scores(p, name="p")
        if len(y_arr) != len(p_arr):
            raise ValueError("y and p must have equal length")
        w_arr = validate_weights(sample_weight, len(p_arr))
        idx = self.index(p_arr)
        k = self.n_grades
        n = np.bincount(idx, weights=w_arr, minlength=k).astype(np.float64)
        events = np.bincount(idx, weights=w_arr * y_arr, minlength=k).astype(np.float64)
        wp = np.bincount(idx, weights=w_arr * p_arr, minlength=k).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_pd = np.where(n > 0, wp / n, np.nan)
            observed = np.where(n > 0, events / n, np.nan)
        return GradeTable(
            grades=self._names,
            lo=self._edges[:-1].copy(),
            hi=self._edges[1:].copy(),
            n=n,
            events=events,
            mean_pd=mean_pd,
            observed_rate=observed,
        )

    def interpret(self) -> Interpretation:
        """The band table in words, with the boundary convention and provenance.

        ``param_names``/``param_values`` list each grade's band as two
        entries, ``"<grade>.lo"`` and ``"<grade>.hi"`` (until 0.3 only the
        upper edges were listed, under the bare grade names).
        """
        last = self.n_grades - 1
        messages = [
            f"{n}: {lo:.6g} <= p < {hi:.6g}" + (" (top band, closed at hi)" if i == last else "")
            for i, (n, (lo, hi)) in enumerate(self._bands)
        ]
        messages.append(
            "assignment is half-open, lo <= p < hi, with the top band closed at its upper "
            "edge; band inversion keeps closed intervals"
        )
        prov = self.provenance
        if prov is not None:
            messages.append(
                "built by build_masterscale: " + ", ".join(f"{k}={v}" for k, v in prov.items())
            )
        return Interpretation(
            method="Masterscale",
            param_names=tuple(f"{n}.{side}" for n in self._names for side in ("lo", "hi")),
            param_values=tuple(float(b) for _, (lo, hi) in self._bands for b in (lo, hi)),
            messages=tuple(messages),
        )

    # --------------------------------------------------------- serialization

    def to_dict(self) -> dict[str, object]:
        """Versioned JSON-native snapshot (schema 1, the envelope every class uses)."""
        return envelope(
            self,
            params={"bands": {n: [lo, hi] for n, (lo, hi) in self._bands}},
            state={"provenance": encode_value(self.provenance)},
            fit_meta={},
        )

    @classmethod
    def from_dict(cls, d: dict) -> "Masterscale":
        """Rebuild from :meth:`to_dict` output.

        Raises
        ------
        ValueError
            If the schema version is unknown or the payload class differs.
        """
        check_payload(cls, d)
        bands = {str(n): (float(b[0]), float(b[1])) for n, b in d["params"]["bands"].items()}
        prov = decode_value(d.get("state", {}).get("provenance"), arrays=False)
        return cls(bands, provenance=prov)  # type: ignore[arg-type]

    # ---------------------------------------------------------------- dunder

    def _key(self) -> tuple[object, str]:
        return (self._bands, canonical_json(encode_value(self.provenance)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Masterscale):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        body = ", ".join(f"{n}: [{lo:g}, {hi:g})" for n, (lo, hi) in self._bands)
        return f"Masterscale({body}; top band closed)"
