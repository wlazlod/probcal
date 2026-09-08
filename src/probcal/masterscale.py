"""A masterscale as one value object: the grade ladder on the calibrated scale.

A masterscale is the fixed ladder of PD bands that defines rating grades.
probcal consumes it in two shapes -- a ``{name: (lo, hi)}`` dict for
translation and a per-observation label array for testing and monitoring --
and :class:`Masterscale` is the single place both shapes come from.

Boundary convention, stated once: :meth:`Masterscale.assign` is half-open,
``lo <= p < hi``, with the top band closed at its upper edge, so every ``p``
in ``[edges[0], edges[-1]]`` belongs to exactly one grade. Band inversion
(:func:`probcal.thresholds.calibrated_bands_to_raw`) keeps closed intervals,
since boundary points have measure zero on the raw scale; a validator
reconciling counts uses the assignment rule above.
"""

import json
import os
from dataclasses import dataclass

import numpy as np

from ._registry import register
from ._results import Interpretation
from ._serialize import SCHEMA_VERSION, check_schema, fingerprint_of_dict
from ._validation import validate_binary_y, validate_scores, validate_weights


@dataclass(frozen=True)
class GradeTable:
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

    def __str__(self) -> str:
        width = max(5, *(len(g) for g in self.grades))
        head = (
            f"{'grade':<{width}}  {'lo':>8}  {'hi':>8}  {'n':>10}  {'events':>10}  "
            f"{'mean_pd':>9}  {'observed_rate':>13}"
        )
        rows = [head, "-" * len(head)]
        for i, g in enumerate(self.grades):
            rows.append(
                f"{g:<{width}}  {self.lo[i]:>8.4f}  {self.hi[i]:>8.4f}  {self.n[i]:>10.1f}  "
                f"{self.events[i]:>10.1f}  {self.mean_pd[i]:>9.5f}  {self.observed_rate[i]:>13.5f}"
            )
        return "\n".join(rows)


@register
class Masterscale:
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
        hand-built scale. Serialized with the object and shown by
        :meth:`interpret`.

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
    keeps closed intervals; the two conventions differ only at a shared
    edge, which has measure zero on the raw scale.

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
    _provenance: dict | None

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
        object.__setattr__(
            self, "_provenance", dict(provenance) if provenance is not None else None
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Masterscale is immutable")

    # ------------------------------------------------------------ constructors

    @classmethod
    def from_edges(
        cls, edges: object, names: object = None, lo: float = 0.0, hi: float = 1.0
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
        return cls({labels[i]: (full[i], full[i + 1]) for i in range(n)})

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
        """How the scale was built (``None`` for a hand-built scale)."""
        return None if self._provenance is None else dict(self._provenance)

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
        """The band table in words, with the boundary convention and provenance."""
        last = self.n_grades - 1
        messages = [
            f"{n}: {lo:.6g} <= p < {hi:.6g}" + (" (top band, closed at hi)" if i == last else "")
            for i, (n, (lo, hi)) in enumerate(self._bands)
        ]
        messages.append(
            "assignment is half-open, lo <= p < hi, with the top band closed at its upper "
            "edge; band inversion keeps closed intervals"
        )
        if self._provenance is not None:
            messages.append(
                "built by build_masterscale: "
                + ", ".join(f"{k}={v}" for k, v in self._provenance.items())
            )
        return Interpretation(
            method="Masterscale",
            param_names=self._names,
            param_values=tuple(float(hi) for _, (_, hi) in self._bands),
            messages=tuple(messages),
        )

    # --------------------------------------------------------- serialization

    def to_dict(self) -> dict[str, object]:
        """Versioned JSON-native snapshot (schema 1, the envelope every class uses)."""
        from . import __version__

        return {
            "probcal_schema": SCHEMA_VERSION,
            "probcal_version": __version__,
            "class": type(self).__name__,
            "params": {"bands": {n: [lo, hi] for n, (lo, hi) in self._bands}},
            "state": {"provenance": self._provenance},
            "fit_meta": {},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Masterscale":
        """Rebuild from :meth:`to_dict` output.

        Raises
        ------
        ValueError
            If the schema version is unknown or the payload class differs.
        """
        check_schema(d)
        if d.get("class") != cls.__name__:
            raise ValueError(f"payload was written by {d.get('class')!r}, not {cls.__name__}")
        bands = {str(n): (float(b[0]), float(b[1])) for n, b in d["params"]["bands"].items()}
        return cls(bands, provenance=d.get("state", {}).get("provenance"))

    def to_json(
        self, path: "str | os.PathLike[str] | None" = None, *, indent: int = 2
    ) -> str | None:
        """Serialize to JSON text, or to ``path`` when given (returns None then)."""
        text = json.dumps(self.to_dict(), indent=indent)
        if path is None:
            return text
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return None

    @classmethod
    def from_json(cls, path_or_str: object) -> "Masterscale":
        """Load from a JSON string or a filesystem path."""
        text = str(path_or_str)
        if not text.lstrip().startswith("{"):
            with open(text, encoding="utf-8") as fh:
                text = fh.read()
        return cls.from_dict(json.loads(text))

    def fingerprint(self) -> str:
        """SHA-256 of the canonical serialized form, blind to the writing version."""
        return fingerprint_of_dict(self.to_dict())

    # ---------------------------------------------------------------- dunder

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Masterscale):
            return NotImplemented
        return self._bands == other._bands

    def __hash__(self) -> int:
        return hash(self._bands)

    def __repr__(self) -> str:
        body = ", ".join(f"{n}: [{lo:g}, {hi:g})" for n, (lo, hi) in self._bands)
        return f"Masterscale({body}; top band closed)"
