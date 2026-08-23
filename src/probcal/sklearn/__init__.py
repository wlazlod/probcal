"""scikit-learn adapter: probcal calibrators as sklearn estimators (spec W6).

Requires the ``probcal[sklearn]`` extra; ``import probcal`` itself stays
numpy-only — this subpackage is imported explicitly by its users.
"""

try:
    import sklearn  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
    raise ImportError(
        "probcal.sklearn requires scikit-learn >= 1.4; install the extra: "
        "pip install 'probcal[sklearn]'"
    ) from exc

from ._calibrator import SklearnCalibrator
from ._classifier import CalibratedClassifier

__all__ = ["CalibratedClassifier", "SklearnCalibrator"]
