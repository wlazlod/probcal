"""Calibration metrics and statistical tests (flat re-exports).

Implementation lives in the submodules (``scores``, ``binned``, ``smooth``,
``regression``, ``kernel``, ``grade``) and in private helpers
(``_evaluate``, ``_summary``, ``_binstats``, ``_grades``, ``_common``).
The sample-weight convention shared by every metric is documented in
:mod:`probcal.metrics._common`. Selection guidance — what may be optimized
and what is report-only — is the table in ``docs/concepts/metrics.md``.
"""

from .._results import GroupedMetricReport
from .._results import MetricReport as MetricReport
from ._evaluate import _METRIC_CATALOG as _METRIC_CATALOG
from ._evaluate import _point_metrics as _point_metrics
from ._evaluate import evaluate
from ._summary import ReliabilitySummary, reliability_summary
from .binned import (
    HosmerLemeshowResult,
    adaptive_ece,
    ece,
    ece_debiased,
    ece_sweep,
    hosmer_lemeshow,
)
from .grade import (
    BinomialGradeResult,
    HlEResult,
    JeffreysGradeResult,
    PlutoTascheResult,
    binomial_grade_test,
    hl_e_test,
    jeffreys_grade_test,
    jeffreys_upper_bands,
    pluto_tasche,
    pluto_tasche_from_arrays,
)
from .kernel import (
    SkceTestResult,
    skce,
    skce_test,
)
from .regression import (
    CalibrationTestResult,
    GuardrailReport,
    calibration_guardrails,
    calibration_intercept,
    calibration_slope,
    calibration_test,
)
from .scores import (
    LogLossDecomposition,
    MurphyCurve,
    MurphyDecomposition,
    brier_score,
    brier_skill_score,
    log_loss,
    logloss_calibration_refinement,
    murphy_curve,
    murphy_decomposition,
)
from .smooth import (
    EcceResult,
    SpiegelhalterResult,
    e50,
    e90,
    ecce,
    emax,
    ici,
    smooth_ece,
    spiegelhalter_z,
)

__all__ = [
    "BinomialGradeResult",
    "CalibrationTestResult",
    "EcceResult",
    "GroupedMetricReport",
    "GuardrailReport",
    "HlEResult",
    "HosmerLemeshowResult",
    "JeffreysGradeResult",
    "LogLossDecomposition",
    "MurphyCurve",
    "MurphyDecomposition",
    "PlutoTascheResult",
    "ReliabilitySummary",
    "SkceTestResult",
    "SpiegelhalterResult",
    "adaptive_ece",
    "binomial_grade_test",
    "brier_score",
    "brier_skill_score",
    "calibration_guardrails",
    "calibration_intercept",
    "calibration_slope",
    "calibration_test",
    "e50",
    "e90",
    "ecce",
    "ece",
    "ece_debiased",
    "ece_sweep",
    "emax",
    "evaluate",
    "hl_e_test",
    "hosmer_lemeshow",
    "ici",
    "jeffreys_grade_test",
    "jeffreys_upper_bands",
    "log_loss",
    "logloss_calibration_refinement",
    "murphy_curve",
    "murphy_decomposition",
    "pluto_tasche",
    "pluto_tasche_from_arrays",
    "reliability_summary",
    "skce",
    "skce_test",
    "smooth_ece",
    "spiegelhalter_z",
]
