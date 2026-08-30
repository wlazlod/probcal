"""The sklearn duck hooks must not drag sklearn into a probcal import.

Runs regardless of whether sklearn is installed: the point is what a fit /
predict / is-fitted round trip pulls into ``sys.modules``, in a fresh
interpreter (this session's own imports would taint the check).
"""

import subprocess
import sys

CODE = (
    "import sys, numpy as np, probcal\n"
    "from probcal import BetaCalibrator\n"
    "s = np.linspace(0.05, 0.95, 50)\n"
    "y = (s > 0.5).astype(float)\n"
    "c = BetaCalibrator().fit(s, y)\n"
    "c.predict_proba(s)\n"
    "c.__sklearn_is_fitted__()\n"
    "assert 'sklearn' not in sys.modules and 'matplotlib' not in sys.modules\n"
)


def test_fit_predict_is_fitted_stays_numpy_only() -> None:
    result = subprocess.run(
        [sys.executable, "-c", CODE], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
