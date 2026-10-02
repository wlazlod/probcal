"""Suite-wide test configuration.

Matplotlib is forced onto the non-interactive Agg backend before anything can
import it (the environment variable is read at matplotlib import time, so no
test needs ``matplotlib.use("Agg")``), and every figure a test leaves open is
closed afterwards so pyplot's global figure registry cannot grow across the run.
"""

import os
import sys

import pytest

os.environ["MPLBACKEND"] = "Agg"
if "matplotlib" in sys.modules:  # imported early by a plugin: switch explicitly
    sys.modules["matplotlib"].use("Agg")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    # Only touch pyplot if the test (or the code under test) already imported it.
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is not None:
        plt.close("all")
