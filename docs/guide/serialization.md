# Persisting calibrators

How-to; what is stored, why JSON and never pickle, and the compatibility
promise live in the *Serialization* concepts chapter.

```python
# s_cal, y_cal: held-out calibration scores and outcomes
import json

from probcal import BetaCalibrator

cal = BetaCalibrator().fit(s_cal, y_cal)

text = cal.to_json()                            # human-readable, versioned string
cal.to_json("calibrator.json")                  # or straight to a file
loaded = BetaCalibrator.from_json("calibrator.json")
# ...or without knowing the class:
from probcal import BaseCalibrator
loaded = BaseCalibrator.from_dict(json.loads(text))   # registry dispatch

cal.fingerprint()                # sha-256 provenance id (identical fits match)
cal.fit_meta_["data_fingerprint"]  # sha-256 of the sorted training triple
```

Objects with external parts keep them as references, never blobs:
`CalibratedModel.from_dict(d, model=...)` reattaches the base model;
`CalibratedScorecard.from_dict(d, scorecard=...)` verifies the scorecard
table's fingerprint before attaching. The monitor serializes its whole
past (`CalibrationMonitor.from_json` resumes bit-for-bit).

Every 0.x release reads schema 1 — enforced by committed golden files in
CI, not promised on trust.
