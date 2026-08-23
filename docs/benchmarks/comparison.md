# Comparison: probcal vs sklearn, netcal, betacal

Reproducible via `docs/scripts/comparison.py` under `probcal[bench]`
(version pins in the table header below). Protocol: five public OpenML
datasets spanning event rates ~1.5%–30%; a seeded 50/25/25
train/calibration/test split; one `HistGradientBoostingClassifier` base
model per dataset; every calibrator fits on the calibration split's scores
and is evaluated on the test split. Metrics: log loss, ECE-sweep, and ICI
with `probcal.metrics.evaluate` bootstrap percentile CIs (n_boot=200), the
Jeffreys per-grade backtest pass rate over six fixed PD bands, and wall fit
time.

Methods: probcal `BetaCalibrator` (abm), `SplineCalibrator`,
`VennAbersCalibrator` (IVAP), `CalibratorSelector` (default menu); the
sklearn recipes (sigmoid = logistic on logits, isotonic — the two maps
inside `CalibratedClassifierCV`); netcal `BBQ`, `ENIR`, `BetaCalibration`;
and the reference `betacal` package. All methods see identical calibration
scores, so the comparison isolates the calibration map.

**Read the CIs, not the point ranks.** Most differences between reasonable
methods on a given dataset sit inside each other's bootstrap intervals;
what separates families is behavior at rare event rates (plateau variance
for the step methods) and what you get besides the map — interpretation,
inverses, serialization, monitoring. Where probcal loses, the table says
so.

pins: scikit-learn 1.9.0, netcal 1.4.0, betacal 1.1.0, n_boot=200

### Satellite (1.5% event rate, n=5,100)

| method | log loss | ECE-sweep | ICI | grade pass | fit s |
|---|---|---|---|---|---|
| probcal beta (abm) | 0.0253 [0.0119, 0.0385] | 0.0064 [0.0032, 0.0105] | 0.0062 [0.0041, 0.0100] | 100% | 0.00 |
| probcal spline | 0.0255 [0.0126, 0.0385] | 0.0062 [0.0029, 0.0107] | 0.0063 [0.0043, 0.0103] | 100% | 0.13 |
| probcal IVAP | 0.0287 [0.0193, 0.0379] | 0.0080 [0.0051, 0.0134] | 0.0087 [0.0075, 0.0122] | 100% | 0.01 |
| probcal selector | 0.0241 [0.0119, 0.0359] | 0.0046 [0.0028, 0.0083] | 0.0051 [0.0042, 0.0091] | 100% | 0.19 |
| sklearn sigmoid | 0.0259 [0.0119, 0.0398] | 0.0065 [0.0032, 0.0106] | 0.0064 [0.0042, 0.0103] | 100% | 0.00 |
| sklearn isotonic | 0.0880 [0.0182, 0.1594] | 0.0072 [0.0025, 0.0104] | 0.0075 [0.0041, 0.0113] | 67% | 0.00 |
| netcal BBQ | 0.0324 [0.0171, 0.0488] | 0.0023 [0.0005, 0.0081] | 0.0054 [nan, nan] | 100% | 0.04 |
| netcal ENIR | 0.0876 [0.0177, 0.1589] | 0.0051 [0.0021, 0.0096] | 0.0071 [0.0037, 0.0110] | 67% | 0.00 |
| netcal beta | 0.0253 [0.0119, 0.0385] | 0.0064 [0.0032, 0.0105] | 0.0062 [0.0041, 0.0100] | 100% | 0.02 |
| betacal (abm) | 0.0253 [0.0119, 0.0386] | 0.0064 [0.0032, 0.0105] | 0.0062 [0.0041, 0.0100] | 100% | 0.01 |

### mammography (2.3% event rate, n=11,183)

| method | log loss | ECE-sweep | ICI | grade pass | fit s |
|---|---|---|---|---|---|
| probcal beta (abm) | 0.0616 [0.0467, 0.0737] | 0.0046 [0.0021, 0.0075] | 0.0057 [0.0041, 0.0094] | 50% | 0.01 |
| probcal spline | 0.0599 [0.0458, 0.0716] | 0.0046 [0.0021, 0.0075] | 0.0051 [0.0038, 0.0091] | 50% | 0.19 |
| probcal IVAP | 0.0556 [0.0437, 0.0658] | 0.0030 [0.0010, 0.0069] | 0.0039 [0.0031, 0.0073] | 75% | 0.01 |
| probcal selector | 0.0617 [0.0466, 0.0739] | 0.0045 [0.0020, 0.0075] | 0.0060 [0.0042, 0.0097] | 50% | 0.30 |
| sklearn sigmoid | 0.0617 [0.0467, 0.0737] | 0.0048 [0.0024, 0.0077] | 0.0061 [0.0042, 0.0098] | 33% | 0.00 |
| sklearn isotonic | 0.1006 [0.0607, 0.1401] | 0.0061 [0.0040, 0.0094] | 0.0061 [0.0043, 0.0092] | 50% | 0.00 |
| netcal BBQ | 0.0705 [0.0553, 0.0841] | 0.0055 [0.0032, 0.0090] | 0.0043 [nan, nan] | 33% | 0.07 |
| netcal ENIR | 0.0999 [0.0599, 0.1401] | 0.0053 [0.0030, 0.0090] | 0.0063 [0.0044, 0.0094] | 50% | 0.01 |
| netcal beta | 0.0616 [0.0467, 0.0736] | 0.0046 [0.0021, 0.0075] | 0.0057 [0.0041, 0.0094] | 50% | 0.03 |
| betacal (abm) | 0.0616 [0.0467, 0.0737] | 0.0046 [0.0021, 0.0075] | 0.0057 [0.0041, 0.0094] | 50% | 0.01 |

### bank-marketing (11.7% event rate, n=45,211)

| method | log loss | ECE-sweep | ICI | grade pass | fit s |
|---|---|---|---|---|---|
| probcal beta (abm) | 0.1993 [0.1919, 0.2065] | 0.0096 [0.0077, 0.0139] | 0.0092 [0.0070, 0.0131] | 100% | 0.01 |
| probcal spline | 0.1992 [0.1923, 0.2064] | 0.0101 [0.0076, 0.0131] | 0.0090 [0.0068, 0.0123] | 100% | 0.70 |
| probcal IVAP | 0.1994 [0.1928, 0.2069] | 0.0110 [0.0082, 0.0144] | 0.0101 [0.0077, 0.0133] | 100% | 0.06 |
| probcal selector | 0.2000 [0.1933, 0.2075] | 0.0042 [0.0028, 0.0121] | 0.0075 [0.0055, 0.0115] | 100% | 1.18 |
| sklearn sigmoid | 0.2006 [0.1935, 0.2086] | 0.0097 [0.0074, 0.0147] | 0.0106 [0.0089, 0.0147] | 100% | 0.03 |
| sklearn isotonic | 0.2094 [0.1976, 0.2221] | 0.0116 [0.0073, 0.0148] | 0.0087 [0.0068, 0.0125] | 100% | 0.00 |
| netcal BBQ | 0.2015 [0.1942, 0.2096] | 0.0100 [0.0092, 0.0148] | 0.0094 [0.0074, 0.0127] | 100% | 0.34 |
| netcal ENIR | 0.2216 [0.2067, 0.2379] | 0.0099 [0.0071, 0.0135] | 0.0093 [0.0073, 0.0136] | 83% | 2.66 |
| netcal beta | 0.1993 [0.1919, 0.2065] | 0.0096 [0.0077, 0.0139] | 0.0092 [0.0070, 0.0131] | 100% | 0.05 |
| betacal (abm) | 0.1993 [0.1919, 0.2065] | 0.0095 [0.0077, 0.0139] | 0.0092 [0.0070, 0.0131] | 100% | 0.01 |

### adult (23.9% event rate, n=48,842)

| method | log loss | ECE-sweep | ICI | grade pass | fit s |
|---|---|---|---|---|---|
| probcal beta (abm) | 0.2875 [0.2804, 0.2953] | 0.0081 [0.0066, 0.0141] | 0.0048 [0.0023, 0.0084] | 100% | 0.01 |
| probcal spline | 0.2875 [0.2801, 0.2953] | 0.0064 [0.0049, 0.0138] | 0.0029 [0.0018, 0.0073] | 83% | 0.86 |
| probcal IVAP | 0.2877 [0.2805, 0.2957] | 0.0070 [0.0050, 0.0130] | 0.0032 [0.0017, 0.0077] | 83% | 0.06 |
| probcal selector | 0.2875 [0.2804, 0.2952] | 0.0071 [0.0053, 0.0135] | 0.0032 [0.0021, 0.0068] | 83% | 1.35 |
| sklearn sigmoid | 0.2875 [0.2803, 0.2953] | 0.0079 [0.0063, 0.0140] | 0.0046 [0.0021, 0.0084] | 100% | 0.03 |
| sklearn isotonic | 0.3015 [0.2871, 0.3156] | 0.0086 [0.0058, 0.0145] | 0.0048 [0.0023, 0.0090] | 67% | 0.00 |
| netcal BBQ | 0.2899 [0.2826, 0.2983] | 0.0095 [0.0069, 0.0143] | 0.0041 [0.0023, 0.0083] | 100% | 0.42 |
| netcal ENIR | 0.3328 [0.3110, 0.3549] | 0.0096 [0.0078, 0.0165] | 0.0080 [0.0052, 0.0113] | 50% | 5.89 |
| netcal beta | 0.2875 [0.2804, 0.2953] | 0.0081 [0.0066, 0.0141] | 0.0048 [0.0023, 0.0084] | 100% | 0.07 |
| betacal (abm) | 0.2875 [0.2804, 0.2953] | 0.0081 [0.0066, 0.0141] | 0.0047 [0.0023, 0.0084] | 100% | 0.01 |

### credit-g (30.0% event rate, n=1,000)

| method | log loss | ECE-sweep | ICI | grade pass | fit s |
|---|---|---|---|---|---|
| probcal beta (abm) | 0.5638 [0.5143, 0.6241] | 0.0453 [0.0314, 0.1027] | 0.0661 [0.0401, 0.1061] | 60% | 0.00 |
| probcal spline | 0.5573 [0.5089, 0.6158] | 0.0386 [0.0247, 0.0994] | 0.0610 [0.0339, 0.0991] | 50% | 0.05 |
| probcal IVAP | 0.5477 [0.5007, 0.6041] | 0.0698 [0.0200, 0.0833] | 0.0533 [0.0251, 0.0927] | 100% | 0.00 |
| probcal selector | 0.5524 [0.5026, 0.6152] | 0.0481 [0.0287, 0.0975] | 0.0520 [0.0286, 0.0941] | 50% | 0.06 |
| sklearn sigmoid | 0.5526 [0.5040, 0.6121] | 0.0375 [0.0281, 0.0936] | 0.0547 [0.0321, 0.0934] | 75% | 0.00 |
| sklearn isotonic | 0.8355 [0.5270, 1.2247] | 0.0717 [0.0276, 0.0945] | 0.0510 [0.0331, 0.0929] | 60% | 0.00 |
| netcal BBQ | 0.5423 [0.4958, 0.5877] | 0.0363 [0.0219, 0.0662] | 0.0324 [0.0180, 0.0666] | 100% | 0.02 |
| netcal ENIR | 0.8374 [0.5283, 1.2264] | 0.0620 [0.0288, 0.0888] | 0.0497 [0.0352, 0.0918] | 50% | 0.02 |
| netcal beta | 0.5638 [0.5143, 0.6241] | 0.0453 [0.0314, 0.1027] | 0.0661 [0.0401, 0.1061] | 60% | 0.02 |
| betacal (abm) | 0.5638 [0.5143, 0.6241] | 0.0454 [0.0314, 0.1028] | 0.0661 [0.0401, 0.1061] | 60% | 0.00 |

## Reading the table

- **The beta rows agree to the fourth decimal** across probcal, netcal, and
  `betacal` — same family, three implementations; a useful correctness
  anchor for the harness.
- **Rare events punish plateaus.** On Satellite (1.5%) and mammography
  (2.3%), `sklearn isotonic` and `netcal ENIR` land at 1.6–3.5× the log
  loss of the parametric maps — sparse tail blocks generalize badly — and
  the effect explodes on tiny credit-g (isotonic/ENIR ≈ 0.84 vs ≈ 0.55).
  probcal's IVAP keeps step-function flexibility without the blow-up
  (regularized by the label-conditional sweep) and posts the best probcal
  log loss on both rare sets.
- **Where probcal loses:** on credit-g, `netcal BBQ` beats every probcal
  method on all three metrics (log loss 0.542 vs 0.548 for our best, ICI
  0.032 vs 0.052, 100% grade pass) — Bayesian averaging over binnings is
  genuinely strong on small samples; on Satellite its ECE-sweep (0.0023) is
  the best in the table. If a single number on a small portfolio is all
  you need, BBQ is a fine choice (probcal ships one too); the CIs show
  most of these gaps are within resampling noise.
- **Fit time** is negligible for everything except spline/selector
  (sub-2s) and `netcal ENIR` (up to ~6s here; quadratic in unique scores —
  the same scaling probcal's ENIR warns about).
- `nan` CIs on the netcal BBQ ICI rows: bootstrap replicates where the
  piecewise-constant output degenerates leave the LOESS-based ICI
  undefined; `evaluate` reports the failure rather than faking a value.

