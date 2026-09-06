# Comparison: probcal vs sklearn, netcal, betacal

Reproducible via `docs/scripts/comparison.py` under `probcal[bench]`
(version pins in the line below the protocol; `--readme` prints the
README-sized block). Protocol: six public OpenML datasets spanning event
rates ~1.5%–30%; a seeded 50/25/25 train/calibration/test split; one base
model per dataset trained on the train split; every calibrator fits on the
calibration split's scores and is evaluated on the test split. Metrics:
log loss, ECE-sweep, and ICI with `probcal.metrics.evaluate` bootstrap
percentile CIs (n_boot=200), the number of rating grades passing the
Jeffreys backtest at the 5% level, and wall fit time.

Two base models. The five general datasets use
`HistGradientBoostingClassifier`, whose scores are nearly all distinct.
The credit-card dataset uses a scorecard: quantile-binned one-hot features,
logistic regression, and the logit rounded to integer points at PDO=20
(twenty points double the odds), so its scores carry the ties a deployed
scorecard has. Each dataset section opens with a line reporting the
distinct-score count, the largest and median tie block, and the base
model's test AUC.

Grades come from a fixed PD-band masterscale applied to each method's own
calibrated PD, which is how a masterscale is used in practice, so grade
sizes differ by method and a grade a method never reaches is absent from
its count. The credit-card dataset uses eight bands (A: PD < 3%, B: < 6%,
C: < 10%, D: < 15%, E: < 25%, F: < 40%, G: < 60%, H: the rest); the other
five use six (0.5%, 1%, 2%, 5%, 15%). A second table per dataset shows the
number of distinct output levels, the grade sizes, how many grades hold
fewer than 30 obligors, and how many hold zero defaults. The Jeffreys test
has no minimum-count guard: a four-obligor grade with zero defaults passes
trivially, which is why the size columns sit next to the pass count.

Methods: probcal `PlattCalibrator`, `BetaCalibrator` (abm),
`IsotonicCalibrator`, `SplineCalibrator`, `VennAbersCalibrator` (IVAP),
`CalibratorSelector` (default menu); the two sklearn maps inside
`CalibratedClassifierCV` (sigmoid = logistic on logits, and isotonic);
netcal `BBQ`, `ENIR`, `BetaCalibration`; and the reference `betacal`
package. All methods see identical calibration scores, so the comparison
isolates the calibration map. NaN scores never reach a calibrator: probcal
rejects them with a named `ValueError` rather than imputing, so no row
exists for them.

**Read the CIs, not the point ranks.** Most differences between reasonable
methods on a given dataset sit inside each other's bootstrap intervals;
what separates families is behavior at rare event rates (plateau variance
for the step methods) and what you get besides the map: interpretation,
inverses, serialization, monitoring. Where probcal loses, the table says
so.

pins: scikit-learn 1.9.0, netcal 1.4.0, betacal 1.1.0, pandas 3.0.3, n_boot=200

### default-of-credit-card-clients (22.1% event rate, n=30,000)

base: scorecard, test AUC 0.750; scores: 161 distinct on n_cal=7,500 (2.1%); largest tie block 164, median block 29; n_test=7,500

| method | log loss | ECE-sweep | ICI | grade pass | fit s |
|---|---|---|---|---|---|
| probcal Platt | 0.4527 [0.4428, 0.4617] | 0.0155 [0.0052, 0.0181] | 0.0056 [0.0041, 0.0122] | 7/8 | 0.00 |
| probcal beta (abm) | 0.4527 [0.4426, 0.4617] | 0.0161 [0.0071, 0.0196] | 0.0074 [0.0041, 0.0146] | 6/8 | 0.01 |
| probcal isotonic | 0.4524 [0.4418, 0.4616] | 0.0127 [0.0067, 0.0199] | 0.0086 [0.0036, 0.0166] | 6/8 | 0.00 |
| probcal spline | 0.4516 [0.4414, 0.4604] | 0.0121 [0.0053, 0.0184] | 0.0057 [0.0031, 0.0133] | 8/8 | 0.33 |
| probcal IVAP | 0.4522 [0.4414, 0.4613] | 0.0095 [0.0058, 0.0179] | 0.0078 [0.0037, 0.0155] | 6/7 | 0.05 |
| probcal selector | 0.4527 [0.4429, 0.4616] | 0.0151 [0.0056, 0.0177] | 0.0053 [0.0042, 0.0116] | 7/8 | 0.44 |
| sklearn sigmoid | 0.4527 [0.4428, 0.4617] | 0.0155 [0.0052, 0.0181] | 0.0056 [0.0041, 0.0122] | 7/8 | 0.00 |
| sklearn isotonic | 0.4524 [0.4418, 0.4616] | 0.0127 [0.0067, 0.0199] | 0.0086 [0.0036, 0.0166] | 6/8 | 0.00 |
| netcal BBQ | 0.4545 [0.4446, 0.4632] | 0.0176 [0.0087, 0.0274] | 0.0089 [0.0044, 0.0169] | 5/6 | 0.26 |
| netcal ENIR | 0.4536 [0.4427, 0.4631] | 0.0163 [0.0104, 0.0250] | 0.0154 [0.0091, 0.0226] | 4/8 | 4.34 |
| netcal beta | 0.4527 [0.4426, 0.4617] | 0.0161 [0.0071, 0.0196] | 0.0074 [0.0041, 0.0146] | 6/8 | 0.03 |
| betacal (abm) | 0.4527 [0.4426, 0.4617] | 0.0161 [0.0071, 0.0196] | 0.0074 [0.0041, 0.0146] | 6/8 | 0.01 |

| method | output levels | grade sizes (A/B/C/D/E/F/G/H) | n<30 | zero-default | pass |
|---|---|---|---|---|---|
| probcal Platt | 162 | 25/291/1386/1962/1885/682/741/528 | 1 | 0 | 7/8 |
| probcal beta (abm) | 162 | 46/396/1407/1815/1791/734/822/489 | 0 | 0 | 6/8 |
| probcal isotonic | 29 | 12/533/1157/1644/2519/382/685/568 | 1 | 1 | 6/8 |
| probcal spline | 162 | 4/187/1511/2104/1743/622/655/674 | 1 | 1 | 8/8 |
| probcal IVAP | 45 | –/545/1157/1644/2466/435/725/528 | 0 | 0 | 6/7 |
| probcal selector | 162 | 25/291/1254/2094/1885/698/725/528 | 1 | 0 | 7/8 |
| sklearn sigmoid | 162 | 25/291/1386/1962/1885/682/741/528 | 1 | 0 | 7/8 |
| sklearn isotonic | 30 | 12/533/1157/1644/2519/382/685/568 | 1 | 1 | 6/8 |
| netcal BBQ | 47 | –/–/2891/773/2201/444/557/634 | 0 | 0 | 5/6 |
| netcal ENIR | 26 | 12/591/1523/2508/1231/382/685/568 | 1 | 1 | 4/8 |
| netcal beta | 162 | 46/396/1407/1815/1791/734/822/489 | 0 | 0 | 6/8 |
| betacal (abm) | 162 | 46/396/1407/1815/1791/734/822/489 | 0 | 0 | 6/8 |

### Satellite (1.5% event rate, n=5,100)

base: hgb, test AUC 0.987; scores: 1,201 distinct on n_cal=1,275 (94.2%); largest tie block 18, median block 1; n_test=1,275

| method | log loss | ECE-sweep | ICI | grade pass | fit s |
|---|---|---|---|---|---|
| probcal Platt | 0.0251 [0.0126, 0.0375] | 0.0061 [0.0033, 0.0102] | 0.0058 [0.0047, 0.0097] | 6/6 | 0.00 |
| probcal beta (abm) | 0.0253 [0.0119, 0.0385] | 0.0064 [0.0032, 0.0105] | 0.0062 [0.0041, 0.0100] | 6/6 | 0.00 |
| probcal isotonic | 0.0879 [0.0181, 0.1593] | 0.0071 [0.0032, 0.0111] | 0.0074 [0.0040, 0.0111] | 2/3 | 0.01 |
| probcal spline | 0.0255 [0.0126, 0.0385] | 0.0062 [0.0029, 0.0107] | 0.0063 [0.0043, 0.0103] | 6/6 | 0.14 |
| probcal IVAP | 0.0287 [0.0193, 0.0379] | 0.0080 [0.0051, 0.0134] | 0.0087 [0.0075, 0.0122] | 6/6 | 0.01 |
| probcal selector | 0.0241 [0.0119, 0.0359] | 0.0046 [0.0028, 0.0083] | 0.0051 [0.0042, 0.0091] | 6/6 | 0.20 |
| sklearn sigmoid | 0.0259 [0.0119, 0.0398] | 0.0065 [0.0032, 0.0106] | 0.0064 [0.0042, 0.0103] | 6/6 | 0.00 |
| sklearn isotonic | 0.0880 [0.0182, 0.1594] | 0.0072 [0.0025, 0.0104] | 0.0075 [0.0041, 0.0113] | 2/3 | 0.00 |
| netcal BBQ | 0.0324 [0.0171, 0.0488] | 0.0023 [0.0005, 0.0081] | 0.0054 [0.0055, 0.0113] | 2/2 | 0.05 |
| netcal ENIR | 0.0876 [0.0177, 0.1589] | 0.0051 [0.0021, 0.0096] | 0.0071 [0.0037, 0.0110] | 2/3 | 0.00 |
| netcal beta | 0.0253 [0.0119, 0.0385] | 0.0064 [0.0032, 0.0105] | 0.0062 [0.0041, 0.0100] | 6/6 | 0.03 |
| betacal (abm) | 0.0253 [0.0119, 0.0386] | 0.0064 [0.0032, 0.0105] | 0.0062 [0.0041, 0.0100] | 6/6 | 0.01 |

| method | output levels | grade sizes (A/B/C/D/E/F) | n<30 | zero-default | pass |
|---|---|---|---|---|---|
| probcal Platt | 1192 | 1068/97/47/26/10/27 | 3 | 2 | 6/6 |
| probcal beta (abm) | 1192 | 1120/65/28/23/9/30 | 3 | 2 | 6/6 |
| probcal isotonic | 6 | 1199/–/–/–/51/25 | 1 | 1 | 2/3 |
| probcal spline | 1192 | 1128/46/28/24/17/32 | 3 | 2 | 6/6 |
| probcal IVAP | 472 | 987/119/48/33/47/41 | 0 | 2 | 6/6 |
| probcal selector | 1192 | 1118/74/31/20/7/25 | 3 | 2 | 6/6 |
| sklearn sigmoid | 1192 | 1115/69/29/24/9/29 | 4 | 2 | 6/6 |
| sklearn isotonic | 12 | 1199/–/–/–/51/25 | 1 | 1 | 2/3 |
| netcal BBQ | 8 | –/1252/–/–/–/23 | 1 | 0 | 2/2 |
| netcal ENIR | 7 | 1199/–/–/–/51/25 | 1 | 1 | 2/3 |
| netcal beta | 1192 | 1120/65/28/23/9/30 | 3 | 2 | 6/6 |
| betacal (abm) | 1192 | 1120/65/28/23/9/30 | 3 | 2 | 6/6 |

### mammography (2.3% event rate, n=11,183)

base: hgb, test AUC 0.933; scores: 1,903 distinct on n_cal=2,796 (68.1%); largest tie block 831, median block 1; n_test=2,796

| method | log loss | ECE-sweep | ICI | grade pass | fit s |
|---|---|---|---|---|---|
| probcal Platt | 0.0613 [0.0467, 0.0728] | 0.0048 [0.0024, 0.0076] | 0.0058 [0.0041, 0.0097] | 2/6 | 0.01 |
| probcal beta (abm) | 0.0616 [0.0467, 0.0737] | 0.0046 [0.0021, 0.0075] | 0.0057 [0.0041, 0.0094] | 3/6 | 0.00 |
| probcal isotonic | 0.1009 [0.0611, 0.1406] | 0.0062 [0.0042, 0.0095] | 0.0063 [0.0046, 0.0094] | 2/4 | 0.02 |
| probcal spline | 0.0599 [0.0458, 0.0716] | 0.0046 [0.0021, 0.0075] | 0.0051 [0.0038, 0.0091] | 3/6 | 0.20 |
| probcal IVAP | 0.0556 [0.0437, 0.0658] | 0.0030 [0.0010, 0.0069] | 0.0039 [0.0031, 0.0073] | 3/4 | 0.01 |
| probcal selector | 0.0617 [0.0466, 0.0739] | 0.0045 [0.0020, 0.0075] | 0.0060 [0.0042, 0.0097] | 3/6 | 0.51 |
| sklearn sigmoid | 0.0617 [0.0467, 0.0737] | 0.0048 [0.0024, 0.0077] | 0.0061 [0.0042, 0.0098] | 2/6 | 0.00 |
| sklearn isotonic | 0.1006 [0.0607, 0.1401] | 0.0061 [0.0040, 0.0094] | 0.0061 [0.0043, 0.0092] | 2/4 | 0.00 |
| netcal BBQ | 0.0705 [0.0553, 0.0841] | 0.0055 [0.0032, 0.0090] | 0.0043 [0.0067, 0.4896] | 1/3 | 0.08 |
| netcal ENIR | 0.0999 [0.0599, 0.1401] | 0.0053 [0.0030, 0.0090] | 0.0063 [0.0044, 0.0094] | 2/4 | 0.01 |
| netcal beta | 0.0616 [0.0467, 0.0736] | 0.0046 [0.0021, 0.0075] | 0.0057 [0.0041, 0.0094] | 3/6 | 0.03 |
| betacal (abm) | 0.0616 [0.0467, 0.0737] | 0.0046 [0.0021, 0.0075] | 0.0057 [0.0041, 0.0094] | 3/6 | 0.01 |

| method | output levels | grade sizes (A/B/C/D/E/F) | n<30 | zero-default | pass |
|---|---|---|---|---|---|
| probcal Platt | 1908 | 1663/107/882/42/30/72 | 0 | 0 | 2/6 |
| probcal beta (abm) | 1908 | 1714/891/53/31/30/77 | 0 | 0 | 3/6 |
| probcal isotonic | 12 | 2632/–/–/41/27/96 | 1 | 0 | 2/4 |
| probcal spline | 1908 | 1616/1016/31/25/25/83 | 2 | 0 | 3/6 |
| probcal IVAP | 693 | 2602/–/–/24/47/123 | 1 | 0 | 3/4 |
| probcal selector | 1908 | 1693/86/879/36/27/75 | 1 | 0 | 3/6 |
| sklearn sigmoid | 1908 | 1691/88/881/36/27/73 | 1 | 0 | 2/6 |
| sklearn isotonic | 22 | 2632/–/–/41/27/96 | 1 | 0 | 2/4 |
| netcal BBQ | 25 | –/2726/–/3/–/67 | 1 | 0 | 1/3 |
| netcal ENIR | 14 | 2633/–/–/41/27/95 | 1 | 0 | 2/4 |
| netcal beta | 1908 | 1714/891/53/31/30/77 | 0 | 0 | 3/6 |
| betacal (abm) | 1908 | 1715/890/53/31/30/77 | 0 | 0 | 3/6 |

### bank-marketing (11.7% event rate, n=45,211)

base: hgb, test AUC 0.932; scores: 11,288 distinct on n_cal=11,303 (99.9%); largest tie block 2, median block 1; n_test=11,303

| method | log loss | ECE-sweep | ICI | grade pass | fit s |
|---|---|---|---|---|---|
| probcal Platt | 0.2006 [0.1935, 0.2086] | 0.0097 [0.0073, 0.0147] | 0.0106 [0.0089, 0.0147] | 6/6 | 0.01 |
| probcal beta (abm) | 0.1993 [0.1919, 0.2065] | 0.0096 [0.0077, 0.0139] | 0.0092 [0.0070, 0.0131] | 6/6 | 0.01 |
| probcal isotonic | 0.2094 [0.1976, 0.2221] | 0.0116 [0.0072, 0.0149] | 0.0087 [0.0068, 0.0124] | 6/6 | 0.05 |
| probcal spline | 0.1992 [0.1923, 0.2064] | 0.0101 [0.0076, 0.0131] | 0.0090 [0.0068, 0.0123] | 6/6 | 1.26 |
| probcal IVAP | 0.1994 [0.1928, 0.2069] | 0.0110 [0.0082, 0.0144] | 0.0101 [0.0077, 0.0133] | 6/6 | 0.07 |
| probcal selector | 0.2000 [0.1933, 0.2075] | 0.0042 [0.0028, 0.0121] | 0.0075 [0.0055, 0.0115] | 6/6 | 1.65 |
| sklearn sigmoid | 0.2006 [0.1935, 0.2086] | 0.0097 [0.0074, 0.0147] | 0.0106 [0.0089, 0.0147] | 6/6 | 0.03 |
| sklearn isotonic | 0.2094 [0.1976, 0.2221] | 0.0116 [0.0073, 0.0148] | 0.0087 [0.0068, 0.0125] | 6/6 | 0.00 |
| netcal BBQ | 0.2015 [0.1942, 0.2096] | 0.0100 [0.0092, 0.0148] | 0.0094 [0.0074, 0.0127] | 3/3 | 0.57 |
| netcal ENIR | 0.2216 [0.2067, 0.2379] | 0.0099 [0.0071, 0.0135] | 0.0093 [0.0073, 0.0136] | 5/6 | 5.01 |
| netcal beta | 0.1993 [0.1919, 0.2065] | 0.0096 [0.0077, 0.0139] | 0.0092 [0.0070, 0.0131] | 6/6 | 0.20 |
| betacal (abm) | 0.1993 [0.1919, 0.2065] | 0.0095 [0.0077, 0.0139] | 0.0092 [0.0070, 0.0131] | 6/6 | 0.02 |

| method | output levels | grade sizes (A/B/C/D/E/F) | n<30 | zero-default | pass |
|---|---|---|---|---|---|
| probcal Platt | 11289 | 3070/1575/1375/1338/1346/2599 | 0 | 0 | 6/6 |
| probcal beta (abm) | 11289 | 4412/1147/870/1049/1110/2715 | 0 | 0 | 6/6 |
| probcal isotonic | 41 | 5489/342/529/714/1443/2786 | 0 | 0 | 6/6 |
| probcal spline | 11289 | 4676/853/745/1006/1266/2757 | 0 | 0 | 6/6 |
| probcal IVAP | 1678 | 5271/354/690/701/1500/2787 | 0 | 1 | 6/6 |
| probcal selector | 11289 | 2832/1685/1461/1413/1390/2522 | 0 | 0 | 6/6 |
| sklearn sigmoid | 11289 | 3080/1575/1369/1337/1343/2599 | 0 | 0 | 6/6 |
| sklearn isotonic | 80 | 5489/342/529/714/1443/2786 | 0 | 0 | 6/6 |
| netcal BBQ | 198 | –/7121/–/–/1372/2810 | 0 | 0 | 3/3 |
| netcal ENIR | 36 | 5495/342/529/714/1443/2780 | 0 | 0 | 5/6 |
| netcal beta | 11289 | 4412/1147/870/1049/1110/2715 | 0 | 0 | 6/6 |
| betacal (abm) | 11289 | 4413/1148/868/1049/1110/2715 | 0 | 0 | 6/6 |

### adult (23.9% event rate, n=48,842)

base: hgb, test AUC 0.922; scores: 11,641 distinct on n_cal=12,210 (95.3%); largest tie block 12, median block 1; n_test=12,211

| method | log loss | ECE-sweep | ICI | grade pass | fit s |
|---|---|---|---|---|---|
| probcal Platt | 0.2875 [0.2803, 0.2952] | 0.0078 [0.0062, 0.0140] | 0.0045 [0.0021, 0.0083] | 6/6 | 0.01 |
| probcal beta (abm) | 0.2875 [0.2804, 0.2953] | 0.0081 [0.0066, 0.0141] | 0.0048 [0.0023, 0.0084] | 6/6 | 0.01 |
| probcal isotonic | 0.3016 [0.2872, 0.3156] | 0.0085 [0.0058, 0.0145] | 0.0047 [0.0023, 0.0089] | 4/6 | 0.06 |
| probcal spline | 0.2875 [0.2801, 0.2953] | 0.0064 [0.0049, 0.0138] | 0.0029 [0.0018, 0.0073] | 5/6 | 1.34 |
| probcal IVAP | 0.2877 [0.2805, 0.2957] | 0.0070 [0.0050, 0.0130] | 0.0032 [0.0017, 0.0077] | 5/6 | 0.07 |
| probcal selector | 0.2875 [0.2804, 0.2952] | 0.0071 [0.0053, 0.0135] | 0.0032 [0.0021, 0.0068] | 5/6 | 1.54 |
| sklearn sigmoid | 0.2875 [0.2803, 0.2953] | 0.0079 [0.0063, 0.0140] | 0.0046 [0.0021, 0.0084] | 6/6 | 0.08 |
| sklearn isotonic | 0.3015 [0.2871, 0.3156] | 0.0086 [0.0058, 0.0145] | 0.0048 [0.0023, 0.0090] | 4/6 | 0.00 |
| netcal BBQ | 0.2899 [0.2826, 0.2983] | 0.0095 [0.0069, 0.0143] | 0.0041 [0.0023, 0.0083] | 3/3 | 0.43 |
| netcal ENIR | 0.3328 [0.3110, 0.3549] | 0.0096 [0.0078, 0.0165] | 0.0080 [0.0052, 0.0113] | 3/6 | 6.17 |
| netcal beta | 0.2875 [0.2804, 0.2953] | 0.0081 [0.0066, 0.0141] | 0.0048 [0.0023, 0.0084] | 6/6 | 0.04 |
| betacal (abm) | 0.2875 [0.2804, 0.2953] | 0.0081 [0.0066, 0.0141] | 0.0047 [0.0023, 0.0084] | 6/6 | 0.01 |

| method | output levels | grade sizes (A/B/C/D/E/F) | n<30 | zero-default | pass |
|---|---|---|---|---|---|
| probcal Platt | 11645 | 2019/931/1192/1434/1573/5062 | 0 | 0 | 6/6 |
| probcal beta (abm) | 11645 | 2083/929/1183/1403/1551/5062 | 0 | 0 | 6/6 |
| probcal isotonic | 49 | 2543/801/1017/1051/1806/4993 | 0 | 0 | 4/6 |
| probcal spline | 11645 | 2582/712/886/1208/1634/5189 | 0 | 0 | 5/6 |
| probcal IVAP | 1691 | 2156/1091/789/1361/1816/4998 | 0 | 0 | 5/6 |
| probcal selector | 11645 | 1996/947/1214/1444/1605/5005 | 0 | 0 | 5/6 |
| sklearn sigmoid | 11645 | 2026/932/1199/1422/1572/5060 | 0 | 0 | 6/6 |
| sklearn isotonic | 96 | 2543/801/1017/1050/1807/4993 | 0 | 0 | 4/6 |
| netcal BBQ | 40 | –/–/5410/–/1646/5155 | 0 | 0 | 3/3 |
| netcal ENIR | 51 | 2563/801/1017/1050/1848/4932 | 0 | 0 | 3/6 |
| netcal beta | 11645 | 2083/929/1183/1403/1551/5062 | 0 | 0 | 6/6 |
| betacal (abm) | 11645 | 2083/928/1184/1405/1549/5062 | 0 | 0 | 6/6 |

### credit-g (30.0% event rate, n=1,000)

base: hgb, test AUC 0.739; scores: 250 distinct on n_cal=250 (100.0%); largest tie block 1, median block 1; n_test=250

| method | log loss | ECE-sweep | ICI | grade pass | fit s |
|---|---|---|---|---|---|
| probcal Platt | 0.5519 [0.5044, 0.6104] | 0.0352 [0.0267, 0.0928] | 0.0548 [0.0320, 0.0935] | 3/4 | 0.00 |
| probcal beta (abm) | 0.5638 [0.5143, 0.6241] | 0.0453 [0.0314, 0.1027] | 0.0661 [0.0401, 0.1061] | 3/5 | 0.00 |
| probcal isotonic | 0.8347 [0.5247, 1.2227] | 0.0717 [0.0259, 0.0947] | 0.0533 [0.0347, 0.0960] | 2/4 | 0.00 |
| probcal spline | 0.5573 [0.5089, 0.6158] | 0.0386 [0.0247, 0.0994] | 0.0610 [0.0339, 0.0991] | 2/4 | 0.05 |
| probcal IVAP | 0.5477 [0.5007, 0.6041] | 0.0698 [0.0200, 0.0833] | 0.0533 [0.0251, 0.0927] | 2/2 | 0.00 |
| probcal selector | 0.5524 [0.5026, 0.6152] | 0.0481 [0.0287, 0.0975] | 0.0520 [0.0286, 0.0941] | 2/4 | 0.07 |
| sklearn sigmoid | 0.5526 [0.5040, 0.6121] | 0.0375 [0.0281, 0.0936] | 0.0547 [0.0321, 0.0934] | 3/4 | 0.00 |
| sklearn isotonic | 0.8355 [0.5270, 1.2247] | 0.0717 [0.0276, 0.0945] | 0.0510 [0.0331, 0.0929] | 3/5 | 0.00 |
| netcal BBQ | 0.5423 [0.4958, 0.5877] | 0.0363 [0.0219, 0.0662] | 0.0324 [0.0180, 0.0666] | 1/1 | 0.02 |
| netcal ENIR | 0.8374 [0.5283, 1.2264] | 0.0620 [0.0288, 0.0888] | 0.0497 [0.0352, 0.0918] | 2/4 | 0.02 |
| netcal beta | 0.5638 [0.5143, 0.6241] | 0.0453 [0.0314, 0.1027] | 0.0661 [0.0401, 0.1061] | 3/5 | 0.02 |
| betacal (abm) | 0.5638 [0.5143, 0.6241] | 0.0454 [0.0314, 0.1028] | 0.0661 [0.0401, 0.1061] | 3/5 | 0.00 |

| method | output levels | grade sizes (A/B/C/D/E/F) | n<30 | zero-default | pass |
|---|---|---|---|---|---|
| probcal Platt | 250 | –/–/1/16/55/178 | 2 | 1 | 3/4 |
| probcal beta (abm) | 250 | –/1/5/25/49/170 | 3 | 1 | 3/5 |
| probcal isotonic | 12 | 26/–/–/19/73/132 | 2 | 0 | 2/4 |
| probcal spline | 250 | –/–/4/18/46/182 | 2 | 0 | 2/4 |
| probcal IVAP | 42 | –/–/–/–/45/205 | 0 | 0 | 2/2 |
| probcal selector | 250 | –/–/2/19/51/178 | 2 | 0 | 2/4 |
| sklearn sigmoid | 250 | –/–/1/19/54/176 | 2 | 1 | 3/4 |
| sklearn isotonic | 22 | 24/–/1/20/73/132 | 3 | 1 | 3/5 |
| netcal BBQ | 26 | –/–/–/–/–/250 | 0 | 0 | 1/1 |
| netcal ENIR | 16 | 26/–/–/19/73/132 | 2 | 0 | 2/4 |
| netcal beta | 250 | –/1/5/25/49/170 | 3 | 1 | 3/5 |
| betacal (abm) | 250 | –/1/5/25/49/170 | 3 | 1 | 3/5 |

## Reading the tables

- **The maps agree where they should.** probcal Platt equals sklearn
  sigmoid, probcal isotonic equals sklearn isotonic, and the beta rows
  agree across probcal, netcal, and `betacal` to the fourth decimal on
  every dataset. Same algorithm, same numbers: a useful correctness anchor
  for the harness, and the reason the README says the map is not where the
  libraries differ.
- **What the ties did (credit-card).** 161 distinct scorecard points on
  7,500 calibration obligors, largest block 164. Every method fit; no IRLS
  separation fallback fired. The step methods collapse the 161 inputs
  further: isotonic to about 30 output levels (probcal 29, sklearn 30; the two maps differ on one boundary row), ENIR to 26, BBQ to 47, IVAP to 45.
  `SplineCalibrator` warned that its fitted curve is not monotone (a
  refusal it makes in the open, and the reason it is not on the README's
  production path). Log loss is a dead heat across all twelve rows; the
  only method clearly behind is ENIR on ICI (0.0154 vs 0.0056 for Platt).
- **What the tiny grades did (credit-card).** Grade A (PD < 3%) holds 25
  obligors under Platt, 46 under beta, 12 under isotonic and ENIR, 4 under
  spline; IVAP's output never reaches it, and BBQ's never reaches A or B.
  The zero-default grade A under isotonic, ENIR, and spline passes the
  Jeffreys test trivially, so spline's 8/8 and isotonic's 6/8 are not
  comparable without the size table. Read pass counts together with the
  `n<30` and `zero-default` columns.
- **Rare events punish plateaus.** On Satellite (1.5%) and mammography
  (2.3%), the isotonic maps and `netcal ENIR` land at 1.6–3.5× the log
  loss of the parametric maps (sparse tail blocks generalize badly), and
  the effect explodes on tiny credit-g (isotonic/ENIR ≈ 0.84 vs ≈ 0.55).
  The same plateaus leave whole grades empty: on Satellite the step methods
  reach only three of six bands. probcal's IVAP keeps step-function
  flexibility without the blow-up and posts the best probcal log loss on
  both rare sets.
- **Where probcal loses.** On credit-g, `netcal BBQ` beats every probcal
  method on all three metrics (log loss 0.542 vs 0.548 for our best, ICI
  0.032 vs 0.052); Bayesian averaging over binnings is genuinely strong on
  small samples. Its 1/1 grade pass there is because its output range
  covers a single band. On Satellite, BBQ's ECE-sweep (0.0023) is the best
  in the table. If a single number on a small portfolio is all you need,
  BBQ is a fine choice (probcal ships one too); the CIs show most of these
  gaps are within resampling noise.
- **Fit time** is negligible for everything except spline and the selector
  (under 2s) and `netcal ENIR` (up to ~6s here; quadratic in unique scores,
  the same scaling probcal's ENIR warns about).
- **ICI CIs on piecewise-constant outputs** are unreliable: the LOESS
  smoother behind ICI degenerates in some bootstrap replicates when the
  calibrated output has few distinct levels, which is why `netcal BBQ`'s
  ICI interval on mammography does not contain its point estimate. A bound
  that is not finite is rendered as `CI undefined`; `evaluate` reports the
  failure rather than faking a value.
