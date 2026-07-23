# Licensing

`probcal` is released under the MIT license (see `LICENSE`).

## Conceptual references and GPL code

Several R packages in the calibration literature are GPL-licensed: `givitiR` (calibration
belt), `rms`, and `CalibratR`. These are used **as conceptual references only**: probcal
reimplements every algorithm from the primary papers cited in the documentation and never
ports, translates, or adapts GPL source code.

## Test-only dependencies

`scipy`, `scikit-learn`, and `statsmodels` appear exclusively in the `dev` extra as numerical
references for the test suite. They are never imported by `src/probcal` (enforced by
`tests/test_package.py::test_no_forbidden_imports`).
