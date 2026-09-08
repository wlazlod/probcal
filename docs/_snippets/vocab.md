??? note "Data these snippets assume"
    Blocks on this page use a fixed set of names for held-out data instead of
    re-deriving it each time. To run them, define the names first:

    ```python
    # docs: no-run — the definitions the snippets on this page assume
    import numpy as np
    from probcal import BetaCalibrator, Masterscale, make_pd_portfolio
    from probcal.monitor import CalibrationMonitor

    cal_set, new_set = make_pd_portfolio(n=3000, random_state=0), make_pd_portfolio(n=1000, random_state=1)
    s_cal, y_cal = cal_set.scores, cal_set.y          # held-out calibration scores and outcomes
    w_cal = np.ones_like(y_cal)                       # uniform sample weights
    s_new = new_set.scores                            # scores of new obligors
    ms = Masterscale.from_edges([0.01, 0.05], names=["G1", "G2", "G3"])   # the masterscale
    p_cal = BetaCalibrator().fit(s_cal, y_cal).predict_proba(s_cal)         # calibrated PDs
    grades = ms.assign(p_cal)                                              # rating labels
    segments = np.array(["seg-a", "seg-b", "seg-c"])[np.arange(len(s_cal)) % 3]  # segment labels
    model = ...   # any object with predict_proba(X); the docs use a stub that reads X[:, 0]
    mon = CalibrationMonitor(alpha=0.05)              # with a few batches of a calibrated forecast applied
    ```

    Every block also names, in its first comment line, which of these it uses.
