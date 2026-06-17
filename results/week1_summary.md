# Week 1 Baseline Results

Test window: 2010-01-01 to 2024-12-31, refit every 21 business days. 
Target: 21-day forward annualized realized variance, S&P 500. 

| Model | QLIKE  | MSE        |
|-------|--------|------------|
|HAR    |0.391157|4.155604e-03|
|GARCH  |0.344189|5.736374e-03|

Notes:
- HAR R-squared on full sample: 0.37
- Diebold-Mariano test (HAR vs GARCH): QLIKE stat = 0.29, p = 0.77; MSE stat = -0.96, p = 0.34
- Neither baseline significantly outperforms the other on either metric
- GARCH wins on QLIKE (scale-invariant); HAR wins on MSE (absolute error)
- QLIKE is the primary metric for the paper (Patton 2011)
- n = 182 non-overlapping 21-day forecast periods
- Implementation in models/har.py and models/garch.py
- Reproduction: notebooks/02_har_baseline.ipynb, notebooks/03_garch_baseline.py