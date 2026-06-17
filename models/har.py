from dataclasses import dataclass
import numpy as np
import pandas as pd
import statsmodels.api as sm

@dataclass
class HARFit:
    coefficients: pd.Series
    residuals: pd.Series
    fitted_values: pd.Series
    r_squared: float

def build_har_features(rv_daily: pd.Series) -> pd.DataFrame:
    """
    Build the three HAR features from a daily RV series.
    Returns a DataFrame indexed by date with columns rv_d, rv_w, rv_m.
    Index t in the output uses information available through the close of day t.
    """
    rv_d = rv_daily.copy()
    rv_w = rv_daily.rolling(5).mean()
    rv_m = rv_daily.rolling(22).mean()
    df = pd.DataFrame({"rv_d": rv_d, "rv_w": rv_w, "rv_m": rv_m})
    return df.dropna()

def fit_har(features: pd.DataFrame, target: pd.Series) -> HARFit:
    """Fit HAR via OLS. Aligns features and target on common dates."""
    common = features.index.intersection(target.index)
    X = features.loc[common]
    y = target.loc[common]

    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    return HARFit(
        coefficients=model.params,
        residuals=model.resid,
        fitted_values=model.fittedvalues,
        r_squared=model.rsquared,
    )

def predict_har(fit: HARFit, features: pd.DataFrame) -> pd.Series:
    """Generate predictions from a fitted HAR model on new features."""
    X_const = sm.add_constant(features, has_constant="add")
    # Reorder columns to match training order
    X_const = X_const[fit.coefficients.index]
    return pd.Series(X_const.values @ fit.coefficients.values, index=features.index)