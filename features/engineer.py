""" Feature engineering for the transformer. 
Produces a DataFrame of features aligned to dates, where the value at date t
uses only information available through the close of day t. 
"""

import numpy as np
import pandas as pd
from features.realized_vol import realized_variance_daily, daily_returns

def build_features(data: dict) -> pd.DataFrame: 
    """ Construct the feature matrix. 
    Columns: 
    log_rv
    log_vix
    term_3m : log(VIX_3m/VIX)
    term_6m
    term_9d
    ret_1d
    ret_5d
    Drawdown: current pct distance from rolling 60-day max close
    """

    spx = data['spx']['Close']
    rv_daily = realized_variance_daily(spx)
    rets = daily_returns(spx)

    df = pd.DataFrame(index=rv_daily.index)
    df['log_rv'] = np.log(rv_daily)
    df['log_vix'] = np.log(data['vix']['Close'] /100)
    df['term_3m'] = np.log(data['vix_3m']['Close'] / data['vix']['Close'])
    df['term_6m'] = np.log(data['vix_6m']['Close'] / data['vix']['Close'])
    df['term_9d'] = np.log(data['vix_9d']['Close'] / data['vix']['Close'])
    df['ret_1d'] = rets
    df['ret_5d'] = rets.rolling(5).sum()
    rolling_max = spx.rolling(60).max()
    df['drawdown'] = np.log(spx / rolling_max)
    return df.dropna()