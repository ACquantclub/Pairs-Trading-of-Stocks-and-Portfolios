import numpy as np
import pandas as pd
from utils import TRADING_DAYS_PER_YEAR

def get_log_period_returns(price_history: pd.DataFrame) -> np.ndarray:
    close = price_history['Close'].values
    return np.log(close[1:] / close[:-1]).reshape(-1, 1)

class Asset:
    def __init__(self, name: str, daily_price_history: pd.DataFrame):
        self.name = name
        self.daily_returns = get_log_period_returns(daily_price_history)
        self.expected_daily_return = np.mean(self.daily_returns)

    @property
    def expected_return(self) -> float:
        return TRADING_DAYS_PER_YEAR * self.expected_daily_return

    @staticmethod
    def covariance_matrix(assets: tuple) -> np.ndarray:
        returns = [asset.daily_returns for asset in assets]
        expected_returns = np.array([a.expected_return for a in assets]).reshape(-1, 1)
        product_expectation = np.cov(np.hstack(returns), rowvar=False)
        return product_expectation * (TRADING_DAYS_PER_YEAR - 1)**2 - (expected_returns @ expected_returns.T)
