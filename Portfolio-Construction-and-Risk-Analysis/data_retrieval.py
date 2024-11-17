import yfinance as yf
import pandas as pd
from typing import List

def yf_retrieve_data(tickers: List[str]) -> List[pd.DataFrame]:
    dataframes = []
    for ticker in tickers:
        history = yf.Ticker(ticker).history(period='10y')
        if history.isnull().any(axis=1).iloc[0]:
            history = history.iloc[1:]
        assert not history.isnull().any(axis=None), f'NaNs in {ticker}'
        dataframes.append(history)
    return dataframes