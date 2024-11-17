import matplotlib.pyplot as plt
from data_retrieval import yf_retrieve_data
from assets import Asset
from portfolio import Portfolio
import numpy as np


def main():
    stocks = ['AAPL', 'AMZN', 'GOOG', 'BRK-B', 'JNJ', 'JPM']
    dataframes = yf_retrieve_data(stocks)
    assets = tuple([Asset(name, df) for name, df in zip(stocks, dataframes)])
    portfolio = Portfolio(assets)

    portfolio.optimize_sharpe_ratio()
    print(f"Optimized Portfolio Weights: {portfolio.weights.flatten()}")

    # Plotting random portfolios
    X, y = [], []
    for _ in range(1000):
        p = Portfolio(assets)
        X.append(np.sqrt(p._variance(p.weights)))
        y.append(p._expected_return(p.weights))
    plt.scatter(X, y)
    plt.xlabel('Portfolio Std. Dev.')
    plt.ylabel('Expected Return')
    plt.show()

if __name__ == "__main__":
    main()