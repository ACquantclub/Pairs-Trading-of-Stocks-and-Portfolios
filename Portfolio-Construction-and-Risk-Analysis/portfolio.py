import numpy as np
from scipy.optimize import minimize
from assets import Asset
from utils import TREASURY_BILL_RATE, random_weights

class Portfolio:
    def __init__(self, assets: tuple):
        """
        Initialize the portfolio with a tuple of assets.
        """
        self.assets = assets
        self.asset_expected_returns = np.array([asset.expected_return for asset in assets]).reshape(-1, 1)
        self.covariance_matrix = Asset.covariance_matrix(assets)
        self.weights = random_weights(len(assets))

    def _expected_return(self, w) -> float:
        """
        Calculate the expected return of the portfolio given weights.
        """
        return (self.asset_expected_returns.T @ w.reshape(-1, 1))[0, 0]

    def _variance(self, w) -> float:
        """
        Calculate the variance of the portfolio given weights.
        """
        return (w.reshape(-1, 1).T @ self.covariance_matrix @ w.reshape(-1, 1))[0, 0]

    def unsafe_optimize_with_risk_tolerance(self, risk_tolerance: float) -> None:
        """
        Optimize the portfolio weights to maximize expected return given a risk tolerance.
        """
        res = minimize(
            lambda w: self._variance(w) - risk_tolerance * self._expected_return(w),
            random_weights(len(self.assets)),
            constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}],
            bounds=[(0., 1.) for _ in range(len(self.assets))]
        )
        assert res.success, f'Optimization failed: {res.message}'
        self.weights = res.x.reshape(-1, 1)

    def optimize_with_risk_tolerance(self, risk_tolerance: float) -> None:
        """
        Safe optimization with a non-negative risk tolerance.
        """
        assert risk_tolerance >= 0., "Risk tolerance must be non-negative."
        self.unsafe_optimize_with_risk_tolerance(risk_tolerance)

    def optimize_with_expected_return(self, expected_portfolio_return: float) -> None:
        """
        Optimize the portfolio to achieve a specific expected return.
        """
        res = minimize(
            lambda w: self._variance(w),
            random_weights(len(self.assets)),
            constraints=[
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: self._expected_return(w) - expected_portfolio_return}
            ],
            bounds=[(0., 1.) for _ in range(len(self.assets))]
        )
        assert res.success, f'Optimization failed: {res.message}'
        self.weights = res.x.reshape(-1, 1)

    def optimize_sharpe_ratio(self) -> None:
        """
        Optimize the portfolio weights to maximize the Sharpe ratio.
        """
        def objective_func(w):
            return -(self._expected_return(w) - TREASURY_BILL_RATE) / np.sqrt(self._variance(w))
        
        # Set up the optimization problem
        res = minimize(
            objective_func,
            random_weights(len(self.assets)),  # x0 is now a 1-dimensional array
            constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}],
            bounds=[(0., 1.)] * len(self.assets)
        )
        
        # Check if optimization was successful
        assert res.success, f'Optimization failed: {res.message}'
        self.weights = res.x.reshape(-1, 1)


    @property
    def expected_return(self) -> float:
        """
        Return the expected return of the portfolio based on current weights.
        """
        return self._expected_return(self.weights)

    @property
    def variance(self) -> float:
        """
        Return the variance of the portfolio based on current weights.
        """
        return self._variance(self.weights)

    def __repr__(self) -> str:
        """
        String representation of the Portfolio.
        """
        return (f'<Portfolio assets={[asset.name for asset in self.assets]}, '
                f'expected return={self.expected_return:.4f}, variance={self.variance:.4f}>')
