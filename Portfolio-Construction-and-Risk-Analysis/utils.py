import numpy as np

TREASURY_BILL_RATE = 0.048
TRADING_DAYS_PER_YEAR = 252

def random_weights(weight_count: int) -> np.ndarray:
    """
    Generates a numpy array of random weights.
    """
    weights = np.random.random(weight_count)
    weights /= np.sum(weights)
    return weights