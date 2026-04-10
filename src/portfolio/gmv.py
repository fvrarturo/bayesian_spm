"""Global minimum variance portfolio construction and backtesting."""

import numpy as np


def gmv_weights(Omega):
    """Compute global minimum variance portfolio weights from precision matrix.

    w* = Omega @ 1 / (1' @ Omega @ 1)

    Parameters
    ----------
    Omega : np.ndarray, shape (p, p)
        Precision matrix (must be positive definite).

    Returns
    -------
    w : np.ndarray, shape (p,)
        Portfolio weights summing to 1.
    """
    raise NotImplementedError


def rolling_backtest(
    returns_df,
    estimation_fn,
    window_size=250,
    rebalance_freq="M",
):
    """Rolling-window out-of-sample portfolio backtest.

    Parameters
    ----------
    returns_df : pd.DataFrame, shape (T_total, p)
        Full return series indexed by date.
    estimation_fn : callable
        Function that takes (T, p) return matrix and returns Omega estimate.
    window_size : int
        Number of trading days in the estimation window.
    rebalance_freq : str
        Rebalancing frequency: "M" (monthly), "Q" (quarterly), "W" (weekly).

    Returns
    -------
    dict
        Keys: "portfolio_returns", "weights_history", "volatility_oos",
        "sharpe_ratio", "max_drawdown", "turnover".
    """
    raise NotImplementedError
