"""Global minimum variance portfolio construction and backtesting."""

import numpy as np
import pandas as pd


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
    ones = np.ones(Omega.shape[0])
    w_unnorm = Omega @ ones
    return w_unnorm / w_unnorm.sum()


def portfolio_variance(w, Sigma):
    """Compute portfolio variance w' Sigma w.

    Parameters
    ----------
    w : np.ndarray, shape (p,)
    Sigma : np.ndarray, shape (p, p)

    Returns
    -------
    float
    """
    return float(w @ Sigma @ w)


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
    dates = returns_df.index
    rebal_dates = returns_df.resample(rebalance_freq).last().index
    rebal_dates = rebal_dates[rebal_dates >= dates[window_size]]

    portfolio_returns = []
    weights_history = {}
    prev_weights = None

    for rebal_date in rebal_dates:
        loc = returns_df.index.get_loc(rebal_date)
        if isinstance(loc, slice):
            loc = loc.stop - 1
        if loc < window_size:
            continue

        Y_window = returns_df.iloc[loc - window_size : loc].values
        Omega_hat = estimation_fn(Y_window)
        w = gmv_weights(Omega_hat)
        weights_history[rebal_date] = w

        next_rebal_idx = np.searchsorted(rebal_dates, rebal_date) + 1
        if next_rebal_idx >= len(rebal_dates):
            end_loc = len(returns_df)
        else:
            end_loc = returns_df.index.get_loc(rebal_dates[next_rebal_idx])
            if isinstance(end_loc, slice):
                end_loc = end_loc.stop - 1

        oos_returns = returns_df.iloc[loc:end_loc].values
        pf_rets = oos_returns @ w
        portfolio_returns.extend(pf_rets.tolist())

        if prev_weights is not None:
            pass  # turnover computed below
        prev_weights = w

    portfolio_returns = np.array(portfolio_returns)

    # Compute turnover from weights history
    w_list = list(weights_history.values())
    turnover_vals = []
    for i in range(1, len(w_list)):
        turnover_vals.append(np.sum(np.abs(w_list[i] - w_list[i - 1])))
    avg_turnover = float(np.mean(turnover_vals)) if turnover_vals else 0.0

    # Out-of-sample statistics
    vol_oos = float(np.std(portfolio_returns) * np.sqrt(252))
    mean_ret = float(np.mean(portfolio_returns) * 252)
    sharpe = mean_ret / vol_oos if vol_oos > 0 else 0.0

    # Max drawdown
    cum_returns = np.cumprod(1 + portfolio_returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - running_max) / running_max
    max_dd = float(np.min(drawdowns))

    return {
        "portfolio_returns": portfolio_returns,
        "weights_history": weights_history,
        "volatility_oos": vol_oos,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "turnover": avg_turnover,
    }
