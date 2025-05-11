import numpy as np
import statsmodels.api as sm

def compute_all_metrics(df, return_col='Strategy_Return', benchmark_col='BuyHold_Return', risk_free_rate=0.01, freq=252):
    df = df.copy().dropna()
    returns = df[return_col]
    benchmark = df[benchmark_col]
    excess = returns - (risk_free_rate / freq)

    total_return = (1 + returns).prod() - 1
    cagr = (1 + total_return) ** (freq / len(returns)) - 1
    std_return = returns.std() * np.sqrt(freq)
    variance = returns.var() * freq
    downside_std = returns[returns < 0].std() * np.sqrt(freq)
    sharpe = excess.mean() / returns.std() * np.sqrt(freq)
    sortino = excess.mean() / downside_std if downside_std != 0 else np.nan

    X = sm.add_constant(benchmark)
    model = sm.OLS(returns, X).fit()
    alpha = model.params[0] * freq
    beta = model.params[1]

    treynor = returns.mean() * freq / beta if beta != 0 else np.nan
    tracking_error = (returns - benchmark).std() * np.sqrt(freq)
    info_ratio = (returns - benchmark).mean() / (returns - benchmark).std() * np.sqrt(freq) if tracking_error != 0 else np.nan

    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = cumulative / rolling_max - 1
    max_drawdown = drawdown.min()

    log_equity = np.log(cumulative)
    x = sm.add_constant(np.arange(len(log_equity)))
    k_model = sm.OLS(log_equity.values, x).fit()
    slope = k_model.params[1]
    stderr = k_model.bse[1]
    k_ratio = slope / stderr if stderr != 0 else np.nan

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = len(wins) / len(returns)
    avg_win = wins.mean()
    avg_loss = losses.mean()
    pl_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else np.nan
    expectancy = returns.mean()

    return {
        'Total Return': f"{total_return:.2%}",
        'CAGR': f"{cagr:.2%}",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Sortino Ratio': f"{sortino:.2f}",
        'Treynor Ratio': f"{treynor:.2f}",
        'Information Ratio': f"{info_ratio:.2f}",
        'Alpha (annual)': f"{alpha:.2%}",
        'Beta': f"{beta:.2f}",
        'K-Ratio': f"{k_ratio:.2f}",
        'Max Drawdown': f"{max_drawdown:.2%}",
        'Annual Variance': f"{variance:.4f}",
        'Annual Std Dev': f"{std_return:.2%}",
        'Win Rate': f"{win_rate:.2%}",
        'Avg Win': f"{avg_win:.2%}",
        'Avg Loss': f"{avg_loss:.2%}",
        'Profit/Loss Ratio': f"{pl_ratio:.2f}",
        'Expectancy per Trade': f"{expectancy:.4f}"
    }