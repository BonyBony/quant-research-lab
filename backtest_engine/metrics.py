import pandas as pd
import numpy as np

def compute_metrics(returns: pd.Series) -> pd.DataFrame:
    cumulative_return = (1 + returns).prod() - 1
    annualized_return = (1 + cumulative_return) ** (252 / len(returns)) - 1
    annualized_vol = returns.std() * np.sqrt(252)
    sharpe = annualized_return / annualized_vol if annualized_vol else np.nan
    drawdown = (1 + returns).cumprod().div((1 + returns).cumprod().cummax()) - 1
    max_dd = drawdown.min()

    return pd.DataFrame.from_dict({
        "cumulative_return": [cumulative_return],
        "annualized_return": [annualized_return],
        "annualized_volatility": [annualized_vol],
        "sharpe_ratio": [sharpe],
        "max_drawdown": [max_dd]
    })
