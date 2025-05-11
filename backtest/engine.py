import pandas as pd

def vectorized_long_only_backtest(df: pd.DataFrame, hold_days: int = 5) -> pd.DataFrame:
    df = df.copy()
    df['BuyHold_Return'] = df['Close'].pct_change().shift(-1)
    df['Future_Return'] = df['Close'].pct_change(periods=hold_days).shift(-hold_days)
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Future_Return']
    return df.dropna()