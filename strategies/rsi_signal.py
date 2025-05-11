import pandas as pd

def compute_rsi(series: pd.Series, lookback: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=lookback).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=lookback).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def rsi_signal(df: pd.DataFrame, rsi_thresh: float = 30, lookback: int = 14) -> pd.DataFrame:
    df = df.copy()
    df['RSI'] = compute_rsi(df['Close'], lookback)
    df['Signal'] = (df['RSI'] < rsi_thresh).astype(int)
    return df