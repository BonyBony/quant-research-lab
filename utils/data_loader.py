# data/loader.py

import yfinance as yf
import pandas as pd

def load_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, group_by='ticker', auto_adjust=False)

    # Handle MultiIndex columns like ('AAPL', 'Open') or ('Open', 'AAPL')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[1] if col[1].lower() in ['open', 'high', 'low', 'close', 'adj close', 'volume']
                      else col[0] for col in df.columns]

    # If column names are just "AAPL", likely need second-level remap
    df.columns = [str(c).strip().lower().replace("adj close", "adj_close") for c in df.columns]

    rename_map = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adj_close": "adj_close",
        "volume": "volume"
    }
    df = df.rename(columns=rename_map)

    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing} | Got: {df.columns.tolist()}")

    df = df[required].dropna()
    df.index.name = "datetime"
    return df
