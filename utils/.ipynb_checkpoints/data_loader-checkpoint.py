import yfinance as yf

def load_data(symbol='SPY', start='2015-01-01', end='2024-12-31'):
    df = yf.download(symbol, start=start, end=end)
    return df[['Close']].dropna()