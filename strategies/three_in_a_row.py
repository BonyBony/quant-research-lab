import pandas as pd

def three_in_a_row_signal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['C_0'] = df['Close']
    df['C_5'] = df['Close'].shift(5)
    df['C_10'] = df['Close'].shift(10)
    df['C_15'] = df['Close'].shift(15)

    df['Signal'] = 0
    df.loc[(df['C_0'] > df['C_5']) & (df['C_5'] > df['C_10']) & (df['C_10'] > df['C_15']), 'Signal'] = 1
    return df.dropna()
