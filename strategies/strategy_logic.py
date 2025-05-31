import vectorbt as vbt
import pandas as pd
from typing import Dict, Any

def rsi_strategy(price: pd.Series, params: Dict[str, Any]) -> Dict[str, pd.Series]:
    rsi = vbt.RSI.run(price, window=params['window'])
    entries = rsi.rsi < params['entry']
    exits = rsi.rsi > params['exit']
    return {'entries': entries, 'exits': exits}

def macd_strategy(price: pd.Series, params: Dict[str, Any]) -> Dict[str, pd.Series]:
    macd = vbt.MACD.run(price, fast_window=params['fast'], slow_window=params['slow'], signal_window=params['signal'])
    entries = macd.macd > macd.signal
    exits = macd.macd < macd.signal
    return {'entries': entries, 'exits': exits}

# Placeholder for more strategies, e.g.
def sma_crossover_strategy(price: pd.Series, params: Dict[str, Any]) -> Dict[str, pd.Series]:
    fast = price.rolling(params['fast']).mean()
    slow = price.rolling(params['slow']).mean()
    entries = fast > slow
    exits = fast < slow
    return {'entries': entries, 'exits': exits}
