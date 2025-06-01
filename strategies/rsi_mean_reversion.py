import backtrader as bt
import numpy as np
from strategies.base_strategy import BaseStrategy

class RSIMeanReversion(BaseStrategy):
    params = dict(
        rsi_period=14,
        rsi_lower=30,
        rsi_upper=70,
        contract_multiplier=1,
        debug=True,
        log_file="strategy_logs.txt"
    )

    def __init__(self):
        super().__init__()
        self.rsi = bt.indicators.RSI(self.datas[0].close, period=self.params.rsi_period)
        self.in_short = False
        self.in_long = False

    def next(self):
        # Ensure RSI is valid
        if len(self.data) < self.params.rsi_period or np.isnan(self.rsi[0]):
            return

        rsi = self.rsi[0]
        rsi_prev = self.rsi[-1] if len(self.rsi) > 1 else rsi  # Fallback if no lookback

        self.log(f"RSI: {rsi:.2f}, Position: {self.position.size}")

        # --- Exit Logic: FIRST, so it executes same-bar ---
        if self.in_long and rsi_prev < 50 and rsi >= 50:
            self.close()
            self.in_long = False
            self.log("🔁 CLOSING LONG: RSI crossed above 50")

        elif self.in_short and rsi_prev > 50 and rsi <= 50:
            self.close()
            self.in_short = False
            self.log("🔁 CLOSING SHORT: RSI crossed below 50")

        # --- Entry Logic ---
        if not self.position:
            if rsi < self.params.rsi_lower:
                size = self.place_safe_order(is_buy=True)
                if size > 0:
                    self.in_long = True
                    self.in_short = False
                    self.log(f"📈 BUY SIGNAL placed with size {size}")
            elif rsi > self.params.rsi_upper:
                size = self.place_safe_order(is_buy=False)
                if size > 0:
                    self.in_short = True
                    self.in_long = False
                    self.log(f"📉 SELL SIGNAL placed with size {size}")
