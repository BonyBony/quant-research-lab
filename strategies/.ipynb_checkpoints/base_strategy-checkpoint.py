import backtrader as bt
import numpy as np
import pandas as pd
from datetime import datetime
from utils.report import StrategyReport


class BaseStrategy(bt.Strategy):
    params = dict(
        account_risk=10000,
        volatility_window=100,
        contract_multiplier=1,
        debug=True,
        log_file="strategy_logs.txt",
        margin_buffer=0.9,
        max_retries=3
    )

    def __init__(self):
        self.trades = []
        self.pnl_total = 0.0
        self.total_contracts = 0
        self.total_return = 0.0
        self.trade_returns = []
        self.contract_entry_sizes = []
        self.current_contract = None
        self.closes = []
        self.returns_series = pd.Series(dtype=float)
        self.report = StrategyReport(returns=self.returns_series)

        self.trade_sizes = {}  # FIX: store executed size per trade

        returns = bt.indicators.PctChange(self.datas[0].close)
        self.volatility = bt.indicators.StandardDeviation(returns, period=self.params.volatility_window)

        self.log_f = open(self.params.log_file, "w")
        self.log("Strategy initialized.")

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        entry = f"{dt.isoformat()}, {txt}\n"
        self.log_f.write(entry)
        if self.params.debug:
            print(entry.strip())

    def next(self):
        self.closes.append(self.data.close[0])

    def get_vol_adjusted_contracts(self):
        vol = self.volatility[0]
        if np.isnan(vol) or vol <= 0:
            self.log("⚠️ Volatility invalid. Defaulting size to 1")
            return 1

        try:
            size = int(self.params.account_risk / (vol * self.params.contract_multiplier))
            return max(size, 1)
        except Exception as e:
            self.log(f"⚠️ Error calculating size: {e}")
            return 1

    def place_safe_order(self, is_buy=True):
        size = self.get_vol_adjusted_contracts()
        price = self.data.close[0]
        cash = self.broker.get_cash()

        if price is None or price <= 0 or cash <= 0:
            self.log("⚠️ Price or cash not available for order sizing.")
            return 0

        size = min(size, int((cash / price) * self.params.margin_buffer))
        if size <= 0:
            self.log("⚠️ Final calculated size is 0. Order skipped.")
            return 0

        retries = 0
        while retries < self.params.max_retries:
            try:
                self.log(f"Placing {'BUY' if is_buy else 'SELL'} for size: {size} (retry {retries})")
                if is_buy:
                    order = self.buy(size=size)
                else:
                    order = self.sell(size=size)
                return size
            except Exception as e:
                self.log(f"⚠️ Order failed: {e}. Reducing size.")
                size = int(size * 0.8)
                retries += 1

        return 0

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        elif order.status == order.Completed:
            action = 'BUY' if order.isbuy() else 'SELL'
            self.log(f"{action} EXECUTED at {order.executed.price:.2f} for {order.executed.size} units")

            size = abs(order.executed.size)
            if size > 0 and self.position:
                self.trade_sizes[self.position] = size  # FIX: store position -> size

            direction = "BUY" if order.isbuy() else "SELL"
            if self.current_contract is None:
                self.current_contract = {"entry_size": size, "direction": direction}
            else:
                if self.current_contract["direction"] != direction:
                    self.contract_entry_sizes.append(self.current_contract["entry_size"])
                    self.current_contract = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"ORDER FAILED: {order.getstatusname()} - Price: {order.created.price:.2f}, Size: {order.created.size}, Cash: {self.broker.get_cash():.2f}")

    def notify_trade(self, trade):
        self.log_trade(trade)

    def log_trade(self, trade):
        if not trade.isclosed:
            return

        try:
            entry_dt = trade.open_datetime()
            exit_dt = trade.close_datetime()
            entry_price = trade.price
            exit_price = (
            trade.price + trade.pnl / trade.size if trade.size != 0 else trade.price
        )

            # ✅ FIX: get contracts from trade_sizes tracking
            contracts = self.trade_sizes.pop(trade, None)
            if contracts is None or contracts == 0:
                contracts = (
                    abs(trade.history[0].size)
                    if hasattr(trade, "history") and trade.history
                    else 1
            )
                self.log(f"⚠️ Contract fallback used: {contracts}")

            pnl = trade.pnlcomm
            duration = (exit_dt - entry_dt).days if entry_dt and exit_dt else np.nan
            trade_return = (exit_price - entry_price) / entry_price if entry_price else 0.0

            trade_record = {
                "entry_dt": entry_dt,
                "exit_dt": exit_dt,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "contracts": contracts,  # ✅ CRITICAL
                "pnl": pnl,
                "return": trade_return,
                "duration": duration,
            }

            self.trades.append(trade_record)
            self.pnl_total += pnl
            self.total_contracts += contracts
            self.total_return += trade_return
            self.trade_returns.append(trade_return)
            self.log(f"📦 TRADE LOGGED: contracts={contracts}, pnl={pnl}")
        except Exception as e:
            self.log(f"⚠️ Failed to log trade: {e}")


    def stop(self):
        self.log(f"Final PnL accumulated: {self.pnl_total:.2f}")
        self.log(f"Total raw trades recorded: {len(self.trades)}")

        for i, t in enumerate(self.trades):
            if not isinstance(t, dict):
                self.log(f"❌ BAD TRADE ENTRY [{i}]: {t} (type={type(t)})")
            elif "contracts" not in t:
                self.log(f"❌ BAD TRADE FORMAT [{i}]: {t}")
            else:
                self.log(f"✅ GOOD TRADE [{i}]: contracts={t['contracts']}, pnl={t['pnl']}")

        self.trades = [t for t in self.trades if isinstance(t, dict) and "contracts" in t]

        if len(self.closes) >= 2:
            prices = pd.Series(self.closes)
            self.returns_series = prices.pct_change().dropna()
            self.report.returns = self.returns_series
        else:
            self.log("⚠️ Not enough close data to compute returns. Skipping return-based metrics.")

        if self.trades:
            self.report.trades = self.trades
            self.report.pnl_total = self.pnl_total

            if self.contract_entry_sizes:
                avg_contract = np.mean(self.contract_entry_sizes)
                total_contracts = len(self.contract_entry_sizes)
                self.report.metrics["average_contracts"] = avg_contract
                self.report.metrics["total_trades"] = total_contracts

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = f"strategy_metrics_{timestamp}.csv"
            pdf_path = f"strategy_report_{timestamp}.pdf"

            try:
                self.report.run_all(csv_path, pdf_path)
                self.log(f"📄 Strategy report saved to: {pdf_path}")
                self.log(f"📊 Metrics CSV exported to: {csv_path}")
            except Exception as e:
                self.log(f"🚨 Report generation failed: {e}")
        else:
            self.log("❌ No valid trades to report.")

        self.log("Stopping strategy and closing log file.")
        self.log_f.close()
