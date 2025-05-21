import backtrader as bt
import pandas as pd
import numpy as np
from backtest_engine.metrics import compute_metrics

class BacktestRunner:
    def __init__(self, cash: float = 100_000):
        self.cash = cash

    def run(self, strategy_class, df, return_trades=False, return_strategy=False, **params):
        cerebro = bt.Cerebro()
        cerebro.broker.set_cash(self.cash)

        data = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data)

        cerebro.addstrategy(strategy_class, **params)

        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="returns")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trade_stats")

        # Run cerebro
        results = cerebro.run()
        strategy_instance = results[0]

        returns = pd.Series(strategy_instance.analyzers.returns.get_analysis())
        trade_analysis = strategy_instance.analyzers.trade_stats.get_analysis()

        # Metrics calculations
        cumulative_return = (1 + returns).prod() - 1
        annualized_return = (1 + cumulative_return) ** (252 / len(returns)) - 1
        annualized_vol = returns.std() * np.sqrt(252)

        metrics = pd.DataFrame([{
            "cumulative_return": cumulative_return,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_vol
        }])

        return_values = (metrics, returns)
        if return_trades:
            return_values += (trade_analysis,)
        if return_strategy:
            return_values += (strategy_instance,)

        return return_values
