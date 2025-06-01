import pandas as pd
import itertools
from typing import List, Dict, Any, Callable
from tuning.base_tuning import BaseTuning
import vectorbt as vbt

class GridSearchTuner(BaseTuning):
    """
    Parameter tuner using full grid search (Cartesian product).
    Pass strategy function via strategy_fn: Callable[[pd.Series, Dict[str, Any]], Dict[str, pd.Series]]
    """

    def __init__(self,
                 symbols: List[str],
                 start_date: str,
                 end_date: str,
                 param_grid: Dict[str, List[Any]],
                 strategy_fn: Callable[[pd.Series, Dict[str, Any]], Dict[str, pd.Series]],
                 output_dir: str = "results"):
        super().__init__(symbols, start_date, end_date, output_dir)
        self.param_grid = param_grid
        self.strategy_fn = strategy_fn

    def define_param_grid(self) -> Dict[str, List[Any]]:
        return self.param_grid

    def run_strategy(self, price: pd.Series, params: Dict[str, Any]) -> Dict[str, pd.Series]:
        return self.strategy_fn(price, params)

    def run(self) -> None:
        for symbol in self.symbols:
            price = vbt.YFData.download(symbol, start=self.start_date, end=self.end_date).get('Close')
            param_grid = self.define_param_grid()
            keys, values = zip(*param_grid.items())
            param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
            result_rows = []
            for param_set in param_combinations:
                signals = self.run_strategy(price, param_set)
                entries, exits = signals['entries'], signals['exits']
                pf = vbt.Portfolio.from_signals(
                    close=price,
                    entries=entries,
                    exits=exits,
                    init_cash=100000,
                    fees=0.001,
                    direction='both',
                    freq='1D'
                )
                stats = pf.stats()
                row = {**param_set, 'Symbol': symbol,
                       'Total Return [%]': stats.loc['Total Return [%]'],
                       'Sharpe Ratio': stats.loc['Sharpe Ratio'],
                       'Max Drawdown [%]': stats.loc['Max Drawdown [%]']}
                result_rows.append(row)
            df = pd.DataFrame(result_rows)
            self.results[symbol] = df
            df.to_csv(self.output_dir / f"grid_search_{symbol}.csv", index=False)

    def report(self) -> None:
        for symbol, df in self.results.items():
            print(f"\n=== Report for {symbol} ===")
            top_row = df.sort_values(by="Sharpe Ratio", ascending=False).iloc[0]
            print("Top Configuration:")
            print(top_row)
            print("\nSummary Statistics:")
            print(df.describe())
