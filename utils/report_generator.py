import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional

class ReportGenerator:
    def __init__(self, results: Dict[str, pd.DataFrame]):
        """
        results: Dict mapping asset symbol to grid search DataFrame.
        """
        self.results = results

    def summary(self, sort_by: str = "Sharpe Ratio", top_n: int = 5) -> pd.DataFrame:
        all_best = []
        for symbol, df in self.results.items():
            top_df = df.sort_values(by=sort_by, ascending=False).head(top_n)
            all_best.append(top_df)
        summary_df = pd.concat(all_best, ignore_index=True)
        print("==== Top Configurations Across All Symbols ====")
        print(summary_df)
        return summary_df

    def plot_heatmap(self, symbol: str, metric: str = "Sharpe Ratio", index: str = None, columns: str = None):
        """
        Plot heatmap for a given symbol and metric.
        If index/columns are not specified, will try to infer (for RSI: entry vs window, etc).
        """
        df = self.results[symbol]
        if not index or not columns:
            # Try to infer
            params = [c for c in df.columns if c not in ["Symbol", "Total Return [%]", "Sharpe Ratio", "Max Drawdown [%]"]]
            if len(params) >= 2:
                index, columns = params[0], params[1]
            else:
                raise ValueError("Specify index/columns for heatmap")

        pivot = df.pivot_table(index=index, columns=columns, values=metric, aggfunc="max")
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
        plt.title(f"{symbol} - {metric} by {index} and {columns}")
        plt.tight_layout()
        plt.show()

    def export_summary_csv(self, filename: str = "all_results_summary.csv"):
        all_results = pd.concat(self.results.values(), ignore_index=True)
        all_results.to_csv(filename, index=False)
        print(f"Exported all results to {filename}")
