import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

class StrategyReport:
    def __init__(self, returns: pd.Series, trades: list = None, pnl_total: float = 0.0, risk_free_rate: float = 0.0):
        self.returns = returns.copy()
        self.risk_free_rate = risk_free_rate
        self.trades = trades or []
        self.pnl_total = pnl_total
        self.metrics = {}

    def calculate_performance_metrics(self):
        returns = self.returns
        cumulative_return = (1 + returns).prod() - 1
        annualized_return = (1 + cumulative_return) ** (252 / len(returns)) - 1
        annualized_vol = returns.std() * np.sqrt(252)

        excess_returns = returns - self.risk_free_rate / 252
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() != 0 else np.nan

        downside_returns = returns[returns < 0]
        downside_dev = downside_returns.std() * np.sqrt(252)
        sortino = excess_returns.mean() / downside_dev if downside_dev != 0 else np.nan

        cumulative = (1 + returns).cumprod()
        drawdown = cumulative.div(cumulative.cummax()) - 1
        max_drawdown = drawdown.min()

        log_equity = np.log(cumulative).replace([np.inf, -np.inf], np.nan).dropna()
        if len(log_equity) > 1:
            x = np.arange(len(log_equity))
            slope, _, _, _, std_err = stats.linregress(x, log_equity)
            k_ratio = slope / std_err if std_err != 0 else np.nan
        else:
            k_ratio = np.nan

        self.metrics.update({
            "cumulative_return": cumulative_return,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "k_ratio": k_ratio,
            "max_drawdown": max_drawdown
        })

    #!/usr/bin/python
# -*- coding: utf-8 -*-


    def calculate_trade_statistics(self):
        trades = [t for t in self.trades if isinstance(t, dict) and "contracts" in t]

        if not trades:
            self.metrics.update(
                {
                    "total_trades": 0,
                    "average_contracts": np.nan,
                    "avg_profit_per_contract": np.nan,
                    "avg_bars_win": np.nan,
                    "avg_bars_loss": np.nan,
                    "win_rate": np.nan,
                    "average_trade_return": np.nan,
                    "profit_factor": np.nan,
                }
            )
            return

        total_trades = len(trades)
        total_contracts = sum(t["contracts"] for t in trades)
        average_contracts = total_contracts / total_trades if total_trades > 0 else np.nan
        denominator = total_trades * average_contracts if average_contracts > 0 else np.nan
        avg_profit_per_contract = (
            self.pnl_total / denominator if denominator and denominator > 0 else np.nan
        )

        winning_trades = [t for t in trades if t["pnl"] > 0]
        losing_trades = [t for t in trades if t["pnl"] <= 0]

        win_rate = len(winning_trades) / total_trades
        avg_bars_win = (
            np.mean([(t["exit_dt"] - t["entry_dt"]).days for t in winning_trades])
            if winning_trades
            else np.nan
        )
        avg_bars_loss = (
            np.mean([(t["exit_dt"] - t["entry_dt"]).days for t in losing_trades])
            if losing_trades
            else np.nan
        )
        average_trade_return = np.mean([t.get("return", 0.0) for t in trades])
    
        gross_profit = sum(t["pnl"] for t in winning_trades)
        gross_loss = abs(sum(t["pnl"] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

        self.metrics.update(
            {
                "total_trades": total_trades,
                "average_contracts": average_contracts,
                "avg_profit_per_contract": avg_profit_per_contract,
                "avg_bars_win": avg_bars_win,
                "avg_bars_loss": avg_bars_loss,
                "win_rate": win_rate,
                "average_trade_return": average_trade_return,
                "profit_factor": profit_factor,
            }
        )



    def calculate_market_structure_correlation(self):
        self.metrics.update({
            "correlation_to_breakout": np.nan,
            "correlation_to_ma_10_40": np.nan
        })

    def generate_pdf_report(self, file_path: str):
        returns = self.returns.copy()
        returns.index = pd.to_datetime(returns.index)

        df_metrics = pd.DataFrame([self.metrics]).T.reset_index()
        df_metrics.columns = ['Metric', 'Value']
        df_metrics['Value'] = df_metrics['Value'].apply(lambda x: f"{x:.6f}" if isinstance(x, (float, int)) else x)

        with PdfPages(file_path) as pdf:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            table = ax.table(
                cellText=df_metrics.values,
                colLabels=["Metric", "Value"],
                colColours=["#f4f4f4", "#f4f4f4"],
                cellLoc='left',
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.1, 1.4)
            ax.set_title("Extended Strategy Metrics", fontsize=14, fontweight='bold')
            pdf.savefig(fig)
            plt.close()

            yearly_pnl = (1 + returns).groupby(returns.index.year).prod() - 1
            fig, ax = plt.subplots(figsize=(8.5, 4))
            yearly_pnl.plot(kind="bar", ax=ax, color="#4a90e2")
            ax.set_title("Net Profit by Year", fontsize=12)
            ax.set_ylabel("Net Profit")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
            ax.set_yticklabels([f"{y:.2%}" for y in ax.get_yticks()])
            pdf.savefig(fig)
            plt.close()

            cumulative = (1 + returns).cumprod()
            fig, ax = plt.subplots(figsize=(8.5, 4))
            cumulative.plot(ax=ax, color="green", title="Equity Curve")
            ax.set_ylabel("Equity")
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
            pdf.savefig(fig)
            plt.close()

            drawdown = cumulative.div(cumulative.cummax()) - 1
            fig, ax = plt.subplots(figsize=(8.5, 4))
            drawdown.plot(ax=ax, color="red", title="Drawdown Curve")
            ax.set_ylabel("Drawdown")
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
            pdf.savefig(fig)
            plt.close()

        return file_path

    def export_to_csv(self, file_path: str):
        df_metrics = pd.DataFrame([self.metrics])
        df_metrics.to_csv(file_path, index=False)
        return file_path

    def run_all(self, csv_path: str, pdf_path: str):
        self.calculate_performance_metrics()
        self.calculate_trade_statistics()
        self.calculate_market_structure_correlation()
        self.export_to_csv(csv_path)
        self.generate_pdf_report(pdf_path)
        return csv_path, pdf_path
