from strategies.three_in_a_row import three_in_a_row_signal
from backtest.engine import vectorized_three_in_a_row_backtest
from backtest.metrics import compute_all_metrics
from utils.data_loader import load_multi_asset_data

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def run_detailed_report():
    tickers = ['AAPL', 'MSFT', 'SPY', 'GLD', 'UUP']
    start_date = '2015-01-01'
    end_date = '2025-01-01'
    hold_days = 5

    all_data = load_multi_asset_data(tickers, start_date, end_date)
    portfolio_equity = pd.Series(dtype=float)
    full_results = []

    print("="*80)
    print("TRADING STRATEGY REPORT: Three-In-A-Row")
    print(f"Backtest Window: {start_date} to {end_date}")
    print(f"Holding Period: {hold_days} Days")
    print("="*80)

    for ticker in tickers:
        df = all_data['Close'][ticker].dropna().to_frame(name='Close')
        df = three_in_a_row_signal(df)
        df = vectorized_three_in_a_row_backtest(df, hold_days=hold_days)
        df['Equity'] = (1 + df['Strategy_Return']).cumprod()
        df['Year'] = df.index.year

        metrics = compute_all_metrics(df)
        metrics['Asset'] = ticker
        full_results.append(metrics)

        # Plot equity curve
        df['Equity'].plot(label=ticker)

        # Combine equity for portfolio view
        portfolio_equity = portfolio_equity.add(df['Equity'], fill_value=0)

        # Yearly breakdown
        yearly = df.groupby('Year')['Strategy_Return'].apply(lambda x: (1 + x).prod() - 1)
        print(f"\n--- {ticker} Yearly Returns ---")
        print(yearly.map(lambda x: f"{x:.2%}"))

    # Plot total equity curve
    plt.title("Equity Curves per Asset")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("="*80)
    print("PORTFOLIO SUMMARY METRICS")
    print("="*80)

    results_df = pd.DataFrame(full_results)
    results_df = results_df[['Asset', 'Sharpe Ratio', 'K-Ratio', 'Max Drawdown',
                             'Total Return', 'Win Rate', 'Profit/Loss Ratio', 'Expectancy per Trade']]
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    run_detailed_report()
