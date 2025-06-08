import pandas as pd
import numpy as np
import vectorbt as vbt
import matplotlib.pyplot as plt
from typing import Dict, List

def compute_sharpe(returns: pd.Series, annualization_factor: int = 252) -> float:
    if returns.std() == 0:
        return 0.0
    return np.mean(returns) / np.std(returns) * np.sqrt(annualization_factor)

def portfolio_metrics(result: Dict[str, pd.Series], test_price: pd.Series) -> Dict[str, float]:
    """
    Compute key performance metrics using Vectorbt's Portfolio engine.
    """
    entries = result['entries']
    exits = result['exits']

    pf = vbt.Portfolio.from_signals(
        close=test_price,
        entries=entries,
        exits=exits,
        freq='1D'
    )

    return {
        'Sharpe Ratio': pf.sharpe_ratio().values[0],
        'Total Return [%]': pf.total_return().values[0] * 100,
        'Max Drawdown [%]': pf.max_drawdown().values[0] * 100,
        'Equity Curve': pf.value()  # Optional for plotting/debugging
    }

def summarize_wfa_results(results: List[Dict]) -> pd.DataFrame:
    """
    Convert list of WFA fold dicts into a structured summary DataFrame.
    """
    summary = []
    for i, fold in enumerate(results):
        metrics = fold['metrics']
        summary.append({
            "Fold": i + 1,
            "Train Start": fold["train_start"],
            "Train End": fold["train_end"],
            "Test Start": fold["test_start"],
            "Test End": fold["test_end"],
            "Sharpe Ratio": float(metrics["Sharpe Ratio"]),
            "Total Return [%]": float(metrics["Total Return [%]"]),
            "Max Drawdown [%]": float(metrics["Max Drawdown [%]"]),
        })
    return pd.DataFrame(summary)

def plot_wfa_performance(results: List[Dict]):
    """
    Plot Sharpe Ratio and Total Return over each WFA fold.
    """
    df = summarize_wfa_results(results)
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax[0].plot(df['Fold'], df['Sharpe Ratio'], marker='o')
    ax[0].set_ylabel('Sharpe Ratio')
    ax[0].set_title('Walk Forward Analysis - Sharpe Ratio per Fold')
    ax[0].grid(True)

    ax[1].plot(df['Fold'], df['Total Return [%]'], marker='o', color='green')
    ax[1].set_ylabel('Total Return [%]')
    ax[1].set_title('Walk Forward Analysis - Total Return per Fold')
    ax[1].set_xlabel('Fold')
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()
