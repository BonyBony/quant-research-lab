import pandas as pd
import numpy as np
from typing import Callable, Dict, Any
from itertools import product


def run_walk_forward_analysis(
    strategy_fn: Callable[[pd.Series, Dict[str, Any]], Dict[str, pd.Series]],
    price_data: pd.Series,
    param_grid: Dict[str, list],
    train_period: pd.DateOffset,
    test_period: pd.DateOffset,
    step_period: pd.DateOffset,
    metrics_fn: Callable[[Dict[str, pd.Series], pd.Series], Dict[str, float]] = None,
    run_name: str = "walk_forward"
):
    """
    Generic walk-forward analysis engine (function-based).

    Args:
        strategy_fn: Strategy function to run.
        price_data: Historical price series with DatetimeIndex.
        param_grid: Dictionary of hyperparameter values to search over.
        train_period: Offset specifying training window size.
        test_period: Offset specifying test window size.
        step_period: Offset by which the window advances.
        metrics_fn: Function to calculate performance metrics on test set.
        run_name: Optional name for this WFA run.

    Returns:
        List of dictionaries with best params, metrics, and fold information.
    """
    results = []
    start = price_data.index[0]
    end = price_data.index[-1]
    fold = 1

    while start + train_period + test_period <= end:
        train_end = start + train_period
        test_end = train_end + test_period

        train_series = price_data.loc[start:train_end]
        test_series = price_data.loc[train_end:test_end]

        print(f"Fold {fold}: Train {start.date()}–{train_end.date()} | Test {train_end.date()}–{test_end.date()}")

        best_score = -np.inf
        best_params = None
        best_test_result = None

        for combo in product(*param_grid.values()):
            params = dict(zip(param_grid.keys(), combo))
            train_result = strategy_fn(train_series, params)

            entries = train_result['entries']
            has_entries = entries.any().any() if isinstance(entries, pd.DataFrame) else entries.any()
            if not has_entries:
                continue

            test_result = strategy_fn(test_series, params)
            metrics = metrics_fn(test_result, test_series) if metrics_fn else {}
            score = metrics.get("sharpe", 0)

            if score > best_score:
                best_score = score
                best_params = params
                best_test_result = test_result

        results.append({
            "fold": fold,
            "train_start": start,
            "train_end": train_end,
            "test_start": train_end,
            "test_end": test_end,
            "best_params": best_params,
            "metrics": metrics_fn(best_test_result, test_series) if metrics_fn and best_test_result else {},
            "result": best_test_result
        })

        start += step_period
        fold += 1

    return results
