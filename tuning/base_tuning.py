import abc
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path


class BaseTuning(abc.ABC):
    """
    Abstract base class for all tuning engines (grid search, Bayesian, random, etc.)
    """

    def __init__(self, 
                 symbols: List[str], 
                 start_date: str, 
                 end_date: str,
                 output_dir: Optional[str] = "results"):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = Path(output_dir)
        self.results: Dict[str, pd.DataFrame] = {}

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

    @abc.abstractmethod
    def run(self) -> None:
        """Run the tuning logic across all symbols."""
        pass

    @abc.abstractmethod
    def report(self) -> None:
        """Generate a report of the tuning results."""
        pass

    @abc.abstractmethod
    def define_param_grid(self) -> Dict[str, List[Any]]:
        """Return a dictionary of parameters to sweep through."""
        pass

    @abc.abstractmethod
    def run_strategy(self, price: pd.Series, params: Dict[str, Any]) -> Dict[str, pd.Series]:
        """Execute strategy logic and return entry/exit signals."""
        pass
