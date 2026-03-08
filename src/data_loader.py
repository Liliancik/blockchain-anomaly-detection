"""Load student performance data from CSV."""
import pandas as pd
from typing import Optional


def load_data(path: str = "data/StudentPerformanceFactors.csv") -> pd.DataFrame:
    """Load CSV into DataFrame."""
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        raise
    except Exception as e:
        raise
