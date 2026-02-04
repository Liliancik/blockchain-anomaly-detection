import pandas as pd
from typing import Optional


def load_data(path: str = "data/StudentPerformanceFactors.csv") -> pd.DataFrame:
    """Load CSV data into a pandas DataFrame.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with the loaded data.
    """
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        raise
    except Exception as e:
        raise
