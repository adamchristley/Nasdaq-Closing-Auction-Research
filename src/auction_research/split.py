from __future__ import annotations

import numpy as np
import pandas as pd


def chronological_date_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split by whole date_id blocks so future dates never leak into training."""
    dates = np.array(sorted(df["date_id"].dropna().unique()))
    if len(dates) < 5:
        raise ValueError("Need at least 5 distinct date_id values for a chronological split.")
    n_train = max(1, int(len(dates) * train_frac))
    n_val = max(1, int(len(dates) * val_frac))
    if n_train + n_val >= len(dates):
        n_val = 1
        n_train = len(dates) - 2
    train_dates = dates[:n_train]
    val_dates = dates[n_train:n_train + n_val]
    test_dates = dates[n_train + n_val:]
    return (df["date_id"].isin(train_dates).to_numpy(),df["date_id"].isin(val_dates).to_numpy(),df["date_id"].isin(test_dates).to_numpy())
