from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    """Masks and date boundaries for one leakage-safe walk-forward fold."""

    fold: int
    train_mask: np.ndarray
    validation_mask: np.ndarray
    train_start: object
    train_end: object
    validation_start: object
    validation_end: object
    n_train_dates: int
    n_validation_dates: int


def chronological_date_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return (
        df["date_id"].isin(train_dates).to_numpy(),
        df["date_id"].isin(val_dates).to_numpy(),
        df["date_id"].isin(test_dates).to_numpy(),
    )


def walk_forward_splits(
    df: pd.DataFrame,
    initial_train_dates: int,
    validation_dates: int,
    step_dates: int | None = None,
    gap_dates: int = 0,
    train_window_dates: int | None = None,
    max_folds: int | None = None,
) -> list[WalkForwardFold]:
    """Build expanding- or rolling-window chronological validation folds.

    Each validation block contains complete future date_id values. Training
    ends before validation starts, with an optional date gap between them.
    train_window_dates=None gives an expanding window; otherwise only the
    most recent train_window_dates dates are used for each fold.
    """
    if initial_train_dates < 1:
        raise ValueError("initial_train_dates must be at least 1.")
    if validation_dates < 1:
        raise ValueError("validation_dates must be at least 1.")
    if gap_dates < 0:
        raise ValueError("gap_dates cannot be negative.")
    if train_window_dates is not None and train_window_dates < initial_train_dates:
        raise ValueError(
            "train_window_dates must be at least initial_train_dates when provided."
        )
    if max_folds is not None and max_folds < 1:
        raise ValueError("max_folds must be at least 1 when provided.")

    step = validation_dates if step_dates is None else step_dates
    if step < 1:
        raise ValueError("step_dates must be at least 1.")

    dates = np.array(sorted(df["date_id"].dropna().unique()))
    first_validation_start = initial_train_dates + gap_dates
    if first_validation_start + validation_dates > len(dates):
        raise ValueError(
            "Not enough distinct date_id values for the requested walk-forward setup."
        )

    folds: list[WalkForwardFold] = []
    validation_start_idx = first_validation_start

    while validation_start_idx + validation_dates <= len(dates):
        train_end_idx = validation_start_idx - gap_dates
        train_start_idx = 0
        if train_window_dates is not None:
            train_start_idx = max(0, train_end_idx - train_window_dates)

        train_dates = dates[train_start_idx:train_end_idx]
        validation_block = dates[
            validation_start_idx:validation_start_idx + validation_dates
        ]

        folds.append(
            WalkForwardFold(
                fold=len(folds),
                train_mask=df["date_id"].isin(train_dates).to_numpy(),
                validation_mask=df["date_id"].isin(validation_block).to_numpy(),
                train_start=train_dates[0],
                train_end=train_dates[-1],
                validation_start=validation_block[0],
                validation_end=validation_block[-1],
                n_train_dates=len(train_dates),
                n_validation_dates=len(validation_block),
            )
        )

        if max_folds is not None and len(folds) >= max_folds:
            break
        validation_start_idx += step

    return folds
