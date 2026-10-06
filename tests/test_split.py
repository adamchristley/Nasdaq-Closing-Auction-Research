import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from auction_research.split import chronological_date_split, walk_forward_splits


def test_split_is_chronological_and_disjoint():
    df = pd.DataFrame({"date_id": sum(([d] * 3 for d in range(10)), [])})
    tr, va, te = chronological_date_split(df)
    tr_dates = set(df.loc[tr, "date_id"])
    va_dates = set(df.loc[va, "date_id"])
    te_dates = set(df.loc[te, "date_id"])
    assert tr_dates.isdisjoint(va_dates)
    assert tr_dates.isdisjoint(te_dates)
    assert va_dates.isdisjoint(te_dates)
    assert max(tr_dates) < min(va_dates) < min(te_dates)


def test_walk_forward_expands_without_leakage():
    df = pd.DataFrame({"date_id": sum(([d] * 3 for d in range(20)), [])})
    folds = walk_forward_splits(
        df,
        initial_train_dates=8,
        validation_dates=3,
        step_dates=3,
    )

    assert [fold.validation_start for fold in folds] == [8, 11, 14, 17]
    assert [fold.n_train_dates for fold in folds] == [8, 11, 14, 17]

    for fold in folds:
        train_dates = set(df.loc[fold.train_mask, "date_id"])
        validation_dates = set(df.loc[fold.validation_mask, "date_id"])
        assert train_dates.isdisjoint(validation_dates)
        assert max(train_dates) < min(validation_dates)


def test_walk_forward_supports_gap_and_rolling_window():
    df = pd.DataFrame({"date_id": sum(([d] * 2 for d in range(20)), [])})
    folds = walk_forward_splits(
        df,
        initial_train_dates=6,
        validation_dates=2,
        step_dates=2,
        gap_dates=1,
        train_window_dates=6,
    )

    first = folds[0]
    train_dates = set(df.loc[first.train_mask, "date_id"])
    validation_dates = set(df.loc[first.validation_mask, "date_id"])

    assert train_dates == set(range(6))
    assert validation_dates == {7, 8}
    assert 6 not in train_dates
    assert 6 not in validation_dates

    for fold in folds:
        assert fold.n_train_dates == 6


def test_walk_forward_rejects_impossible_setup():
    df = pd.DataFrame({"date_id": list(range(6))})
    with pytest.raises(ValueError):
        walk_forward_splits(
            df,
            initial_train_dates=5,
            validation_dates=2,
        )
