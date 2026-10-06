import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from auction_research.split import chronological_date_split


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
