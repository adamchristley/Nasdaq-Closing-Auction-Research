import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from auction_research.features import build_features


def test_target_is_never_a_feature():
    df = pd.DataFrame({"stock_id":[0,0],"date_id":[0,0],"seconds_in_bucket":[0,10],"imbalance_size":[100,120],"imbalance_buy_sell_flag":[1,1],"reference_price":[1.0,1.01],"matched_size":[1000,1010],"far_price":[1.0,1.01],"near_price":[1.0,1.01],"bid_price":[0.99,1.00],"bid_size":[500,520],"ask_price":[1.01,1.02],"ask_size":[480,490],"wap":[1.0,1.01],"target":[4.0,-2.0]})
    _, cols = build_features(df)
    assert "target" not in cols
    assert all("target" not in c for c in cols)


def test_lags_do_not_look_forward():
    rows = []
    for sec, wap in [(0,1.00),(10,1.01),(20,1.03)]:
        rows.append({"stock_id":0,"date_id":0,"seconds_in_bucket":sec,"imbalance_size":100,"imbalance_buy_sell_flag":1,"reference_price":wap,"matched_size":1000,"far_price":wap,"near_price":wap,"bid_price":wap-.01,"bid_size":500,"ask_price":wap+.01,"ask_size":500,"wap":wap})
    f, _ = build_features(pd.DataFrame(rows))
    assert np.isnan(f.loc[0, "wap_change_lag1"])
    assert np.isclose(f.loc[1, "wap_change_lag1"], 0.01)
