from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9
REQUIRED_COLUMNS = {"stock_id","date_id","seconds_in_bucket","imbalance_size","imbalance_buy_sell_flag","reference_price","matched_size","far_price","near_price","bid_price","bid_size","ask_price","ask_size","wap"}


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create contemporaneous and strictly lagged features without using target values."""
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    x = df.copy()
    x = x.sort_values(["date_id","stock_id","seconds_in_bucket"]).reset_index(drop=True)
    x["spread"] = x["ask_price"] - x["bid_price"]
    x["mid_price"] = (x["ask_price"] + x["bid_price"]) / 2.0
    x["book_imbalance"] = _safe_div(x["bid_size"] - x["ask_size"], x["bid_size"] + x["ask_size"])
    x["auction_imbalance_ratio"] = _safe_div(x["imbalance_size"], x["imbalance_size"] + x["matched_size"])
    x["signed_auction_imbalance"] = x["imbalance_buy_sell_flag"] * x["auction_imbalance_ratio"]
    x["microprice"] = _safe_div(x["ask_price"] * x["bid_size"] + x["bid_price"] * x["ask_size"], x["bid_size"] + x["ask_size"])
    x["microprice_minus_mid"] = x["microprice"] - x["mid_price"]
    x["reference_minus_wap"] = x["reference_price"] - x["wap"]
    x["near_minus_wap"] = x["near_price"] - x["wap"]
    x["far_minus_wap"] = x["far_price"] - x["wap"]
    x["mid_minus_wap"] = x["mid_price"] - x["wap"]
    x["log_matched_size"] = np.log1p(x["matched_size"].clip(lower=0))
    x["log_imbalance_size"] = np.log1p(x["imbalance_size"].clip(lower=0))
    x["log_book_size"] = np.log1p((x["bid_size"] + x["ask_size"]).clip(lower=0))
    x["auction_progress"] = x["seconds_in_bucket"] / 540.0
    x["auction_progress_sq"] = x["auction_progress"] ** 2
    grp = x.groupby(["stock_id","date_id"], sort=False)
    for lag in (1,3,6):
        x[f"wap_change_lag{lag}"] = grp["wap"].pct_change(periods=lag, fill_method=None)
        x[f"imbalance_change_lag{lag}"] = grp["imbalance_size"].pct_change(periods=lag, fill_method=None)
        x[f"book_imbalance_lag{lag}"] = grp["book_imbalance"].shift(lag)
        x[f"spread_lag{lag}"] = grp["spread"].shift(lag)
    tgrp = x.groupby(["date_id","seconds_in_bucket"], sort=False)
    for col in ("signed_auction_imbalance","book_imbalance","spread","wap"):
        mean = tgrp[col].transform("mean")
        std = tgrp[col].transform("std")
        x[f"{col}_xs_z"] = (x[col] - mean) / (std + EPS)
        x[f"{col}_xs_rank"] = tgrp[col].rank(pct=True)
    base = ["stock_id","seconds_in_bucket","imbalance_buy_sell_flag","reference_price","matched_size","far_price","near_price","bid_price","bid_size","ask_price","ask_size","wap","spread","mid_price","book_imbalance","auction_imbalance_ratio","signed_auction_imbalance","microprice","microprice_minus_mid","reference_minus_wap","near_minus_wap","far_minus_wap","mid_minus_wap","log_matched_size","log_imbalance_size","log_book_size","auction_progress","auction_progress_sq"]
    lagged = [c for c in x.columns if c.startswith(("wap_change_lag","imbalance_change_lag","book_imbalance_lag","spread_lag"))]
    cross_sectional = [c for c in x.columns if c.endswith(("_xs_z","_xs_rank"))]
    feature_cols = base + lagged + cross_sectional
    return x, feature_cols
