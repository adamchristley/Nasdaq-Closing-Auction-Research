from __future__ import annotations

import numpy as np
import pandas as pd


def long_short_diagnostic(
    frame: pd.DataFrame,
    pred_col: str = "prediction",
    quantile: float = 0.10,
    round_trip_cost_bps: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Cross-sectional signal diagnostic using top/bottom prediction tails.

    Target is already expressed in basis points relative to a synthetic index.
    This is a research diagnostic, not a claim of executable PnL.
    """
    rows = []
    for (date_id, seconds), g in frame.groupby(["date_id", "seconds_in_bucket"], sort=True):
        g = g[["stock_id", "target", pred_col]].dropna()
        if len(g) < 20:
            continue
        q_lo = g[pred_col].quantile(quantile)
        q_hi = g[pred_col].quantile(1.0 - quantile)
        long = g[g[pred_col] >= q_hi]
        short = g[g[pred_col] <= q_lo]
        if long.empty or short.empty:
            continue
        gross = float(long["target"].mean() - short["target"].mean())
        net = gross - round_trip_cost_bps
        rows.append({"date_id":int(date_id),"seconds_in_bucket":int(seconds),"gross_spread_bps":gross,"net_spread_bps":net,"n_long":len(long),"n_short":len(short)})
    ts = pd.DataFrame(rows)
    if ts.empty:
        return ts, {"mean_gross_bps":np.nan,"mean_net_bps":np.nan,"positive_net_share":np.nan,"diagnostic_sharpe":np.nan}
    std = ts["net_spread_bps"].std(ddof=1)
    sharpe = np.nan if std == 0 or not np.isfinite(std) else ts["net_spread_bps"].mean() / std * np.sqrt(len(ts))
    summary = {"mean_gross_bps":float(ts["gross_spread_bps"].mean()),"mean_net_bps":float(ts["net_spread_bps"].mean()),"positive_net_share":float((ts["net_spread_bps"] > 0).mean()),"diagnostic_sharpe":float(sharpe)}
    return ts, summary
