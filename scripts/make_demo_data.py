from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_demo(n_dates: int = 24, n_stocks: int = 50, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        regime = rng.normal(0, 0.25)
        for s in range(n_stocks):
            stock_bias = rng.normal(0, 0.2)
            base = 1.0 + rng.normal(0, 0.004)
            for sec in range(0, 550, 10):
                bid = base + rng.normal(0, 0.0015)
                spread = abs(rng.normal(0.0012, 0.00035)) + 1e-4
                ask = bid + spread
                bid_size = rng.lognormal(8.5, 0.55)
                ask_size = rng.lognormal(8.5, 0.55)
                book_imb = (bid_size - ask_size) / (bid_size + ask_size)
                flag = int(rng.choice([-1, 0, 1], p=[0.44, 0.12, 0.44]))
                imbalance = rng.lognormal(10.2, 0.7)
                matched = rng.lognormal(11.2, 0.55)
                auction_ratio = flag * imbalance / (imbalance + matched)
                wap = (bid * ask_size + ask * bid_size) / (bid_size + ask_size)
                reference = wap + rng.normal(0, 0.001)
                near = wap + 0.0020 * auction_ratio + rng.normal(0, 0.001)
                far = wap + 0.0030 * auction_ratio + rng.normal(0, 0.0014)
                signal = 6.0 * auction_ratio + 2.8 * book_imb - 500.0 * spread + 0.35 * regime + 0.2 * stock_bias
                target = signal + rng.normal(0, 2.5)
                rows.append({"stock_id":s,"date_id":d,"seconds_in_bucket":sec,"imbalance_size":imbalance,"imbalance_buy_sell_flag":flag,"reference_price":reference,"matched_size":matched,"far_price":far,"near_price":near,"bid_price":bid,"bid_size":bid_size,"ask_price":ask,"ask_size":ask_size,"wap":wap,"target":target})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/demo_train.csv")
    p.add_argument("--dates", type=int, default=24)
    p.add_argument("--stocks", type=int, default=50)
    args = p.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = make_demo(args.dates, args.stocks)
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} rows to {out}")


if __name__ == "__main__":
    main()
