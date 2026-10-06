from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from auction_research.backtest import long_short_diagnostic
from auction_research.features import build_features
from auction_research.metrics import cross_sectional_ic, regression_metrics
from auction_research.models import make_models
from auction_research.split import chronological_date_split


def _fit_model(model, x_train, y_train, x_test):
    model.fit(x_train, y_train)
    return model.predict(x_test)


def run(input_path: Path, output_dir: Path, max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(input_path, nrows=max_rows)
    df = df.dropna(subset=["target"]).copy()
    feat, feature_cols = build_features(df)
    train_mask, val_mask, test_mask = chronological_date_split(feat)

    x_train = feat.loc[train_mask, feature_cols]
    y_train = feat.loc[train_mask, "target"].to_numpy()
    x_val = feat.loc[val_mask, feature_cols]
    y_val = feat.loc[val_mask, "target"].to_numpy()
    x_test = feat.loc[test_mask, feature_cols]
    y_test = feat.loc[test_mask, "target"].to_numpy()

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    zero_val = np.zeros(len(y_val))
    rows.append({"split":"validation","model":"zero_baseline",**regression_metrics(y_val, zero_val)})

    models = make_models()
    for name, model in models.items():
        pred = _fit_model(model, x_train, y_train, x_val)
        rows.append({"split":"validation","model":name,**regression_metrics(y_val, pred)})

    metrics_df = pd.DataFrame(rows)
    best_name = metrics_df[metrics_df["model"] != "zero_baseline"].sort_values("mae").iloc[0]["model"]

    model = make_models()[best_name]
    trainval_mask = train_mask | val_mask
    x_trainval = feat.loc[trainval_mask, feature_cols]
    y_trainval = feat.loc[trainval_mask, "target"].to_numpy()
    model.fit(x_trainval, y_trainval)
    pred_test = model.predict(x_test)

    test_metrics = {"split":"test","model":best_name,**regression_metrics(y_test, pred_test)}
    pred_frame = feat.loc[test_mask, ["stock_id","date_id","seconds_in_bucket","target"]].copy()
    pred_frame["prediction"] = pred_test
    test_metrics.update(cross_sectional_ic(pred_frame))

    bt, bt_summary = long_short_diagnostic(pred_frame, round_trip_cost_bps=1.0)
    test_metrics.update(bt_summary)
    metrics_df = pd.concat([metrics_df, pd.DataFrame([test_metrics])], ignore_index=True)

    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    pred_frame.to_csv(output_dir / "test_predictions.csv", index=False)
    bt.to_csv(output_dir / "signal_diagnostic.csv", index=False)
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump({"selected_model":best_name,"n_rows":int(len(feat)),"n_features":int(len(feature_cols)),"train_dates":int(feat.loc[train_mask,"date_id"].nunique()),"validation_dates":int(feat.loc[val_mask,"date_id"].nunique()),"test_dates":int(feat.loc[test_mask,"date_id"].nunique()),"feature_columns":feature_cols,"test_metrics":test_metrics}, f, indent=2)

    print(metrics_df.to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print(f"Artifacts written to: {output_dir}")
    return metrics_df


def main() -> None:
    p = argparse.ArgumentParser(description="Leakage-aware Nasdaq closing-auction research pipeline")
    p.add_argument("--input", default="data/train.csv")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--max-rows", type=int, default=None)
    args = p.parse_args()
    run(Path(args.input), Path(args.output_dir), args.max_rows)


if __name__ == "__main__":
    main()
