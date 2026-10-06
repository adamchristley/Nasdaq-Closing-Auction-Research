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
from auction_research.models import fit_predict, make_models
from auction_research.split import walk_forward_splits


def run_walk_forward(
    input_path: Path,
    output_dir: Path,
    initial_train_dates: int,
    validation_dates: int,
    step_dates: int | None,
    gap_dates: int,
    train_window_dates: int | None,
    max_folds: int | None,
    model_names: list[str],
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(input_path, nrows=max_rows).dropna(subset=["target"]).copy()
    feat, feature_cols = build_features(df)

    folds = walk_forward_splits(
        feat,
        initial_train_dates=initial_train_dates,
        validation_dates=validation_dates,
        step_dates=step_dates,
        gap_dates=gap_dates,
        train_window_dates=train_window_dates,
        max_folds=max_folds,
    )

    available = make_models()
    unknown = sorted(set(model_names).difference(available))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Available: {sorted(available)}")

    rows: list[dict[str, float | int | str]] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for fold in folds:
        train = feat.loc[fold.train_mask]
        val = feat.loc[fold.validation_mask]
        x_train = train[feature_cols]
        y_train = train["target"].to_numpy()
        x_val = val[feature_cols]
        y_val = val["target"].to_numpy()

        zero_pred = np.zeros(len(y_val))
        zero_frame = val[
            ["stock_id", "date_id", "seconds_in_bucket", "target"]
        ].copy()
        zero_frame["prediction"] = zero_pred
        zero_metrics = regression_metrics(y_val, zero_pred)
        zero_metrics.update(cross_sectional_ic(zero_frame))
        _, zero_signal = long_short_diagnostic(zero_frame)

        rows.append(
            {
                "fold": fold.fold,
                "model": "zero_baseline",
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "validation_start": fold.validation_start,
                "validation_end": fold.validation_end,
                "n_train_dates": fold.n_train_dates,
                "n_validation_dates": fold.n_validation_dates,
                **zero_metrics,
                **zero_signal,
            }
        )

        for name in model_names:
            model = make_models()[name]
            pred = fit_predict(model, x_train, y_train, x_val)

            pred_frame = val[
                ["stock_id", "date_id", "seconds_in_bucket", "target"]
            ].copy()
            pred_frame["prediction"] = pred

            metrics = regression_metrics(y_val, pred)
            metrics.update(cross_sectional_ic(pred_frame))
            _, signal = long_short_diagnostic(pred_frame)

            rows.append(
                {
                    "fold": fold.fold,
                    "model": name,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "validation_start": fold.validation_start,
                    "validation_end": fold.validation_end,
                    "n_train_dates": fold.n_train_dates,
                    "n_validation_dates": fold.n_validation_dates,
                    **metrics,
                    **signal,
                }
            )

    fold_metrics = pd.DataFrame(rows)
    metadata_cols = {
        "fold",
        "model",
        "train_start",
        "train_end",
        "validation_start",
        "validation_end",
        "n_train_dates",
        "n_validation_dates",
    }
    metric_cols = [c for c in fold_metrics.columns if c not in metadata_cols]

    summary = fold_metrics.groupby("model")[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    fold_metrics.to_csv(
        output_dir / "walk_forward_fold_metrics.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "walk_forward_summary.csv",
        index=False,
    )

    with open(output_dir / "walk_forward_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "input": str(input_path),
                "n_rows": int(len(feat)),
                "n_features": int(len(feature_cols)),
                "n_folds": len(folds),
                "initial_train_dates": initial_train_dates,
                "validation_dates": validation_dates,
                "step_dates": step_dates,
                "gap_dates": gap_dates,
                "train_window_dates": train_window_dates,
                "models": model_names,
            },
            f,
            indent=2,
        )

    print(fold_metrics.to_string(index=False))
    print("\nAggregate walk-forward results:")
    print(summary.to_string(index=False))
    print(f"\nArtifacts written to: {output_dir}")

    return fold_metrics, summary


def main() -> None:
    p = argparse.ArgumentParser(
        description="Expanding- or rolling-window validation for closing-auction models"
    )
    p.add_argument("--input", default="data/train.csv")
    p.add_argument("--output-dir", default="results/walk_forward")
    p.add_argument("--initial-train-dates", type=int, default=250)
    p.add_argument("--validation-dates", type=int, default=50)
    p.add_argument("--step-dates", type=int, default=None)
    p.add_argument("--gap-dates", type=int, default=0)
    p.add_argument("--train-window-dates", type=int, default=None)
    p.add_argument("--max-folds", type=int, default=None)
    p.add_argument(
        "--models",
        default="ridge,hist_gbdt,xgboost",
        help="Comma-separated model names",
    )
    p.add_argument("--max-rows", type=int, default=None)
    args = p.parse_args()

    run_walk_forward(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        initial_train_dates=args.initial_train_dates,
        validation_dates=args.validation_dates,
        step_dates=args.step_dates,
        gap_dates=args.gap_dates,
        train_window_dates=args.train_window_dates,
        max_folds=args.max_folds,
        model_names=[m.strip() for m in args.models.split(",") if m.strip()],
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
