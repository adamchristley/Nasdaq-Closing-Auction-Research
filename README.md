# Nasdaq Closing Auction Research

Leakage-aware quantitative research pipeline for the **Optiver - Trading at the Close** dataset, which contains historical data from the final ten minutes of Nasdaq's daily closing auction.

The research question is simple: **which observable order-book and auction features contain short-horizon information about a stock's future move relative to the market?**

This repository is intentionally structured like a small quant-research project rather than a leaderboard-only notebook. It emphasizes chronological validation, interpretable baselines, feature ablations, cross-sectional information coefficients, and a transaction-cost-sensitive signal diagnostic.

## Research workflow

1. **Feature engineering** from order-book and closing-auction state
   - bid/ask spread and book imbalance
   - signed auction imbalance and matched/unmatched liquidity
   - microprice and reference/near/far-price dislocations
   - strictly lagged WAP, spread, and imbalance changes
   - contemporaneous cross-sectional z-scores and ranks
2. **Leakage-aware validation**
   - complete `date_id` blocks are held out
   - no random row shuffling
   - target is never used as a feature
   - lagged features are constructed only within each stock/day
3. **Model comparison**
   - zero baseline
   - ridge regression
   - histogram gradient boosting
   - XGBoost
4. **Research diagnostics**
   - MAE and RMSE
   - Pearson and Spearman IC
   - cross-sectional Spearman IC by timestamp
   - directional accuracy
   - top/bottom-decile long-short spread diagnostic with an explicit cost assumption

## Dataset

The project expects the official Kaggle competition file at:

```text
data/train.csv
```

Competition: **Optiver - Trading at the Close**  
<https://www.kaggle.com/competitions/optiver-trading-at-the-close>

The training data includes `stock_id`, `date_id`, `seconds_in_bucket`, auction imbalance fields, top-of-book prices/sizes, WAP, and a target representing the stock's 60-second future WAP move relative to a synthetic Nasdaq index, in basis points.

The raw competition data is **not committed to this repository**. Accept the competition rules and download it from Kaggle, then place `train.csv` in `data/`.

With the Kaggle CLI configured:

```bash
kaggle competitions download -c optiver-trading-at-the-close
unzip optiver-trading-at-the-close.zip -d data/
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pytest -q
python scripts/run_research.py --input data/train.csv --output-dir results
```

### Synthetic smoke test

A deterministic synthetic generator is included solely to verify the pipeline end-to-end before downloading the competition data:

```bash
python scripts/make_demo_data.py --output data/demo_train.csv
python scripts/run_research.py --input data/demo_train.csv --output-dir results/demo
```

**Synthetic metrics are not market results and should never be reported as trading performance.**

## Output

A completed run writes:

- `metrics.csv` - validation comparison plus untouched test metrics
- `test_predictions.csv` - held-out predictions
- `signal_diagnostic.csv` - cross-sectional top/bottom-tail spread series
- `run_summary.json` - split sizes, selected model, feature list, and test metrics

## Methodological notes

### Why split by date?
Rows from the same closing-auction session are highly related. Random row splits can let the model learn patterns from the same day on both sides of the split and make out-of-sample performance look stronger than it really is. This project holds out complete future dates.

### Why use simple baselines?
A nonlinear model is only interesting if it beats a transparent baseline out of sample. Ridge regression and a zero prediction make it easier to tell whether feature engineering is adding signal or the model is simply adding complexity.

### Why report IC as well as MAE?
MAE measures forecast error, while information coefficient measures whether predictions correctly rank future relative moves. In trading research, a weak point forecast can still contain useful ranking information, so both views matter.

### What the long-short diagnostic is not
The target is already a relative future move in basis points. The top/bottom-tail calculation is useful for checking monotonicity and cost sensitivity, but it is **not a production execution backtest**. It does not model queue position, fills, impact, exchange fees, auction-specific constraints, or latency.

## Next experiments

- walk-forward retraining rather than a single chronological split
- explicit feature-family ablations
- stock embeddings / stock-specific effects
- regime analysis by volatility, spread, and auction progress
- LightGBM / CatBoost comparison
- temporal sequence models after a strong tabular baseline is established
- reconstructed synthetic-index features using only information available at prediction time

## License

Code in this repository is provided for educational and research purposes. The Optiver competition data remains subject to Kaggle's competition rules and is not redistributed here.
