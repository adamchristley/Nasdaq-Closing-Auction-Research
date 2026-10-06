from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y = np.asarray(y_true)[mask]
    p = np.asarray(y_pred)[mask]
    if len(y) < 3:
        return {"mae":np.nan,"rmse":np.nan,"pearson_ic":np.nan,"spearman_ic":np.nan,"directional_accuracy":np.nan}
    return {"mae":float(mean_absolute_error(y,p)),"rmse":float(np.sqrt(mean_squared_error(y,p))),"pearson_ic":float(pearsonr(y,p).statistic),"spearman_ic":float(spearmanr(y,p).statistic),"directional_accuracy":float(np.mean(np.sign(y) == np.sign(p)))}


def cross_sectional_ic(frame: pd.DataFrame, pred_col: str = "prediction") -> dict[str, float]:
    values=[]
    for _, g in frame.groupby(["date_id","seconds_in_bucket"], sort=False):
        g = g[["target", pred_col]].dropna()
        if len(g) >= 5 and g["target"].nunique() > 1 and g[pred_col].nunique() > 1:
            values.append(float(g["target"].corr(g[pred_col], method="spearman")))
    if not values:
        return {"mean_xs_spearman_ic":np.nan,"median_xs_spearman_ic":np.nan,"positive_ic_share":np.nan}
    a=np.asarray(values)
    return {"mean_xs_spearman_ic":float(np.nanmean(a)),"median_xs_spearman_ic":float(np.nanmedian(a)),"positive_ic_share":float(np.mean(a > 0))}
