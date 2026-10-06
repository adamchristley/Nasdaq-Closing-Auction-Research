from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor


@dataclass
class FittedModel:
    name: str
    model: Any


def make_models(random_state: int = 42) -> dict[str, Any]:
    return {"ridge":Pipeline([("impute",SimpleImputer(strategy="median", add_indicator=True)),("scale",StandardScaler()),("model",Ridge(alpha=10.0))]),"hist_gbdt":Pipeline([("impute",SimpleImputer(strategy="median", add_indicator=True)),("model",HistGradientBoostingRegressor(learning_rate=0.06,max_iter=250,max_leaf_nodes=31,l2_regularization=1.0,random_state=random_state))]),"xgboost":XGBRegressor(n_estimators=350,learning_rate=0.05,max_depth=6,min_child_weight=8,subsample=0.8,colsample_bytree=0.8,reg_alpha=0.05,reg_lambda=2.0,objective="reg:absoluteerror",tree_method="hist",random_state=random_state,n_jobs=-1)}


def fit_predict(model: Any, x_train, y_train, x_eval) -> np.ndarray:
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_eval), dtype=float)
