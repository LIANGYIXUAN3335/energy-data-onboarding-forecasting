from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


@dataclass(frozen=True)
class ModelConfig:
    seed: int
    max_iter: int
    learning_rate: float
    max_leaf_nodes: int
    l2_regularization: float


def build_hist_gradient_boosting(config: ModelConfig) -> Pipeline:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
        max_leaf_nodes=config.max_leaf_nodes,
        l2_regularization=config.l2_regularization,
        random_state=config.seed,
        early_stopping=False,
    )
    return Pipeline([("preprocess", preprocess), ("model", regressor)])


def fit_and_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: ModelConfig,
) -> np.ndarray:
    usable = train[train["load"].notna()].copy()
    if usable.empty:
        raise ValueError("No non-null training targets remain after data-quality processing")
    estimator = build_hist_gradient_boosting(config)
    estimator.fit(usable[NUMERIC_FEATURES + CATEGORICAL_FEATURES], usable["load"])
    return estimator.predict(test[NUMERIC_FEATURES + CATEGORICAL_FEATURES])

