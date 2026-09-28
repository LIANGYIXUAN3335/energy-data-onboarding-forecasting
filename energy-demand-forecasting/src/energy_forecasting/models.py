from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


LEARNED_MODELS = ("hist_gradient_boosting", "ridge", "random_forest")
SUPPORTED_MODELS = ("seasonal_naive",) + LEARNED_MODELS


@dataclass(frozen=True)
class ModelConfig:
    seed: int
    max_iter: int
    learning_rate: float
    max_leaf_nodes: int
    l2_regularization: float
    ridge_alpha: float = 1.0
    random_forest_n_estimators: int = 200
    random_forest_min_samples_leaf: int = 5
    random_forest_max_features: float = 0.5
    random_forest_n_jobs: int = 4

    def parameters(self, model_name: str) -> dict[str, object]:
        """Hyperparameters recorded in the run manifest for one model."""

        if model_name == "hist_gradient_boosting":
            return {
                "seed": self.seed,
                "max_iter": self.max_iter,
                "learning_rate": self.learning_rate,
                "max_leaf_nodes": self.max_leaf_nodes,
                "l2_regularization": self.l2_regularization,
            }
        if model_name == "ridge":
            return {"alpha": self.ridge_alpha, "standardized_numeric_inputs": True}
        if model_name == "random_forest":
            return {
                "seed": self.seed,
                "n_estimators": self.random_forest_n_estimators,
                "min_samples_leaf": self.random_forest_min_samples_leaf,
                "max_features": self.random_forest_max_features,
            }
        if model_name == "seasonal_naive":
            return {"fallback_order_hours": [168, 24]}
        raise ValueError(f"Unknown model {model_name!r}; choose from {SUPPORTED_MODELS}")


def _preprocessor(
    numeric_features: list[str], categorical_features: list[str], *, scale: bool
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric_features),
            ("categorical", categorical, categorical_features),
        ],
        remainder="drop",
    )


def build_model(
    model_name: str,
    config: ModelConfig,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    """Build one learned model with training-only preprocessing."""

    numeric = list(numeric_features or NUMERIC_FEATURES)
    categorical = list(categorical_features or CATEGORICAL_FEATURES)
    if model_name == "hist_gradient_boosting":
        regressor = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=config.learning_rate,
            max_iter=config.max_iter,
            max_leaf_nodes=config.max_leaf_nodes,
            l2_regularization=config.l2_regularization,
            random_state=config.seed,
            early_stopping=False,
        )
        preprocess = _preprocessor(numeric, categorical, scale=False)
    elif model_name == "ridge":
        regressor = Ridge(alpha=config.ridge_alpha)
        preprocess = _preprocessor(numeric, categorical, scale=True)
    elif model_name == "random_forest":
        regressor = RandomForestRegressor(
            n_estimators=config.random_forest_n_estimators,
            min_samples_leaf=config.random_forest_min_samples_leaf,
            max_features=config.random_forest_max_features,
            random_state=config.seed,
            n_jobs=config.random_forest_n_jobs,
        )
        preprocess = _preprocessor(numeric, categorical, scale=False)
    else:
        raise ValueError(f"Unknown learned model {model_name!r}; choose from {LEARNED_MODELS}")
    return Pipeline([("preprocess", preprocess), ("model", regressor)])


def build_hist_gradient_boosting(config: ModelConfig) -> Pipeline:
    """v2 entry point kept for the committed canonical run."""

    return build_model("hist_gradient_boosting", config)


def fit_and_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: ModelConfig,
    *,
    model_name: str = "hist_gradient_boosting",
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> np.ndarray:
    usable = train[train["load"].notna()].copy()
    if usable.empty:
        raise ValueError("No non-null training targets remain after data-quality processing")
    numeric = list(numeric_features or NUMERIC_FEATURES)
    categorical = list(categorical_features or CATEGORICAL_FEATURES)
    estimator = build_model(model_name, config, numeric, categorical)
    estimator.fit(usable[numeric + categorical], usable["load"])
    return estimator.predict(test[numeric + categorical])
