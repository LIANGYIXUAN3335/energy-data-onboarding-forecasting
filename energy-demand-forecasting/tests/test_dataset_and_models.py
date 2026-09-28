from __future__ import annotations

import numpy as np
import pandas as pd

from energy_forecasting.dataset import (
    choose_explicit_split_boundaries,
    choose_split_boundaries,
    partition,
)
from energy_forecasting.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from energy_forecasting.models import ModelConfig, build_hist_gradient_boosting


def test_split_is_strictly_chronological(hourly_frame: pd.DataFrame) -> None:
    boundary = choose_split_boundaries(hourly_frame, 0.6, 0.2)
    labels = partition(hourly_frame, boundary)
    assert hourly_frame.loc[labels == "train", "timestamp"].max() <= boundary.train_end
    assert hourly_frame.loc[labels == "validation", "timestamp"].min() > boundary.train_end
    assert (
        hourly_frame.loc[labels == "test", "timestamp"].min()
        > boundary.validation_end
    )


def test_explicit_calendar_split_is_preserved(hourly_frame: pd.DataFrame) -> None:
    reference = hourly_frame.copy()
    reference["timestamp"] = pd.to_datetime(reference["timestamp"], utc=True)
    boundary = choose_explicit_split_boundaries(
        reference,
        "2024-01-14T23:00:00Z",
        "2024-01-21T23:00:00Z",
    )
    assert boundary.train_end == pd.Timestamp("2024-01-14T23:00:00Z")
    assert boundary.validation_end == pd.Timestamp("2024-01-21T23:00:00Z")


def test_preprocessing_statistics_are_fit_on_training_only() -> None:
    train = pd.DataFrame({name: [1.0, 1.0, np.nan] for name in NUMERIC_FEATURES})
    test = pd.DataFrame({name: [1_000_000.0] for name in NUMERIC_FEATURES})
    for name in CATEGORICAL_FEATURES:
        train[name] = ["train_a", "train_b", None]
        test[name] = ["unseen_test_value"]
    estimator = build_hist_gradient_boosting(
        ModelConfig(
            seed=1,
            max_iter=2,
            learning_rate=0.1,
            max_leaf_nodes=3,
            l2_regularization=0.0,
        )
    )
    estimator.fit(
        train[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
        np.array([1.0, 2.0, 3.0]),
    )
    estimator.predict(test[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    imputer = estimator.named_steps["preprocess"].named_transformers_["numeric"].named_steps[
        "impute"
    ]
    assert np.all(imputer.statistics_ == 1.0)
