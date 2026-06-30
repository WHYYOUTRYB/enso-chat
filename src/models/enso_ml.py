from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model_suite(random_state: int = 42) -> dict[str, object]:
    return {
        "linear_ridge": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=120,
            max_depth=8,
            min_samples_leaf=3,
            random_state=random_state,
        ),
    }


def train_and_predict_for_lead(
    models: dict[str, object],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    lead: int,
) -> dict[str, np.ndarray]:
    target_col = f"target_lead_{lead}"
    if target_col not in train_df.columns or target_col not in test_df.columns:
        raise ValueError(f"Target column not found: {target_col}")

    x_train = train_df[feature_cols]
    y_train = train_df[target_col]
    x_test = test_df[feature_cols]

    predictions: dict[str, np.ndarray] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions[name] = model.predict(x_test)
    return predictions


def fit_models_for_latest_forecast(
    models: dict[str, object],
    table: pd.DataFrame,
    feature_cols: list[str],
    lead: int,
) -> dict[str, float]:
    target_col = f"target_lead_{lead}"
    latest_features = table.iloc[[-1]][feature_cols]
    forecasts: dict[str, float] = {}

    for name, model in models.items():
        model.fit(table[feature_cols], table[target_col])
        forecasts[name] = float(model.predict(latest_features)[0])
    return forecasts
