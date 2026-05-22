from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .rating_features import build_rating_features, feature_coverage


_rating_head = None


class RatingHead:
    def __init__(
        self,
        model_path: str | Path = "models/rating_head.cbm",
        schema_path: str | Path = "models/rating_feature_schema.json",
        thresholds_path: str | Path = "models/rating_thresholds.json",
    ) -> None:
        from catboost import CatBoostRegressor

        self.model = CatBoostRegressor()
        self.model.load_model(str(model_path))
        self.schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        try:
            thresholds = json.loads(Path(thresholds_path).read_text(encoding="utf-8"))["thresholds"]
        except FileNotFoundError:
            thresholds = [1.5, 2.5, 3.5, 4.5]
        self.thresholds = [float(value) for value in thresholds]

    def prepare_row(self, features: dict[str, Any]) -> pd.DataFrame:
        row = {col: features.get(col) for col in self.schema["feature_cols"]}
        df = pd.DataFrame([row])
        categorical = set(self.schema["categorical_features"])
        fill_values = self.schema.get("numeric_fill_values", {})
        for col in df.columns:
            if col in categorical:
                df[col] = df[col].fillna("unknown").astype(str)
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(float(fill_values.get(col, 0.0)))
        return df

    def predict_float(self, features: dict[str, Any]) -> float:
        pred = float(self.model.predict(self.prepare_row(features))[0])
        return float(np.clip(pred, 1.0, 5.0))

    def predict_int(self, features: dict[str, Any]) -> int:
        return apply_thresholds(self.predict_float(features), self.thresholds)

    def coverage(self, features: dict[str, Any]) -> dict[str, Any]:
        return feature_coverage(features, self.schema["feature_cols"])


def get_rating_head() -> RatingHead:
    global _rating_head
    if _rating_head is None:
        _rating_head = RatingHead(
            os.environ.get("RATING_HEAD_MODEL_PATH", "models/rating_head.cbm"),
            os.environ.get("RATING_HEAD_SCHEMA_PATH", "models/rating_feature_schema.json"),
            os.environ.get("RATING_HEAD_THRESHOLDS_PATH", "models/rating_thresholds.json"),
        )
    return _rating_head


def apply_thresholds(pred: float, thresholds: list[float]) -> int:
    t12, t23, t34, t45 = thresholds
    if pred < t12:
        return 1
    if pred < t23:
        return 2
    if pred < t34:
        return 3
    if pred < t45:
        return 4
    return 5
