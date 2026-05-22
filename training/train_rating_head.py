from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_CATEGORICAL_FEATURES = [
    "item_category",
    "item_sub_category",
    "item_brand",
    "item_store",
    "item_platform",
    "item_locale",
    "review_locale",
    "review_platform",
    "item_price_bucket",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CatBoost rating head.")
    parser.add_argument("--input", default="data/rating_train.parquet")
    parser.add_argument("--model-out", default="models/rating_head.cbm")
    parser.add_argument("--schema-out", default="models/rating_feature_schema.json")
    parser.add_argument("--metrics-out", default="models/rating_metrics.json")
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--exclude-features-containing",
        default="",
        help="Comma-separated substrings; matching feature columns are dropped before training.",
    )
    parser.add_argument(
        "--exclude-feature-names",
        default="",
        help="Comma-separated exact feature column names to drop before training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_model(
        input_path=args.input,
        model_out=args.model_out,
        schema_out=args.schema_out,
        metrics_out=args.metrics_out,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        test_size=args.test_size,
        random_state=args.random_state,
        exclude_features_containing=_split_csv(args.exclude_features_containing),
        exclude_feature_names=_split_csv(args.exclude_feature_names),
    )
    print(json.dumps(metrics, indent=2))


def train_model(
    *,
    input_path: str | Path,
    model_out: str | Path,
    schema_out: str | Path,
    metrics_out: str | Path,
    iterations: int = 1200,
    learning_rate: float = 0.035,
    depth: int = 6,
    test_size: float = 0.2,
    random_state: int = 42,
    exclude_features_containing: list[str] | None = None,
    exclude_feature_names: list[str] | None = None,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    from catboost import CatBoostRegressor, Pool
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import train_test_split

    df = pd.read_parquet(input_path)
    if "target_rating" not in df.columns:
        raise ValueError("Dataset must contain target_rating")
    if len(df) < 5:
        raise ValueError("Need at least 5 rows to train a validation-backed rating head")

    target_col = "target_rating"
    id_cols = ["review_id", "user_id", "item_id", "timestamp"]
    drop_cols = {target_col, *id_cols}
    feature_cols = [col for col in df.columns if col not in drop_cols]
    excluded_features = [
        col for col in feature_cols
        if col in set(exclude_feature_names or [])
        or any(token and token in col for token in (exclude_features_containing or []))
    ]
    if excluded_features:
        feature_cols = [col for col in feature_cols if col not in set(excluded_features)]
    categorical_features = [col for col in DEFAULT_CATEGORICAL_FEATURES if col in feature_cols]

    X = df[feature_cols].copy()
    y = df[target_col].astype(float)
    numeric_fill_values: dict[str, float] = {}
    for col in feature_cols:
        if col in categorical_features:
            X[col] = X[col].fillna("unknown").astype(str)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            median = X[col].median()
            fill_value = float(median) if not pd.isna(median) else 0.0
            numeric_fill_values[col] = fill_value
            X[col] = X[col].fillna(fill_value)

    stratify = _stratify_labels(y)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    train_pool = Pool(X_train, y_train, cat_features=categorical_features)
    val_pool = Pool(X_val, y_val, cat_features=categorical_features)
    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=6,
        random_seed=random_state,
        od_type="Iter",
        od_wait=min(80, max(iterations // 5, 10)),
        verbose=100 if iterations >= 100 else False,
        allow_writing_files=False,
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    pred_float = np.clip(model.predict(val_pool), 1.0, 5.0)
    pred_round = np.clip(np.rint(pred_float), 1, 5)
    metrics = {
        "rmse_float": float(mean_squared_error(y_val, pred_float) ** 0.5),
        "mae_float": float(mean_absolute_error(y_val, pred_float)),
        "rmse_round": float(mean_squared_error(y_val, pred_round) ** 0.5),
        "mae_round": float(mean_absolute_error(y_val, pred_round)),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "best_iteration": int(model.get_best_iteration() or 0),
    }

    model_out = Path(model_out)
    schema_out = Path(schema_out)
    metrics_out = Path(metrics_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    schema_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_out)

    schema = {
        "feature_cols": feature_cols,
        "categorical_features": categorical_features,
        "numeric_fill_values": numeric_fill_values,
        "target_col": target_col,
        "model_type": "CatBoostRegressor",
        "excluded_features": excluded_features,
    }
    schema_out.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _stratify_labels(y):
    labels = y.round().astype(int)
    counts = labels.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        return None
    return labels


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


if __name__ == "__main__":
    main()
