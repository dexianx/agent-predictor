from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.train_rating_head import DEFAULT_CATEGORICAL_FEATURES


SUSPICIOUS_TOKENS = (
    "rating",
    "mean",
    "std",
    "low",
    "high",
    "five",
    "source_average_rating",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate CatBoost rating head for leakage risk.")
    parser.add_argument("--input", default="data/rating_train.parquet")
    parser.add_argument("--model-path", default="models/rating_head.cbm")
    parser.add_argument("--schema-path", default="models/rating_feature_schema.json")
    parser.add_argument("--output", default="models/rating_validation_report.json")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sample-rows", type=int, default=0, help="Optional cap for faster diagnostics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_rating_head(
        input_path=args.input,
        model_path=args.model_path,
        schema_path=args.schema_path,
        top_n=args.top_n,
        iterations=args.iterations,
        test_size=args.test_size,
        random_state=args.random_state,
        sample_rows=args.sample_rows or None,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def validate_rating_head(
    *,
    input_path: str | Path,
    model_path: str | Path,
    schema_path: str | Path,
    top_n: int = 50,
    iterations: int = 500,
    test_size: float = 0.2,
    random_state: int = 42,
    sample_rows: int | None = None,
) -> dict[str, Any]:
    import pandas as pd
    from catboost import CatBoostRegressor, Pool

    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    df = pd.read_parquet(input_path)
    if sample_rows and sample_rows < len(df):
        df = df.sample(n=sample_rows, random_state=random_state).reset_index(drop=True)

    model = CatBoostRegressor()
    model.load_model(model_path)
    feature_importances = _top_importances(model, schema["feature_cols"], top_n)

    prepared = prepare_features(df, schema["feature_cols"], schema["categorical_features"], schema.get("numeric_fill_values", {}))
    y = df[schema["target_col"]].astype(float)

    report = {
        "dataset": {
            "rows": int(len(df)),
            "target_min": float(y.min()),
            "target_max": float(y.max()),
            "target_mean": float(y.mean()),
        },
        "top_feature_importances": feature_importances,
        "target_shuffle_random_split": train_eval(
            prepared.X,
            y.sample(frac=1.0, random_state=random_state).reset_index(drop=True),
            prepared.categorical_features,
            split_kind="random",
            iterations=iterations,
            test_size=test_size,
            random_state=random_state,
        ),
        "grouped_user_id": train_eval(
            prepared.X,
            y,
            prepared.categorical_features,
            split_kind="group",
            groups=df["user_id"],
            iterations=iterations,
            test_size=test_size,
            random_state=random_state,
        ),
        "grouped_item_id": train_eval(
            prepared.X,
            y,
            prepared.categorical_features,
            split_kind="group",
            groups=df["item_id"],
            iterations=iterations,
            test_size=test_size,
            random_state=random_state,
        ),
    }

    keep_cols = [col for col in schema["feature_cols"] if not _is_suspicious(col)]
    stripped = prepare_features(df, keep_cols, schema["categorical_features"], schema.get("numeric_fill_values", {}))
    report["drop_suspicious_rating_stat_columns"] = {
        "dropped_count": len(schema["feature_cols"]) - len(keep_cols),
        "kept_count": len(keep_cols),
        **train_eval(
            stripped.X,
            y,
            stripped.categorical_features,
            split_kind="random",
            iterations=iterations,
            test_size=test_size,
            random_state=random_state,
        ),
    }
    return report


class Prepared:
    def __init__(self, X, categorical_features: list[str]) -> None:
        self.X = X
        self.categorical_features = categorical_features


def prepare_features(df, feature_cols: list[str], categorical_features: list[str], fill_values: dict[str, float]) -> Prepared:
    import pandas as pd

    X = df[feature_cols].copy()
    cats = [col for col in categorical_features if col in feature_cols]
    for col in feature_cols:
        if col in cats:
            X[col] = X[col].fillna("unknown").astype(str)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            fill_value = fill_values.get(col)
            if fill_value is None:
                median = X[col].median()
                fill_value = float(median) if not pd.isna(median) else 0.0
            X[col] = X[col].fillna(fill_value)
    return Prepared(X, cats)


def train_eval(
    X,
    y,
    categorical_features: list[str],
    *,
    split_kind: str,
    groups=None,
    iterations: int,
    test_size: float,
    random_state: int,
) -> dict[str, Any]:
    import numpy as np
    from catboost import CatBoostRegressor, Pool
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    if split_kind == "group":
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, val_idx = next(splitter.split(X, y, groups=groups))
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    else:
        labels = y.round().astype(int)
        counts = labels.value_counts()
        stratify = labels if len(counts) >= 2 and counts.min() >= 2 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=iterations,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=6,
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
    )
    train_pool = Pool(X_train, y_train, cat_features=categorical_features)
    val_pool = Pool(X_val, y_val, cat_features=categorical_features)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    pred_float = np.clip(model.predict(val_pool), 1.0, 5.0)
    pred_round = np.clip(np.rint(pred_float), 1, 5)
    return {
        "rmse_float": float(mean_squared_error(y_val, pred_float) ** 0.5),
        "mae_float": float(mean_absolute_error(y_val, pred_float)),
        "rmse_round": float(mean_squared_error(y_val, pred_round) ** 0.5),
        "mae_round": float(mean_absolute_error(y_val, pred_round)),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "best_iteration": int(model.get_best_iteration() or 0),
    }


def _top_importances(model, feature_cols: list[str], top_n: int) -> list[dict[str, Any]]:
    values = model.get_feature_importance()
    rows = [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in zip(feature_cols, values)
    ]
    return sorted(rows, key=lambda row: row["importance"], reverse=True)[:top_n]


def _is_suspicious(feature: str) -> bool:
    lowered = feature.lower()
    return any(token in lowered for token in SUSPICIOUS_TOKENS)


if __name__ == "__main__":
    main()
