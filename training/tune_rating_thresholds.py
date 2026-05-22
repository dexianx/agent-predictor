from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune integer-rating thresholds for CatBoost predictions.")
    parser.add_argument("--input", default="data/rating_train.parquet")
    parser.add_argument("--model-path", default="models/rating_head.cbm")
    parser.add_argument("--schema-path", default="models/rating_feature_schema.json")
    parser.add_argument("--output", default="models/rating_thresholds.json")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    best = tune_thresholds(
        input_path=args.input,
        model_path=args.model_path,
        schema_path=args.schema_path,
        output_path=args.output,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(json.dumps(best, indent=2))


def tune_thresholds(
    *,
    input_path: str | Path,
    model_path: str | Path,
    schema_path: str | Path,
    output_path: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    import numpy as np
    import pandas as pd
    from catboost import CatBoostRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    df = pd.read_parquet(input_path)
    X = df[schema["feature_cols"]].copy()
    y = df[schema["target_col"]].astype(float)

    for col in schema["feature_cols"]:
        if col in schema["categorical_features"]:
            X[col] = X[col].fillna("unknown").astype(str)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            fill_value = schema.get("numeric_fill_values", {}).get(col, 0.0)
            X[col] = X[col].fillna(fill_value)

    stratify = _stratify_labels(y)
    _, X_val, _, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    model = CatBoostRegressor()
    model.load_model(model_path)
    pred_float = np.clip(model.predict(X_val), 1.0, 5.0)

    best = None
    for t12 in np.arange(1.30, 1.81, 0.05):
        for t23 in np.arange(2.30, 2.81, 0.05):
            for t34 in np.arange(3.20, 3.81, 0.05):
                for t45 in np.arange(4.10, 4.71, 0.05):
                    thresholds = (float(t12), float(t23), float(t34), float(t45))
                    if not (thresholds[0] < thresholds[1] < thresholds[2] < thresholds[3]):
                        continue
                    rounded = apply_thresholds(pred_float, thresholds)
                    rmse = float(mean_squared_error(y_val, rounded) ** 0.5)
                    if best is None or rmse < best["rmse"]:
                        best = {"thresholds": list(thresholds), "rmse": rmse}

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(best, indent=2), encoding="utf-8")
    return best or {"thresholds": [1.5, 2.5, 3.5, 4.5], "rmse": None}


def apply_thresholds(pred, thresholds):
    import numpy as np

    t12, t23, t34, t45 = thresholds
    return np.select(
        [pred < t12, pred < t23, pred < t34, pred < t45],
        [1, 2, 3, 4],
        default=5,
    )


def _stratify_labels(y):
    labels = y.round().astype(int)
    counts = labels.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        return None
    return labels


if __name__ == "__main__":
    main()
