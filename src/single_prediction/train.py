"""Train four candidate classifiers, compare validation metrics and save the best."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import __version__
from .config import METADATA_PATH, MODEL_PATH, PROCESSED_PATH
from .features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )


def candidate_models(random_state: int) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=500, class_weight="balanced", random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=80,
            max_depth=16,
            max_samples=0.35,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=60,
            learning_rate=0.08,
            max_depth=2,
            subsample=0.35,
            max_features="sqrt",
            random_state=random_state,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=160,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
            verbosity=-1,
        ),
    }


def train(
    input_path: Path = PROCESSED_PATH,
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
    max_rows: int | None = None,
    random_state: int = 42,
) -> dict[str, object]:
    frame = pd.read_csv(input_path, nrows=max_rows)
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]).difference(frame.columns)
    if missing:
        raise ValueError(f"训练表缺少字段: {', '.join(sorted(missing))}")
    if len(frame) < 100:
        raise ValueError("训练样本过少；至少需要 100 行")

    x = frame[FEATURE_COLUMNS]
    y = frame[TARGET_COLUMN].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=random_state
    )

    results: dict[str, dict[str, float]] = {}
    fitted: dict[str, Pipeline] = {}
    for name, estimator in candidate_models(random_state).items():
        pipeline = Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        probability = pipeline.predict_proba(x_test)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        results[name] = {
            "roc_auc": round(float(roc_auc_score(y_test, probability)), 6),
            "log_loss": round(float(log_loss(y_test, probability)), 6),
            "accuracy": round(float(accuracy_score(y_test, prediction)), 6),
            "f1": round(float(f1_score(y_test, prediction)), 6),
        }
        fitted[name] = pipeline
        print(f"{name}: {results[name]}", flush=True)

    best_name = max(results, key=lambda name: results[name]["roc_auc"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best_name], model_path)
    metadata: dict[str, object] = {
        "model_name": best_name,
        "model_version": __version__,
        "trained_at": datetime.now(UTC).isoformat(),
        "training_rows": int(len(x_train)),
        "validation_rows": int(len(x_test)),
        "positive_rate": round(float(y.mean()), 6),
        "target_definition": "liked OR shared OR commented OR followed_creator OR replayed",
        "selection_metric": "roc_auc",
        "features": FEATURE_COLUMNS,
        "metrics": results,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"最佳模型: {best_name}; 已保存到 {model_path}", flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="训练并比较四种互动预测模型")
    parser.add_argument("--input", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--model-output", type=Path, default=MODEL_PATH)
    parser.add_argument("--metadata-output", type=Path, default=METADATA_PATH)
    parser.add_argument("--max-rows", type=int, default=None, help="仅用于快速烟雾验证")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    train(args.input, args.model_output, args.metadata_output, args.max_rows, args.random_state)


if __name__ == "__main__":
    main()
