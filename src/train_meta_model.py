from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scoring.dimension_scores import dimension_scores_frame


RANDOM_STATE = 42
EPS = 1e-6

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs" / "2_model"

BASE_MODEL_PATH = MODEL_DIR / "oss_health_best_model.joblib"
BASE_FEATURES_PATH = MODEL_DIR / "oss_health_best_features.json"
METADATA_PATH = MODEL_DIR / "oss_health_model_metadata.json"
TRAINING_DATA_PATH = OUTPUT_DIR / "final_training_dataset.csv"
SCORE_DATA_PATH = BASE_DIR / "oss_health_score_labeled_train.csv"
META_MODEL_PATH = MODEL_DIR / "oss_health_meta_model.joblib"
META_FEATURES_PATH = MODEL_DIR / "oss_health_meta_features.json"
META_TRAINING_OUTPUT_PATH = OUTPUT_DIR / "meta_model_training_predictions.csv"


def safe_logit(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(probabilities, EPS, 1 - EPS)
    return np.log(probabilities / (1 - probabilities))


def build_meta_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])


def main() -> None:
    with open(BASE_FEATURES_PATH, encoding="utf-8") as f:
        base_features = json.load(f)

    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)

    label_col = metadata.get("target_col", "new_label")
    target_col = "oss_health_score"

    training_df = pd.read_csv(TRAINING_DATA_PATH)
    score_df = pd.read_csv(SCORE_DATA_PATH)
    missing = [feature for feature in base_features if feature not in training_df.columns]
    if missing:
        raise ValueError(f"Training data is missing base features: {missing[:10]}")
    if label_col not in training_df.columns:
        raise ValueError(f"Training data is missing label column: {label_col}")
    if target_col not in score_df.columns:
        raise ValueError(f"Score data is missing target column: {target_col}")
    if len(training_df) != len(score_df):
        raise ValueError("Training data and score data must have the same row count")
    if label_col in score_df.columns:
        label_match = training_df[label_col].reset_index(drop=True).equals(
            score_df[label_col].reset_index(drop=True)
        )
        if not label_match:
            raise ValueError("Training data and score data label order do not match")

    X = training_df[base_features].copy()
    for col in base_features:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    y_label = training_df[label_col].astype(int)
    y_score = score_df[target_col].astype(float).clip(0, 1)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_splits = list(cv.split(X, y_label))
    base_model = joblib.load(BASE_MODEL_PATH)

    base_oof_probability = np.zeros(len(X), dtype=float)
    for train_idx, valid_idx in cv_splits:
        fold_model = clone(base_model)
        fold_model.fit(X.iloc[train_idx], y_label.iloc[train_idx])
        base_oof_probability[valid_idx] = fold_model.predict_proba(X.iloc[valid_idx])[:, 1]

    dimension_df = dimension_scores_frame(features_df=X, reference_df=X)
    meta_dimension_cols = list(dimension_df.columns)
    meta_features = ["raw_model_logit", *meta_dimension_cols]

    meta_X = pd.concat([
        pd.Series(safe_logit(base_oof_probability), index=X.index, name="raw_model_logit"),
        dimension_df.reindex(X.index),
    ], axis=1)

    meta_model = build_meta_model()
    meta_oof_score = cross_val_predict(
        meta_model,
        meta_X[meta_features],
        y_score,
        cv=cv_splits,
        method="predict",
    )
    meta_oof_score = np.clip(meta_oof_score, 0, 1)

    meta_metrics = {
        "mae": float(mean_absolute_error(y_score, meta_oof_score)),
        "rmse": float(mean_squared_error(y_score, meta_oof_score) ** 0.5),
        "r2": float(r2_score(y_score, meta_oof_score)),
    }

    meta_model.fit(meta_X[meta_features], y_score)

    joblib.dump(meta_model, META_MODEL_PATH)
    with open(META_FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(meta_features, f, ensure_ascii=False, indent=2)

    metadata.update({
        "overall_score_method": "meta_model_regression_score",
        "healthy_probability_method": "base_model_predict_proba",
        "meta_model_name": "Ridge",
        "meta_model_target_col": target_col,
        "meta_model_features": meta_features,
        "meta_model_metrics": meta_metrics,
    })
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    output = pd.DataFrame({
        label_col: y_label,
        target_col: y_score,
        "base_oof_probability": base_oof_probability,
        "meta_oof_score": meta_oof_score,
        "base_oof_score": base_oof_probability * 100,
        "meta_oof_display_score": meta_oof_score * 100,
    })
    output = pd.concat([output, meta_X[meta_features]], axis=1)
    output.to_csv(META_TRAINING_OUTPUT_PATH, index=False)

    print("Saved meta model:", META_MODEL_PATH)
    print("Saved meta features:", META_FEATURES_PATH)
    print("Saved metadata:", METADATA_PATH)
    print("Saved meta training predictions:", META_TRAINING_OUTPUT_PATH)
    print("Meta metrics:", meta_metrics)


if __name__ == "__main__":
    main()
