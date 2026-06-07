from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend_handoff.features.engineered_features import add_engineered_features
from scoring.dimension_scores import dimension_scores_frame
from scoring.pseudo_labels import add_pseudo_health_targets


RANDOM_STATE = 42
EPS = 1e-6


@dataclass
class ChallengerTrainingResult:
    trained: bool
    run_id: str
    challenger_path: str
    base_model_path: str
    meta_model_path: str
    base_features_path: str
    meta_features_path: str
    metadata_path: str
    metrics_path: str
    training_predictions_path: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(project_root: str | Path, path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(project_root) / path


def safe_logit(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(probabilities, EPS, 1 - EPS)
    return np.log(probabilities / (1 - probabilities))


def build_meta_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def prepare_training_dataset(
    dataset_path: str | Path,
    base_features: list[str],
) -> pd.DataFrame:
    raw = pd.read_csv(dataset_path)

    if "new_label" in raw.columns and "oss_health_score" in raw.columns:
        prepared = raw.copy()
    else:
        prepared = add_engineered_features(raw)
        prepared, _dimension_cols = add_pseudo_health_targets(prepared)

    missing = [feature for feature in base_features if feature not in prepared.columns]
    if missing:
        raise ValueError(f"Training data is missing base features: {missing[:20]}")

    if "new_label" not in prepared.columns:
        raise ValueError("Training data is missing new_label")

    if "oss_health_score" not in prepared.columns:
        raise ValueError("Training data is missing oss_health_score")

    for feature in base_features:
        prepared[feature] = pd.to_numeric(prepared[feature], errors="coerce")

    prepared["new_label"] = prepared["new_label"].astype(int)
    prepared["oss_health_score"] = pd.to_numeric(
        prepared["oss_health_score"],
        errors="coerce",
    )

    prepared = prepared.dropna(subset=["new_label", "oss_health_score"]).reset_index(
        drop=True
    )

    return prepared


def calculate_score_separation(
    y_label: pd.Series,
    predicted_score: np.ndarray,
) -> float:
    healthy_scores = predicted_score[y_label.to_numpy() == 1]
    unhealthy_scores = predicted_score[y_label.to_numpy() == 0]

    if len(healthy_scores) == 0 or len(unhealthy_scores) == 0:
        return 0.0

    return float(healthy_scores.mean() - unhealthy_scores.mean())


def train_challenger_model(
    project_root: str,
    run_id: str,
    training_dataset_path: str,
    base_model_path: str,
    base_features_path: str,
    metadata_path: str,
    challenger_root: str,
) -> ChallengerTrainingResult:
    project_root_path = Path(project_root)

    training_dataset = resolve_path(project_root_path, training_dataset_path)
    source_base_model_path = resolve_path(project_root_path, base_model_path)
    source_base_features_path = resolve_path(project_root_path, base_features_path)
    source_metadata_path = resolve_path(project_root_path, metadata_path)

    base_features = load_json(source_base_features_path)
    metadata = load_json(source_metadata_path)

    challenger_path = resolve_path(project_root_path, challenger_root) / run_id
    challenger_path.mkdir(parents=True, exist_ok=True)

    df = prepare_training_dataset(
        dataset_path=training_dataset,
        base_features=base_features,
    )

    X = df[base_features].copy()
    y_label = df["new_label"].astype(int)
    y_score = df["oss_health_score"].astype(float)

    base_template = joblib.load(source_base_model_path)
    base_model = clone(base_template)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_splits = list(cv.split(X, y_label))

    base_oof_probability = cross_val_predict(
        base_template,
        X,
        y_label,
        cv=cv_splits,
        method="predict_proba",
    )[:, 1]
    base_oof_label = (base_oof_probability >= 0.5).astype(int)

    base_metrics = {
        "accuracy": float(accuracy_score(y_label, base_oof_label)),
        "precision": float(
            precision_score(y_label, base_oof_label, zero_division=0)
        ),
        "recall": float(recall_score(y_label, base_oof_label, zero_division=0)),
        "f1": float(f1_score(y_label, base_oof_label, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_label, base_oof_probability)),
    }

    dimension_df = dimension_scores_frame(features_df=X, reference_df=X)
    meta_dimension_cols = list(dimension_df.columns)
    meta_features = ["raw_model_logit", *meta_dimension_cols]

    meta_X = pd.concat(
        [
            pd.Series(
                safe_logit(base_oof_probability),
                index=X.index,
                name="raw_model_logit",
            ),
            dimension_df.reindex(X.index),
        ],
        axis=1,
    )

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

    score_separation = calculate_score_separation(y_label, meta_oof_score)

    base_model.fit(X, y_label)
    meta_model.fit(meta_X[meta_features], y_score)

    output_base_model_path = challenger_path / "oss_health_best_model.joblib"
    output_meta_model_path = challenger_path / "oss_health_meta_model.joblib"
    output_base_features_path = challenger_path / "oss_health_best_features.json"
    output_meta_features_path = challenger_path / "oss_health_meta_features.json"
    output_metadata_path = challenger_path / "oss_health_model_metadata.json"
    output_metrics_path = challenger_path / "metrics.json"
    output_predictions_path = challenger_path / "meta_model_training_predictions.csv"

    joblib.dump(base_model, output_base_model_path)
    joblib.dump(meta_model, output_meta_model_path)

    save_json(base_features, output_base_features_path)
    save_json(meta_features, output_meta_features_path)

    updated_metadata = dict(metadata)
    updated_metadata.update(
        {
            "model_version": run_id,
            "updated_at": utc_now(),
            "update_type": "parameter_refit_from_existing_best_model",
            "best_model_name": metadata.get("best_model_name", "LogisticRegression"),
            "target_col": "new_label",
            "num_features": len(base_features),
            "features": base_features,
            "overall_score_method": "meta_model_regression_score",
            "healthy_probability_method": "base_model_predict_proba",
            "meta_model_name": "Ridge",
            "meta_model_target_col": "oss_health_score",
            "meta_model_features": meta_features,
            "challenger_base_cv_metrics": base_metrics,
            "challenger_meta_model_metrics": meta_metrics,
            "score_separation": score_separation,
        }
    )
    save_json(updated_metadata, output_metadata_path)

    metrics = {
        "model_version": run_id,
        "base_roc_auc": base_metrics["roc_auc"],
        "base_f1": base_metrics["f1"],
        "base_accuracy": base_metrics["accuracy"],
        "base_precision": base_metrics["precision"],
        "base_recall": base_metrics["recall"],
        "meta_mae": meta_metrics["mae"],
        "meta_rmse": meta_metrics["rmse"],
        "meta_r2": meta_metrics["r2"],
        "score_separation": score_separation,
        "trained_at": utc_now(),
    }
    save_json(metrics, output_metrics_path)

    predictions = pd.DataFrame(
        {
            "new_label": y_label,
            "oss_health_score": y_score,
            "base_oof_probability": base_oof_probability,
            "meta_oof_score": meta_oof_score,
            "base_oof_score": base_oof_probability * 100,
            "meta_oof_display_score": meta_oof_score * 100,
        }
    )
    predictions = pd.concat([predictions, meta_X[meta_features]], axis=1)
    predictions.to_csv(output_predictions_path, index=False)

    return ChallengerTrainingResult(
        trained=True,
        run_id=run_id,
        challenger_path=str(challenger_path),
        base_model_path=str(output_base_model_path),
        meta_model_path=str(output_meta_model_path),
        base_features_path=str(output_base_features_path),
        meta_features_path=str(output_meta_features_path),
        metadata_path=str(output_metadata_path),
        metrics_path=str(output_metrics_path),
        training_predictions_path=str(output_predictions_path),
    )