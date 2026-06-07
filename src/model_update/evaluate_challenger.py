from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class EvaluationResult:
    promote_recommended: bool
    reason: str
    failed_checks: list[str]
    champion_metrics: dict[str, Any]
    challenger_metrics: dict[str, Any]
    output_report_path: str


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metric file not found: {path}")

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def extract_champion_metrics(champion_metadata: dict[str, Any]) -> dict[str, float]:
    cv = champion_metadata.get("final_cv_summary", {})
    holdout = champion_metadata.get("tuned_holdout_metrics", {})
    meta = champion_metadata.get("meta_model_metrics", {})

    return {
        "base_roc_auc": float(
            cv.get("roc_auc_mean", holdout.get("roc_auc", 0.0))
        ),
        "base_f1": float(
            cv.get("f1_mean", holdout.get("f1", 0.0))
        ),
        "base_accuracy": float(
            cv.get("accuracy_mean", holdout.get("accuracy", 0.0))
        ),
        "base_precision": float(
            cv.get("precision_mean", holdout.get("precision", 0.0))
        ),
        "base_recall": float(
            cv.get("recall_mean", holdout.get("recall", 0.0))
        ),
        "meta_mae": float(meta.get("mae", 999.0)),
        "meta_rmse": float(meta.get("rmse", 999.0)),
        "meta_r2": float(meta.get("r2", -999.0)),
        "score_separation": float(champion_metadata.get("score_separation", 0.0)),
    }


def evaluate_champion_vs_challenger(
    champion_metadata_path: str,
    challenger_metrics_path: str,
    output_report_path: str,
    min_auc_drop_tolerance: float = 0.01,
    min_f1_drop_tolerance: float = 0.02,
    max_mae_increase_tolerance: float = 0.01,
    max_rmse_increase_tolerance: float = 0.01,
) -> EvaluationResult:
    champion_metadata = load_json(champion_metadata_path)
    challenger_metrics = load_json(challenger_metrics_path)

    champion_metrics = extract_champion_metrics(champion_metadata)

    failed_checks = []

    if challenger_metrics["base_roc_auc"] < (
        champion_metrics["base_roc_auc"] - min_auc_drop_tolerance
    ):
        failed_checks.append("base_roc_auc_degraded")

    if challenger_metrics["base_f1"] < (
        champion_metrics["base_f1"] - min_f1_drop_tolerance
    ):
        failed_checks.append("base_f1_degraded")

    if challenger_metrics["meta_mae"] > (
        champion_metrics["meta_mae"] + max_mae_increase_tolerance
    ):
        failed_checks.append("meta_mae_degraded")

    if challenger_metrics["meta_rmse"] > (
        champion_metrics["meta_rmse"] + max_rmse_increase_tolerance
    ):
        failed_checks.append("meta_rmse_degraded")

    promote_recommended = len(failed_checks) == 0

    reason = (
        "challenger passed champion comparison"
        if promote_recommended
        else f"challenger failed checks: {failed_checks}"
    )

    result = EvaluationResult(
        promote_recommended=promote_recommended,
        reason=reason,
        failed_checks=failed_checks,
        champion_metrics=champion_metrics,
        challenger_metrics=challenger_metrics,
        output_report_path=output_report_path,
    )

    save_json(asdict(result), output_report_path)
    return result