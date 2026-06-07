from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from model_update.backend_alert import send_backend_model_promoted_alert
from model_update.evaluate_challenger import evaluate_champion_vs_challenger
from model_update.promote_model import promote_challenger_to_champion
from model_update.train_challenger import train_challenger_model
from monitoring.drift_detector import detect_feature_drift
from monitoring.drift_thresholds import (
    append_drift_history,
    calculate_dynamic_thresholds,
    load_drift_history,
    save_thresholds,
)
from reference.candidate_search import build_candidate_pool
from reference.reference_manager import run_reference_update


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def to_dict(value: Any) -> Any:
    if value is None:
        return None
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: to_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_dict(item) for item in value]
    return value


def resolve_path(project_root: str | Path, path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(project_root) / path


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def ensure_candidate_pool_if_needed(args: argparse.Namespace) -> None:
    build_candidate_pool(
        active_reference_path=args.active_reference,
        scoring_reference_path=args.scoring_reference,
        candidate_pool_path=args.candidate_pool,
        output_path=args.candidate_pool,
        target_size=args.candidate_target_size,
        candidate_buffer_ratio=args.candidate_buffer_ratio,
        min_candidate_size=args.min_candidate_size,
        languages=args.languages,
        max_search_candidates=args.max_search_candidates,
        per_page=args.search_per_page,
        pages=args.search_pages,
    )


def run_reference_update_with_optional_replenish(args: argparse.Namespace):
    try:
        return run_reference_update(
            active_reference_path=args.active_reference,
            candidate_pool_path=args.candidate_pool,
            reference_feature_path=args.scoring_reference,
            output_reference_path=args.output_reference,
            output_candidate_pool_path=args.output_candidate_pool,
            output_report_path=args.reference_report,
        )

    except ValueError as exc:
        message = str(exc).lower()
        candidate_issue = (
            "candidate" in message
            or "replace" in message
            or "available" in message
        )

        if not args.replenish_candidates_on_demand or not candidate_issue:
            raise

        ensure_candidate_pool_if_needed(args)

        return run_reference_update(
            active_reference_path=args.active_reference,
            candidate_pool_path=args.candidate_pool,
            reference_feature_path=args.scoring_reference,
            output_reference_path=args.output_reference,
            output_candidate_pool_path=args.output_candidate_pool,
            output_report_path=args.reference_report,
        )


def run_feature_drift_check(args: argparse.Namespace, current_run_id: str):
    reference_df = pd.read_csv(args.drift_reference_features)
    current_df = pd.read_csv(args.current_batch_features)
    feature_columns = load_json(args.model_features)

    history = load_drift_history(args.drift_history)
    thresholds = calculate_dynamic_thresholds(
        history=history,
        min_history_size=args.min_drift_history_size,
        quantile=args.drift_threshold_quantile,
    )

    save_thresholds(thresholds, args.drift_thresholds)

    drift_report = detect_feature_drift(
        reference=reference_df,
        current=current_df,
        feature_columns=feature_columns,
        overall_threshold=thresholds.overall_threshold,
        drifted_ratio_threshold=thresholds.drifted_ratio_threshold,
        high_count_threshold=thresholds.high_count_threshold,
    )

    save_json(asdict(drift_report), args.drift_report)

    append_drift_history(
        args.drift_history,
        {
            "run_id": current_run_id,
            "overall_drift_score": drift_report.overall_drift_score,
            "drifted_feature_ratio": drift_report.drifted_feature_ratio,
            "high_drift_feature_count": drift_report.high_drift_feature_count,
            "retrain_required": drift_report.retrain_required,
            "threshold_source": thresholds.source,
            "created_at": utc_now(),
        },
    )

    return drift_report, thresholds


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    current_run_id = make_run_id()

    reference_result = run_reference_update_with_optional_replenish(args)
    reference_changed = reference_result.reference_changed

    drift_report = None
    drift_thresholds = None

    if reference_changed:
        retrain_required = True
        trigger_reason = "reference_changed"
        training_dataset_path = args.output_reference
    else:
        drift_report, drift_thresholds = run_feature_drift_check(
            args=args,
            current_run_id=current_run_id,
        )
        retrain_required = drift_report.retrain_required
        trigger_reason = "feature_drift" if retrain_required else "none"
        training_dataset_path = args.current_batch_features

    challenger_training = None
    evaluation_result = None
    promotion_result = None
    backend_alert_result = None

    if retrain_required:
        challenger_training = train_challenger_model(
            project_root=args.project_root,
            run_id=current_run_id,
            training_dataset_path=training_dataset_path,
            base_model_path=args.base_model_path,
            base_features_path=args.model_features,
            metadata_path=args.model_metadata,
            challenger_root=args.challenger_root,
        )

        evaluation_result = evaluate_champion_vs_challenger(
            champion_metadata_path=args.model_metadata,
            challenger_metrics_path=challenger_training.metrics_path,
            output_report_path=args.evaluation_report,
            min_auc_drop_tolerance=args.min_auc_drop_tolerance,
            min_f1_drop_tolerance=args.min_f1_drop_tolerance,
            max_mae_increase_tolerance=args.max_mae_increase_tolerance,
            max_rmse_increase_tolerance=args.max_rmse_increase_tolerance,
        )

        if evaluation_result.promote_recommended:
            promotion_result = promote_challenger_to_champion(
                project_root=args.project_root,
                challenger_path=challenger_training.challenger_path,
                champion_path=args.champion_path,
                archive_root=args.archive_root,
                model_version=current_run_id,
                trigger_reason=trigger_reason,
                evaluation_report_path=args.evaluation_report,
                backend_handoff_models_path=args.backend_handoff_models_path,
                data_models_path=args.data_models_path,
            )
            backend_alert_result = send_backend_model_promoted_alert(
                webhook_url=args.backend_webhook_url,
                model_version=promotion_result.model_version,
                trigger_reason=trigger_reason,
                champion_path=promotion_result.champion_path,
                metadata_path=promotion_result.metadata_path,
                backend_handoff_models_path=promotion_result.backend_handoff_models_path,
                payload_path=args.backend_alert_payload,
            )

    pipeline_report = {
        "run_id": current_run_id,
        "created_at": utc_now(),
        "reference_changed": reference_changed,
        "retrain_required": retrain_required,
        "trigger_reason": trigger_reason,
        "model_promoted": promotion_result is not None,
        "reference_update": to_dict(reference_result),
        "drift_report": to_dict(drift_report),
        "drift_thresholds": to_dict(drift_thresholds),
        "challenger_training": to_dict(challenger_training),
        "evaluation": to_dict(evaluation_result),
        "promotion": to_dict(promotion_result),
        "backend_alert": to_dict(backend_alert_result),
    }

    save_json(pipeline_report, args.pipeline_report)
    return pipeline_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OSS Health reference, drift, challenger, promotion pipeline."
    )

    parser.add_argument(
        "--project-root",
        default="/Users/carolyn/Desktop/opensource/data/oss-health-data",
    )

    parser.add_argument(
        "--active-reference",
        default="src/reference_store/active/reference_latest.csv",
    )

    parser.add_argument(
        "--scoring-reference",
        default="src/reference_store/active/reference_latest.csv",
    )

    parser.add_argument(
        "--candidate-pool",
        default="src/reference_store/candidates/candidate_pool.csv",
    )

    parser.add_argument(
        "--output-reference",
        default="src/reference_store/active/reference_latest.csv",
    )

    parser.add_argument(
        "--output-candidate-pool",
        default="src/reference_store/candidates/candidate_pool.csv",
    )

    parser.add_argument(
        "--reference-report",
        default="src/reference_store/reports/reference_update_report.json",
    )

    parser.add_argument(
        "--drift-reference-features",
        default="src/outputs/2_model/final_training_dataset.csv",
    )
    parser.add_argument(
        "--current-batch-features",
        default="src/outputs/model_update/current_batch_features.csv",
    )
    parser.add_argument(
        "--model-features",
        default="src/models/oss_health_best_features.json",
    )
    parser.add_argument(
        "--base-model-path",
        default="src/models/oss_health_best_model.joblib",
    )
    parser.add_argument(
        "--model-metadata",
        default="src/models/oss_health_model_metadata.json",
    )

    parser.add_argument(
        "--drift-history",
        default="src/monitoring_store/drift_history.csv",
    )
    parser.add_argument(
        "--drift-thresholds",
        default="src/monitoring_store/drift_thresholds.json",
    )
    parser.add_argument(
        "--drift-report",
        default="src/monitoring_store/reports/drift_report_latest.json",
    )

    parser.add_argument(
        "--challenger-root",
        default="src/model_registry/challenger",
    )
    parser.add_argument(
        "--champion-path",
        default="src/model_registry/champion",
    )
    parser.add_argument(
        "--archive-root",
        default="src/model_registry/archive",
    )
    parser.add_argument(
        "--evaluation-report",
        default="src/outputs/model_update/evaluation_report.json",
    )
    parser.add_argument(
        "--pipeline-report",
        default="src/outputs/model_update/pipeline_report.json",
    )

    parser.add_argument(
        "--backend-handoff-models-path",
        default="src/backend_handoff/models",
    )

    parser.add_argument(
        "--data-models-path",
        default="src/models",
    )

    parser.add_argument(
        "--backend-webhook-url",
        default="",
    )
    parser.add_argument(
        "--backend-alert-payload",
        default="src/outputs/model_update/backend_alert_payload.json",
    )

    parser.add_argument("--min-drift-history-size", type=int, default=8)
    parser.add_argument("--drift-threshold-quantile", type=float, default=0.95)

    parser.add_argument("--min-auc-drop-tolerance", type=float, default=0.01)
    parser.add_argument("--min-f1-drop-tolerance", type=float, default=0.02)
    parser.add_argument("--max-mae-increase-tolerance", type=float, default=0.01)
    parser.add_argument("--max-rmse-increase-tolerance", type=float, default=0.01)

    parser.add_argument("--replenish-candidates-on-demand", action="store_true")
    parser.add_argument("--candidate-target-size", type=int, default=None)
    parser.add_argument("--candidate-buffer-ratio", type=float, default=0.20)
    parser.add_argument("--min-candidate-size", type=int, default=30)
    parser.add_argument("--max-search-candidates", type=int, default=200)
    parser.add_argument("--search-per-page", type=int, default=50)
    parser.add_argument("--search-pages", type=int, default=2)
    parser.add_argument("--languages", nargs="*", default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_pipeline(args)

    print("Saved pipeline report:", args.pipeline_report)
    print("retrain_required:", report["retrain_required"])
    print("model_promoted:", report["model_promoted"])
    print("trigger_reason:", report["trigger_reason"])


if __name__ == "__main__":
    main()