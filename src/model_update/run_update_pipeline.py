# 전체 pipeline 실행
from __future__ import annotations
import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

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

def run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save_json(payload: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
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
    current_run_id = run_id()

    reference_result = run_reference_update_with_optional_replenish(args)
    reference_changed = reference_result.reference_changed

    drift_report = None
    thresholds = None
    model_update_required = False
    trigger_reason = None

    if reference_changed:
        model_update_required = True
        trigger_reason = "reference_changed"
    
    else:
        drift_report, thresholds = run_feature_drift_check(
            args=args,
            current_run_id=current_run_id,
        )

        if drift_report.retrain_required:
            model_update_required = True
            trigger_reason = "feature_drift"
        else:
            trigger_reason = "none"

    pipeline_report = {
        "run_id": current_run_id,
        "created_at": utc_now(),
        "model_update_required": model_update_required,
        "trigger_reason": trigger_reason,
        "reference_changed": reference_changed,
        "reference_update": asdict(reference_result),
        "drift_checked": drift_report is not None,
        "drift_report": asdict(drift_report) if drift_report is not None else None,
        "drift_thresholds": asdict(thresholds) if thresholds is not None else None,
        "next_step": (
            "train_challenger_model"
            if model_update_required
            else "no_model_update"
        ),
        "notes": [
            "This pipeline decides whether model update is required.",
            "Actual challenger training, champion-challenger evaluation, model registry promotion, and backend handoff are handled in the next stage.",
        ],
    }

    save_json(pipeline_report, args.pipeline_report)
    return pipeline_report

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OSS Health reference update and model update decision pipeline."
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
        default="src/reference_store/active/reference_next.csv",
    )
    parser.add_argument(
        "--output-candidate-pool",
        default="src/reference_store/candidates/candidate_pool_next.csv",
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
        "--pipeline-report",
        default="src/outputs/model_update/pipeline_report.json",
    )

    parser.add_argument(
        "--min-drift-history-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--drift-threshold-quantile",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--replenish-candidates-on-demand",
        action="store_true",
        help="Build candidate pool if reference replacement is required but candidates are missing.",
    )
    parser.add_argument(
        "--candidate-target-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--candidate-buffer-ratio",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--min-candidate-size",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--max-search-candidates",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--search-per-page",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--search-pages",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    report = run_pipeline(parse_args())

    print("Saved pipeline report.")
    print("model_update_required:", report["model_update_required"])
    print("trigger_reason:", report["trigger_reason"])


if __name__ == "__main__":
    main()