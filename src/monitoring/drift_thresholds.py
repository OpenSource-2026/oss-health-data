from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


@dataclass
class DriftThresholds:
    overall_threshold: float
    drifted_ratio_threshold: float
    high_count_threshold: int
    source: str
    history_size: int

DEFAULT_THRESHOLDS = DriftThresholds(
    overall_threshold=0.15,
    drifted_ratio_threshold=0.30,
    high_count_threshold=5,
    source="cold_start_default",
    history_size=0,
)

def load_drift_history(history_path: str | Path) -> pd.DataFrame:
    path = Path(history_path)
    columns = [
        "run_id",
        "overall_drift_score",
        "drifted_feature_ratio",
        "high_drift_feature_count",
        "retrain_required",
        "created_at",
    ]

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)

    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame(columns=columns)

def calculate_dynamic_thresholds(
        history: pd.DataFrame,
        min_history_size: int = 8,
        quantile: float = 0.95,
) -> DriftThresholds:
    if len(history) < min_history_size:
        return DEFAULT_THRESHOLDS
    clean = history.copy()
    clean["overall_drift_score"] = pd.to_numeric(
        clean["overall_drift_score"],
        errors="coerce",
    )
    clean["drifted_feature_ratio"] = pd.to_numeric(
        clean["drifted_feature_ratio"],
        errors="coerce",
    )
    clean["high_drift_feature_count"] = pd.to_numeric(
        clean["high_drift_feature_count"],
        errors="coerce",
    )
    
    clean = clean.dropna(
        subset=[
            "overall_drift_score",
            "drifted_feature_ratio",
            "high_drift_feature_count",
        ]
    )

    if len(clean) < min_history_size:
        return DEFAULT_THRESHOLDS

    overall_threshold = float(clean["overall_drift_score"].quantile(quantile))
    drifted_ratio_threshold = float(clean["drifted_feature_ratio"].quantile(quantile))
    high_count_threshold = int(np.ceil(clean["high_drift_feature_count"].quantile(quantile))) # 가장 가까운 정수로 

    return DriftThresholds(
        overall_threshold=overall_threshold,
        drifted_ratio_threshold=drifted_ratio_threshold,
        high_count_threshold=max(1, high_count_threshold),
        source=f"history_q{int(quantile * 100)}",
        history_size=len(clean),
    )

def save_thresholds(
    thresholds: DriftThresholds,
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(thresholds), f, ensure_ascii=False, indent=2)

def append_drift_history(
    history_path: str | Path,
    run_record: dict,
) -> None:
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    history = load_drift_history(path)
    next_history = pd.concat(
        [history, pd.DataFrame([run_record])],
        ignore_index=True,
    )
    next_history.to_csv(path, index=False)