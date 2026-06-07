# feature drift 감지

from dataclasses import asdict, dataclass
import pandas as pd
import numpy as np

@dataclass
class FeatureDrift:
    feature: str
    method: str
    score: float
    severity: str

@dataclass
class DriftReport:
    retrain_required: bool
    overall_drift_score: float
    drifted_feature_ratio: float
    high_drift_feature_count: int
    reasons: list[str]
    features: list[FeatureDrift]

def is_binary(series: pd.Series) -> bool:
    values = set(pd.to_numeric(series, errors="coerce").dropna().unique())
    return values <= {0, 1, 0.0, 1.0}

def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype=float)

    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) <= 2:
        return 0.0
    # ref data가 각 구간 bin에 몇 개씩 들어가는지 세는 함수
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

def binary_ratio_diff(reference: pd.Series, current: pd.Series) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if len(ref) == 0 or len(cur) == 0: return 0.0
    return float(abs(cur.mean() - ref.mean()))

def severity(score: float) -> str:
    if score >= 0.25:
        return "high"
    if score >= 0.10:
        return "medium"
    return "low"

def detect_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_columns: list[str],
    overall_threshold: float,
    drifted_ratio_threshold: float,
    high_count_threshold: int,
) -> DriftReport:
    
    results: list[FeatureDrift] = []

    for feature in feature_columns:
        if feature not in reference.columns or feature not in current.columns:
            continue

        if is_binary(reference[feature]) and is_binary(current[feature]):
            method = "binary_ratio_diff"
            score = binary_ratio_diff(reference[feature], current[feature])
        else:
            method = "psi"
            score = psi(reference[feature], current[feature])

        results.append(
            FeatureDrift(
                feature=feature,
                method=method,
                score=score,
                severity=severity(score),
            )
        )

    if not results:
        raise ValueError("Cannot compare drift")
    scores = np.array([item.score for item in results])
    drifted = [item for item in results if item.severity in {"medium", "high"}]
    high = [item for item in results if item.severity == "high"]
    overall = float(scores.mean())
    drifted_ratio = len(drifted) / len(results)

    reasons = []

    if overall >= overall_threshold:
        reasons.append(f"overall_drift_score >= {overall_threshold:.4f}")

    if drifted_ratio >= drifted_ratio_threshold:
        reasons.append(f"drifted_feature_ratio >= {drifted_ratio_threshold:.4f}")

    if len(high) >= high_count_threshold:
        reasons.append(f"high_drift_feature_count >= {high_count_threshold}")

    return DriftReport(
        retrain_required=bool(reasons),
        overall_drift_score=overall,
        drifted_feature_ratio=float(drifted_ratio),
        high_drift_feature_count=len(high),
        reasons=reasons,
        features=results,
    )