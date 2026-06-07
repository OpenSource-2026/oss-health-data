# 5DAS 계산 _ 5 Dimension Average Score 

from __future__ import annotations
import pandas as pd
from scoring.dimension_scores import score_all_dimensions
from pathlib import Path

FIVE_DIMENSION_KEYS = [
    "community_activity",
    "sustainability",
    "code_quality_reliability",
    "legal_operational_governance",
    "project_maturity",  
]

# 각 dimension 별 feature keys_개별 repo
def calculate_five_das(keys: list[str], scores: dict[str, float]) -> float:
    valid_scores = [scores[key] for key in keys if key in scores and pd.notna(scores[key])]
    if not valid_scores:
        return float('nan')
    return float(sum(valid_scores) / len(valid_scores))

# 5DAS 계산_개별 repo
def calculate_five_das_for_repo(repo_features: pd.Series, reference_df: pd.DataFrame) -> float:
    all_scores = score_all_dimensions(repo_features, reference_df)
    return calculate_five_das(FIVE_DIMENSION_KEYS, all_scores)

# 5DAS 계산_전체 dataset
def add_five_das(
    features_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    out = features_df.copy()
    out["five_das"] = out.apply(lambda row: calculate_five_das_for_repo(row, reference_df), axis=1)
    return out

def healthy_lower_bound(scores, reference_scores) -> float:
    return reference_scores["five_das"].quantile(0.10)

def load_reference(reference_path: str)-> pd.DataFrame:
    return pd.read_csv(reference_path)

def save_reference_with_5das(
    reference_with_5das: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    reference_with_5das.to_csv(output, index=False)

def calculate_reference_5das_snapshot(
    active_reference_path: str | Path,
    scoring_reference_path: str | Path,
    output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, float]:
    active_reference = load_reference(active_reference_path)
    scoring_reference = load_reference(scoring_reference_path)

    active_with_5das = add_five_das(
        features_df=active_reference,
        reference_df=scoring_reference,
    )

    lower_bound = healthy_lower_bound(active_with_5das)

    if output_path is not None:
        save_reference_with_5das(active_with_5das, output_path)

    return active_with_5das, lower_bound

# active reference _ 점수를 매길 대상 데이터셋
# scoring reference _ 점수 매김의 기준이 되는 데이터셋 (5DAS 계산에 활용)