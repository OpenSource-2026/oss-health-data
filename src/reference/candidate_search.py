# GitHub Search API 후보 수집
# GitHub Search API에서 바로 필터링 가능한 값은 제한적임!

from __future__ import annotations

import os
import time
from dataclasses import dataclass
import pandas as pd
import requests

GITHUB_API = "https://api.github.com"

@dataclass
class CandidateSearchPolicy:
    min_proxy_score: float
    min_stars: int
    min_forks: int
    max_push_recency_days: int
    min_repo_age_days: int
    require_not_archived: bool = True
    require_not_fork: bool = True

QUERYABLE_PROXY_FEATURES = {
    "community_activity": ["last_push_recency_days"],
    "project_maturity": ["stargazers_count", "forks_count", "repo_age_days"],
    "governance": ["archived"],
}


NEGATIVE_PROXY_FEATURES = {
    "last_push_recency_days",
    "archived",
}

def minmax_series(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    min_value = values.min()
    max_value = values.max()

    if pd.isna(min_value) or pd.isna(max_value) or min_value == max_value:
        return pd.Series(0.5, index=values.index)

    return (values - min_value) / (max_value - min_value)

def calculate_candidate_proxy_score(active_reference: pd.DataFrame) -> pd.Series:
    proxy_features = [
        "stargazers_count",
        "forks_count",
        "repo_age_days",
        "last_push_recency_days",
        "archived",
    ]
    available = [col for col in proxy_features if col in active_reference.columns]
    scaled = pd.DataFrame(index=active_reference.index)
    for feature in available:
        scaled[feature] = minmax_series(active_reference[feature])
        if feature in NEGATIVE_PROXY_FEATURES:
            scaled[feature] = 1-scaled[feature]
    return scaled.mean(axis=1)

def build_candidate_search_policy(
    active_reference: pd.DataFrame,
    quantile: float=0.25,
) -> CandidateSearchPolicy:
    proxy_score = calculate_candidate_proxy_score(active_reference)
    stars = pd.to_numeric(active_reference["stargazers_count"], errors="coerce")
    forks = pd.to_numeric(active_reference["forks_count"], errors="coerce")
    push_recency = pd.to_numeric(active_reference["last_push_recency_days"], errors="coerce")
    repo_age = pd.to_numeric(active_reference["repo_age_days"], errors="coerce")

    return CandidateSearchPolicy(
        min_proxy_score=float(proxy_score.quantile(quantile)),
        min_starts=max(1, int(stars.quantile(quantile))),
        min_forks=max(0, int(forks.quantile(quantile))),
        max_push_recency_days=max(30, int(push_recency.quantile(1-quantile))),
        min_repo_age_days=max(90, int(repo_age.quantile(quantile))),
    )

