# GitHub Search API 후보 수집
# GitHub Search API에서 바로 필터링 가능한 값은 제한적임!

from __future__ import annotations
from datetime import datetime, timedelta, timezone
import os
import time
from dataclasses import dataclass
import pandas as pd
import requests
from features.build_features import build_all_features
from data.extract_github_repo import build_repo_dataframe
from reference.five_das import (add_five_das, calculate_reference_5das_snapshot, load_reference,)

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

# 정규화
def minmax_series(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    min_value = values.min()
    max_value = values.max()

    if pd.isna(min_value) or pd.isna(max_value) or min_value == max_value:
        return pd.Series(0.5, index=values.index)

    return (values - min_value) / (max_value - min_value)

# 후보 점수 계산
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

# 후보 탐색 정책
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
        min_stars=max(1, int(stars.quantile(quantile))),
        min_forks=max(0, int(forks.quantile(quantile))),
        max_push_recency_days=max(30, int(push_recency.quantile(1-quantile))),
        min_repo_age_days=max(90, int(repo_age.quantile(quantile))),
    )

def github_headers() -> dict[str, str]:
    headers = {"Accept":"application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

def search_repositories(
        query: str,
        per_page: int = 50,
        pages: int = 2,
        sleep_sec: float = 1.0,
) -> list[str]:
    repos: list[str] = [] 
    
    for page in range(1, pages + 1):
        response = requests.get(
            f"{GITHUB_API}/search/repositories",
            headers=github_headers(),
            params = {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": per_page,
                "page": page,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        repos.extend(item["full_name"] for item in payload.get("items", []))
        time.sleep(sleep_sec)
    
    return sorted(set(repos))

def github_date_days_ago(days: int) -> str:
    target_date = datetime.now(timezone.utc).date() - timedelta(days=days)
    return target_date.isoformat()

def build_candidate_queries(
    thresholds: CandidateSearchPolicy,
    languages: list[str],
) -> list[str]:
    pushed_after = github_date_days_ago(thresholds.max_push_recency_days)
    created_before = github_date_days_ago(thresholds.min_repo_age_days)

    clauses = [
        "is:public",
        f"stars:>={thresholds.min_stars}",
        f"forks:>={thresholds.min_forks}",
        f"pushed:>={pushed_after}",
        f"created:<={created_before}",
    ]

    if thresholds.require_not_archived:
        clauses.append("archived:false")

    if thresholds.require_not_fork:
        clauses.append("fork:false")

    base_query = " ".join(clauses)

    return [
        f"{base_query} language:{language}"
        for language in languages
    ]

def collect_candidate_repositories(
        thresholds: CandidateSearchPolicy,
        languages: list[str] | None = None) -> list[str]:
   
    if languages is None:
        languages = ["Python", "JavaScript", "TypeScript", "Go", "Rust", "Java"]
 
    queries = build_candidate_queries(
    thresholds=thresholds,
    languages=languages,
    )
    repos: list[str] = []

    for query in queries:
        found_repos = search_repositories(query)
        repos.extend(found_repos)

    return list(dict.fromkeys(repos))

# 아래부터는 for 5das 검증

def build_dataset(repos: list[str]) -> pd.DataFrame:
    feature_rows = []
    failed_rows = []

    for repo in repos:
        print(f"\nProcessing: {repo}")

        try:
            raw_df = build_repo_dataframe(repo)
            features = build_all_features(raw_df)

            # build_all_features가 Series를 반환한다고 가정
            if isinstance(features, pd.Series):
                features = features.to_frame().T

            features.insert(0, "repo_name", repo)
            feature_rows.append(features)

        except Exception as e:
            failed_rows.append({
                "repo_name": repo,
                "error": str(e),
            })
            print(f"Failed: {repo}")
            print(e)

    if not feature_rows:
        return pd.DataFrame()

    return pd.concat(feature_rows, ignore_index=True)

def five_das_check(
    repos: list[str],
    active_reference_path: str | Path,
    scoring_reference_path: str | Path,
    reference_snapshot_output_path: str | Path | None = None,
) -> pd.DataFrame:
    
    _, lower_bound = calculate_reference_5das_snapshot(
        active_reference_path=active_reference_path,
        scoring_reference_path=scoring_reference_path,
        output_path=reference_snapshot_output_path,
    )

    candidate_features = build_dataset(repos)

    if candidate_features.empty:
        return candidate_features

    scoring_reference = load_reference(scoring_reference_path)

    candidate_with_5das = add_five_das(
        features_df=candidate_features,
        reference_df=scoring_reference
    )

    candidate_with_5das["passes_5das"] = (
        candidate_with_5das["five_das"] >= lower_bound
    )

    passed_candidates = candidate_with_5das[
        candidate_with_5das["passes_5das"]
    ].copy()

    return passed_candidates