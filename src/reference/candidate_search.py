# src/reference/candidate_search.py

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from pandas.errors import EmptyDataError

from data.extract_github_repo import build_repo_dataframe
from features.build_features import build_all_features
from reference.five_das import (
    add_five_das,
    calculate_reference_5das_snapshot,
    load_reference,
)


GITHUB_API = "https://api.github.com"

DEFAULT_LANGUAGES = [
    "Python",
    "JavaScript",
    "TypeScript",
    "Go",
    "Rust",
    "Java",
]


@dataclass
class CandidateSearchPolicy:
    min_proxy_score: float
    min_stars: int
    min_forks: int
    max_push_recency_days: int
    min_repo_age_days: int
    require_not_archived: bool = True
    require_not_fork: bool = True


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

    available = [
        feature for feature in proxy_features
        if feature in active_reference.columns
    ]

    if not available:
        raise ValueError("No proxy features available in active_reference.")

    scaled = pd.DataFrame(index=active_reference.index)

    for feature in available:
        scaled[feature] = minmax_series(active_reference[feature])

        if feature in NEGATIVE_PROXY_FEATURES:
            scaled[feature] = 1 - scaled[feature]

    return scaled.mean(axis=1)


def build_candidate_search_policy(
    active_reference: pd.DataFrame,
    quantile: float = 0.25,
) -> CandidateSearchPolicy:
    proxy_score = calculate_candidate_proxy_score(active_reference)

    required = [
        "stargazers_count",
        "forks_count",
        "last_push_recency_days",
        "repo_age_days",
    ]
    missing = [column for column in required if column not in active_reference.columns]
    if missing:
        raise ValueError(f"active_reference is missing required columns: {missing}")

    stars = pd.to_numeric(active_reference["stargazers_count"], errors="coerce")
    forks = pd.to_numeric(active_reference["forks_count"], errors="coerce")
    push_recency = pd.to_numeric(
        active_reference["last_push_recency_days"],
        errors="coerce",
    )
    repo_age = pd.to_numeric(active_reference["repo_age_days"], errors="coerce")

    return CandidateSearchPolicy(
        min_proxy_score=float(proxy_score.quantile(quantile)),
        min_stars=max(1, int(stars.quantile(quantile))),
        min_forks=max(0, int(forks.quantile(quantile))),
        max_push_recency_days=max(30, int(push_recency.quantile(1 - quantile))),
        min_repo_age_days=max(90, int(repo_age.quantile(quantile))),
    )


def github_headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")

    if token:
        headers["Authorization"] = f"Bearer {token}"

    return headers


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
            params={
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
        repos.extend(
            item["full_name"]
            for item in payload.get("items", [])
            if "full_name" in item
        )

        time.sleep(sleep_sec)

    return sorted(set(repos))


def collect_candidate_repositories(
    thresholds: CandidateSearchPolicy,
    languages: list[str] | None = None,
    per_page: int = 50,
    pages: int = 2,
) -> list[str]:
    if languages is None:
        languages = DEFAULT_LANGUAGES

    queries = build_candidate_queries(
        thresholds=thresholds,
        languages=languages,
    )

    repos: list[str] = []

    for query in queries:
        found_repos = search_repositories(
            query=query,
            per_page=per_page,
            pages=pages,
        )
        repos.extend(found_repos)

    return list(dict.fromkeys(repos))


def build_dataset(repos: list[str]) -> pd.DataFrame:
    feature_rows = []
    failed_rows = []

    for repo in repos:
        print(f"\nProcessing: {repo}")

        try:
            raw_df = build_repo_dataframe(repo)
            features = build_all_features(raw_df)

            if isinstance(features, pd.Series):
                features = features.to_frame().T

            features.insert(0, "repo_name", repo)
            feature_rows.append(features)

        except Exception as exc:
            failed_rows.append(
                {
                    "repo_name": repo,
                    "error": str(exc),
                }
            )
            print(f"Failed: {repo}")
            print(exc)

    if not feature_rows:
        return pd.DataFrame()

    dataset = pd.concat(feature_rows, ignore_index=True)

    if failed_rows:
        failed_df = pd.DataFrame(failed_rows)
        print("\nFailed candidate repos:")
        print(failed_df.to_string(index=False))

    return dataset


def read_optional_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(csv_path)
    except EmptyDataError:
        return pd.DataFrame()


def filter_healthy_reference(active_reference: pd.DataFrame) -> pd.DataFrame:
    if "label" not in active_reference.columns:
        raise ValueError("active_reference must contain a 'label' column.")

    healthy = active_reference[active_reference["label"] == 1].copy()

    if healthy.empty:
        raise ValueError("No healthy reference rows found where label == 1.")

    return healthy


def existing_repo_names(*frames: pd.DataFrame) -> set[str]:
    names: set[str] = set()

    for frame in frames:
        if frame.empty or "repo_name" not in frame.columns:
            continue

        names.update(frame["repo_name"].dropna().astype(str).tolist())

    return names


def five_das_check(
    repos: list[str],
    active_reference_path: str | Path,
    scoring_reference_path: str | Path,
    reference_snapshot_output_path: str | Path | None = None,
) -> pd.DataFrame:
    active_with_5das, lower_bound = calculate_reference_5das_snapshot(
        active_reference_path=active_reference_path,
        scoring_reference_path=scoring_reference_path,
        output_path=reference_snapshot_output_path,
    )
    active_with_5das = filter_healthy_reference(active_with_5das)

    candidate_features = build_dataset(repos)

    if candidate_features.empty:
        return candidate_features

    scoring_reference = load_reference(scoring_reference_path)

    candidate_with_5das = add_five_das(
        features_df=candidate_features,
        reference_df=scoring_reference,
    )

    candidate_with_5das["passes_5das"] = (
        candidate_with_5das["five_das"] >= lower_bound
    )

    passed_candidates = candidate_with_5das[
        candidate_with_5das["passes_5das"]
    ].copy()

    return passed_candidates


def build_candidate_pool(
    active_reference_path: str | Path,
    scoring_reference_path: str | Path,
    candidate_pool_path: str | Path,
    output_path: str | Path,
    target_size: int | None = None,
    candidate_buffer_ratio: float = 0.20,
    min_candidate_size: int = 30,
    languages: list[str] | None = None,
    max_search_candidates: int = 200,
    per_page: int = 50,
    pages: int = 2,
) -> pd.DataFrame:
    active_reference = load_reference(active_reference_path)
    existing_candidate_pool = read_optional_csv(candidate_pool_path)

    healthy_reference = filter_healthy_reference(active_reference)

    active_with_5das, lower_bound = calculate_reference_5das_snapshot(
        active_reference_path=active_reference_path,
        scoring_reference_path=scoring_reference_path,
        output_path=None,
    )
    healthy_with_5das = filter_healthy_reference(active_with_5das)

    if target_size is None:
        target_size = max(
            min_candidate_size,
            int(len(healthy_with_5das) * candidate_buffer_ratio),
        )

    thresholds = build_candidate_search_policy(
        active_reference=healthy_reference,
        quantile=0.25,
    )

    repo_candidates = collect_candidate_repositories(
        thresholds=thresholds,
        languages=languages,
        per_page=per_page,
        pages=pages,
    )

    excluded = existing_repo_names(
        active_reference,
        existing_candidate_pool,
    )

    repo_candidates = [
        repo for repo in repo_candidates
        if repo not in excluded
    ]
    repo_candidates = repo_candidates[:max_search_candidates]

    if not repo_candidates:
        raise ValueError("No repository candidates found after excluding known repos.")

    passed_candidates = five_das_check(
        repos=repo_candidates,
        active_reference_path=active_reference_path,
        scoring_reference_path=scoring_reference_path,
        reference_snapshot_output_path=None,
    )

    if passed_candidates.empty:
        raise ValueError("No candidate passed 5DAS validation.")

    passed_candidates = passed_candidates[
        passed_candidates["five_das"] >= lower_bound
    ].copy()

    if passed_candidates.empty:
        raise ValueError("No candidate remained after lower-bound filtering.")

    passed_candidates["label"] = 1
    passed_candidates["candidate_status"] = "standby"
    passed_candidates["candidate_source"] = "github_search_prefilter_5das"
    passed_candidates["below_bound_count"] = 0

    if not existing_candidate_pool.empty:
        combined = pd.concat(
            [existing_candidate_pool, passed_candidates],
            ignore_index=True,
        )
        combined = combined.drop_duplicates(subset=["repo_name"], keep="first")
    else:
        combined = passed_candidates

    combined = combined.sort_values("five_das", ascending=False)
    output = combined.head(target_size).copy()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_file, index=False)

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or replenish healthy reference candidate pool."
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
        "--output",
        default="src/reference_store/candidates/candidate_pool.csv",
    )
    parser.add_argument(
        "--target-size",
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
        "--per-page",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=DEFAULT_LANGUAGES,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pool = build_candidate_pool(
        active_reference_path=args.active_reference,
        scoring_reference_path=args.scoring_reference,
        candidate_pool_path=args.candidate_pool,
        output_path=args.output,
        target_size=args.target_size,
        candidate_buffer_ratio=args.candidate_buffer_ratio,
        min_candidate_size=args.min_candidate_size,
        languages=args.languages,
        max_search_candidates=args.max_search_candidates,
        per_page=args.per_page,
        pages=args.pages,
    )

    print("Saved candidate pool:", args.output)
    print("Candidate pool size:", len(pool))

    if "repo_name" in pool.columns and "five_das" in pool.columns:
        print(pool[["repo_name", "five_das"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()