from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DIMENSION_CONFIG = {
    "community_activity": {
        "label_ko": "커뮤니티 활성도",
        "core_question": "이 프로젝트는 현재 살아 움직이고 있는가?",
        "concepts": "Activity Volume, Responsiveness, Engagement Quality",
        "features": [
            "num_events",
            "num_unique_event_types",
            "event_type_entropy",
            "recent_event_density",
            "has_IssuesEvent",
            "has_PullRequestEvent",
            "has_IssueCommentEvent",
            "IssuesEvent_ratio",
            "IssueCommentEvent_ratio",
            "PullRequestEvent_ratio",
            "interaction_ratio",
            "development_ratio",
            "collaboration_event_score",
            "activity_diversity_score",
            "healthy_activity_score",
            "non_external_activity_ratio",
            "external_interest_event_ratio",
            "dominant_event_ratio",
        ],
    },
    "sustainability": {
        "label_ko": "지속 가능성",
        "core_question": "이 프로젝트는 앞으로도 유지될 수 있는가?",
        "concepts": "Contributor Structure, Diversity, Activity Stability",
        "features": [
            "num_contributors",
            "total_contributions",
            "median_contributions",
            "contribution_entropy",
            "top1_contribution_share",
            "top5_contribution_share",
            "contribution_gini",
            "top1_contribution_ratio",
            "top3_contribution_ratio",
            "contributors_to_stars_ratio",
            "bus_factor_risk",
            "distributed_contribution_score",
            "contributor_depth_score",
            "last_update_recency_days",
            "last_push_recency_days",
            "update_push_gap_days",
            "is_recently_pushed_30d",
            "is_recently_updated_90d",
            "is_stale_365d",
            "push_update_consistency",
            "freshness_score",
            "maintainer_activity_score",
        ],
    },
    "code_quality_reliability": {
        "label_ko": "코드 품질 및 신뢰성",
        "core_question": "이 프로젝트의 산출물은 믿을 수 있는가?",
        "concepts": "Engineering Practice, Defect Signals, Security Signals",
        "features": [
            "semver_tag_ratio",
            "stable_tag_ratio",
            "latest_tag_is_stable",
            "latest_tag_is_prerelease",
            "prerelease_tag_ratio",
            "release_maturity_score",
            "release_quality_score",
            "has_pull_requests",
            "PullRequestEvent_ratio",
            "development_ratio",
            "open_issues_to_stars_ratio",
            "issues_per_size",
            "issue_burden_score",
            "compiled_ratio",
            "language_entropy",
        ],
    },
    "legal_operational_governance": {
        "label_ko": "법적/운영 거버넌스",
        "core_question": "이 프로젝트는 조직적으로 안전하게 운영되는가?",
        "concepts": "Legal Compliance, Governance Structure",
        "features": [
            "has_issues",
            "has_projects",
            "has_downloads",
            "has_wiki",
            "has_pages",
            "has_discussions",
            "allow_forking",
            "has_pull_requests",
            "governance_openness_score",
            "archived",
            "disabled",
            "negative_repo_state",
        ],
    },
    "project_maturity": {
        "label_ko": "프로젝트 성숙도",
        "core_question": "이 프로젝트는 성숙한 운영 체계를 갖추었는가?",
        "concepts": "Release Engineering, Adoption/Popularity, Lifecycle/Scale",
        "features": [
            "num_tags",
            "num_major_versions",
            "num_minor_versions",
            "tag_release_velocity",
            "num_deployments",
            "has_deployments",
            "num_unique_refs",
            "tag_based_deployment_ratio",
            "deployment_recency_days",
            "release_recency_score",
            "active_release_score",
            "release_quality_score",
            "stargazers_count",
            "subscribers_count",
            "forks_count",
            "network_count",
            "stars_per_repo_age_day",
            "forks_per_repo_age_day",
            "stars_per_size",
            "forks_per_size",
            "adoption_efficiency",
            "fork_interest_efficiency",
            "repo_age_days",
            "repo_size",
        ],
    },
}


NEGATIVE_SCORE_FEATURES = {
    "top1_contribution_share",
    "top5_contribution_share",
    "contribution_gini",
    "top1_contribution_ratio",
    "top3_contribution_ratio",
    "bus_factor_risk",
    "deployment_recency_days",
    "prerelease_tag_ratio",
    "latest_tag_is_prerelease",
    "last_update_recency_days",
    "last_push_recency_days",
    "update_push_gap_days",
    "is_stale_365d",
    "open_issues_to_stars_ratio",
    "issues_per_size",
    "issue_burden_score",
    "dominant_event_ratio",
    "external_interest_event_ratio",
    "archived",
    "disabled",
    "negative_repo_state",
}


def percentile_score(
    value: Any,
    reference_values: pd.Series,
    higher_is_better: bool = True,
) -> float:
    values = pd.to_numeric(reference_values, errors="coerce").dropna().to_numpy(dtype=float)

    if len(values) == 0 or pd.isna(value):
        return np.nan

    percentile = (values <= float(value)).mean()

    if not higher_is_better:
        percentile = 1 - percentile

    return float(np.clip(percentile * 100, 0, 100))


def score_dimension_row(
    repo_features: pd.Series | pd.DataFrame,
    reference_df: pd.DataFrame,
    dimension_key: str,
) -> tuple[float, pd.DataFrame]:
    if dimension_key not in DIMENSION_CONFIG:
        raise KeyError(f"Unknown dimension key: {dimension_key}")

    if isinstance(repo_features, pd.DataFrame):
        if repo_features.empty:
            return np.nan, pd.DataFrame()
        repo_row = repo_features.iloc[0]
    else:
        repo_row = repo_features

    config = DIMENSION_CONFIG[dimension_key]
    rows = []

    for feature in config["features"]:
        if feature not in repo_row.index or feature not in reference_df.columns:
            continue

        value = repo_row[feature]
        higher_is_better = feature not in NEGATIVE_SCORE_FEATURES

        score = percentile_score(
            value=value,
            reference_values=reference_df[feature],
            higher_is_better=higher_is_better,
        )

        if pd.isna(score):
            continue

        rows.append({
            "dimension": dimension_key,
            "dimension_label": config["label_ko"],
            "feature": feature,
            "raw_value": value,
            "higher_is_better": higher_is_better,
            "feature_score": score,
        })

    detail = pd.DataFrame(rows)

    if detail.empty:
        return np.nan, detail

    dimension_score = float(np.clip(detail["feature_score"].mean(), 0, 100))
    return dimension_score, detail


def score_all_dimensions(
    repo_features: pd.Series | pd.DataFrame,
    reference_df: pd.DataFrame,
) -> dict[str, float]:
    scores = {}

    for dimension_key in DIMENSION_CONFIG:
        score, _detail = score_dimension_row(
            repo_features=repo_features,
            reference_df=reference_df,
            dimension_key=dimension_key,
        )
        scores[dimension_key] = score

    return scores


def score_all_dimensions_with_details(
    repo_features: pd.Series | pd.DataFrame,
    reference_df: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, pd.DataFrame]]:
    scores = {}
    details = {}

    for dimension_key in DIMENSION_CONFIG:
        score, detail = score_dimension_row(
            repo_features=repo_features,
            reference_df=reference_df,
            dimension_key=dimension_key,
        )
        scores[dimension_key] = score
        details[dimension_key] = detail

    return scores, details


def dimension_scores_frame(
    features_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for idx, row in features_df.iterrows():
        scores = score_all_dimensions(row, reference_df)

        rows.append({
            "index": idx,
            **{f"{key}_score": value for key, value in scores.items()},
        })

    return pd.DataFrame(rows).set_index("index")