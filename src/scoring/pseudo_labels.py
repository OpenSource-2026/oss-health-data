from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_FAMILIES = {
    "contributor": [
        "num_contributors",
        "total_contributions",
        "top1_contribution_share",
        "top5_contribution_share",
        "contribution_gini",
        "median_contributions",
        "top1_contribution_ratio",
        "top3_contribution_ratio",
        "contribution_entropy",
        "contributors_to_stars_ratio",
    ],
    "release": [
        "num_deployments",
        "has_deployments",
        "num_unique_refs",
        "tag_based_deployment_ratio",
        "deployment_recency_days",
        "num_tags",
        "stable_tag_ratio",
        "prerelease_tag_ratio",
        "latest_tag_is_stable",
        "latest_tag_is_prerelease",
        "semver_tag_ratio",
        "num_major_versions",
        "num_minor_versions",
        "tag_release_velocity",
    ],
    "event": [
        "num_events",
        "num_unique_event_types",
        "dominant_event_ratio",
        "event_type_entropy",
        "has_IssuesEvent",
        "has_PullRequestEvent",
        "has_IssueCommentEvent",
        "recent_event_density",
        "IssuesEvent_ratio",
        "IssueCommentEvent_ratio",
        "PullRequestEvent_ratio",
        "PushEvent_ratio",
        "WatchEvent_ratio",
        "ForkEvent_ratio",
        "interaction_ratio",
        "development_ratio",
        "external_interest_event_ratio",
    ],
    "popularity": [
        "stargazers_count",
        "subscribers_count",
        "subscribers_to_stars_ratio",
        "forks_count",
        "network_count",
        "forks_to_stars_ratio",
        "open_issues_to_stars_ratio",
        "stars_per_repo_age_day",
        "forks_per_repo_age_day",
        "stars_per_size",
        "forks_per_size",
    ],
    "language": [
        "primary_language_ratio",
        "top2_ratio",
        "top3_ratio",
        "language_entropy",
        "minor_lang_ratio",
        "infra_ratio",
        "markup_ratio",
        "is_monolingual",
        "compiled_ratio",
    ],
    "maintenance": [
        "repo_age_days",
        "last_update_recency_days",
        "last_push_recency_days",
        "update_push_gap_days",
    ],
    "governance": [
        "repo_size",
        "open_issues_count",
        "has_issues",
        "has_projects",
        "has_downloads",
        "has_wiki",
        "has_pages",
        "has_discussions",
        "archived",
        "disabled",
        "allow_forking",
        "has_pull_requests",
        "issues_per_size",
    ],
}

SCORE_DIMENSIONS = {
    "community_activity": [
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
        "PushEvent_ratio",
        "interaction_ratio",
        "development_ratio",
    ],
    "contributor_sustainability": FEATURE_FAMILIES["contributor"],
    "release_engineering": FEATURE_FAMILIES["release"],
    "popularity_adoption": [
        "stargazers_count",
        "subscribers_count",
        "forks_count",
        "network_count",
        "subscribers_to_stars_ratio",
        "forks_to_stars_ratio",
        "stars_per_repo_age_day",
        "forks_per_repo_age_day",
        "stars_per_size",
        "forks_per_size",
        "external_interest_event_ratio",
    ],
    "language_structure": FEATURE_FAMILIES["language"],
    "maintenance": FEATURE_FAMILIES["maintenance"],
    "governance": [
        "repo_size",
        "open_issues_count",
        "has_issues",
        "has_projects",
        "has_downloads",
        "has_wiki",
        "has_pages",
        "has_discussions",
        "archived",
        "disabled",
        "allow_forking",
        "has_pull_requests",
        "open_issues_to_stars_ratio",
        "issues_per_size",
    ],
}

NEGATIVE_FEATURES = {
    "top1_contribution_share",
    "top5_contribution_share",
    "contribution_gini",
    "top1_contribution_ratio",
    "top3_contribution_ratio",
    "deployment_recency_days",
    "prerelease_tag_ratio",
    "latest_tag_is_prerelease",
    "last_update_recency_days",
    "last_push_recency_days",
    "update_push_gap_days",
    "open_issues_to_stars_ratio",
    "issues_per_size",
    "dominant_event_ratio",
    "archived",
    "disabled",
}


def existing_columns(columns: list[str], data: pd.DataFrame) -> list[str]:
    return [column for column in columns if column in data.columns]


def minmax_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    median = numeric.median()
    numeric = numeric.fillna(median)

    min_value = numeric.min()
    max_value = numeric.max()

    if pd.isna(min_value) or pd.isna(max_value) or min_value == max_value:
        return pd.Series(0.5, index=values.index)

    return (numeric - min_value) / (max_value - min_value)


def add_pseudo_health_targets(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = data.copy()
    scaled = pd.DataFrame(index=out.index)

    all_score_features = sorted(
        {
            feature
            for features in SCORE_DIMENSIONS.values()
            for feature in features
            if feature in out.columns
        }
    )

    for feature in all_score_features:
        scaled[feature] = minmax_series(out[feature])
        if feature in NEGATIVE_FEATURES:
            scaled[feature] = 1 - scaled[feature]

    dimension_score_cols = []

    for dimension, features in SCORE_DIMENSIONS.items():
        available = existing_columns(features, scaled)
        score_col = f"{dimension}_score"
        dimension_score_cols.append(score_col)

        if available:
            out[score_col] = scaled[available].mean(axis=1)
        else:
            out[score_col] = np.nan

    out["oss_health_score"] = out[dimension_score_cols].mean(axis=1)
    threshold = out["oss_health_score"].median()
    out["new_label"] = (out["oss_health_score"] >= threshold).astype(int)

    return out, dimension_score_cols