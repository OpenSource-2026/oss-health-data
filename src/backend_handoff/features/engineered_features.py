# 기존 파생 feature 생성 로직 분리
import numpy as np
import pandas as pd

def minmax_series(values):
    values = values.astype(float)
    min_value = values.min()
    max_value = values.max()

    if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
        return pd.Series(0.5, index=values.index)

    return (values - min_value) / (max_value - min_value)


def to_numeric_columns(data, columns):
    out = data.copy()

    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out
def add_engineered_features(data):
    out = data.copy()
    eps = 1e-9

    numeric_cols = [col for col in out.columns if col not in ["repo_name", "dominant_event_type", "error"]]
    out = to_numeric_columns(out, numeric_cols)

    out["is_recently_pushed_30d"] = (out["last_push_recency_days"] <= 30).astype(int)
    out["is_recently_updated_90d"] = (out["last_update_recency_days"] <= 90).astype(int)
    out["is_recently_pushed_90d"] = (out["last_push_recency_days"] <= 90).astype(int)
    out["is_stale_365d"] = (out["last_push_recency_days"] > 365).astype(int)

    out["push_update_consistency"] = 1 / (1 + out["update_push_gap_days"])
    out["freshness_score"] = (
        1 / (1 + out["last_push_recency_days"])
        + 1 / (1 + out["last_update_recency_days"])
    ) / 2

    out["bus_factor_risk"] = (
        out["top1_contribution_share"] + out["contribution_gini"]
    ) / 2
    out["distributed_contribution_score"] = (
        out["contribution_entropy"] * (1 - out["top1_contribution_share"])
    )
    out["contributor_depth_score"] = (
        np.log1p(out["num_contributors"])
        * np.log1p(out["median_contributions"])
    )

    out["release_maturity_score"] = (
        out["stable_tag_ratio"]
        * out["semver_tag_ratio"]
        * out["latest_tag_is_stable"]
    )
    out["active_release_score"] = (
        np.log1p(out["num_tags"]) * out["tag_release_velocity"]
    )
    out["release_recency_score"] = 1 / (1 + out["deployment_recency_days"])
    out["release_quality_score"] = (
        out["release_maturity_score"]
        + out["release_recency_score"]
        + minmax_series(out["active_release_score"].fillna(0))
    ) / 3

    out["collaboration_event_score"] = (
        out["PullRequestEvent_ratio"]
        + out["IssueCommentEvent_ratio"]
        + out["IssuesEvent_ratio"]
    )
    out["activity_diversity_score"] = (
        out["event_type_entropy"] * out["num_unique_event_types"]
    )
    out["healthy_activity_score"] = (
        out["interaction_ratio"]
        + out["development_ratio"]
        + out["collaboration_event_score"]
    ) / 3
    out["non_external_activity_ratio"] = 1 - out["external_interest_event_ratio"]

    out["adoption_efficiency"] = (
        np.log1p(out["stargazers_count"])
        / np.log1p(out["repo_age_days"] + 1)
    )
    out["issue_burden_score"] = (
        out["open_issues_count"]
        / (np.log1p(out["stargazers_count"]) + eps)
    )
    out["fork_interest_efficiency"] = (
        np.log1p(out["forks_count"])
        / np.log1p(out["repo_age_days"] + 1)
    )

    out["governance_openness_score"] = (
        out["has_issues"]
        + out["has_projects"]
        + out["has_wiki"]
        + out["has_discussions"]
        + out["has_pull_requests"]
    ) / 5
    out["negative_repo_state"] = (
        out["archived"] + out["disabled"]
    ).clip(0, 1)

    out["maintainer_activity_score"] = (
        out["is_recently_pushed_30d"]
        + out["is_recently_updated_90d"]
        + out["push_update_consistency"]
    ) / 3

    out.replace([np.inf, -np.inf], np.nan, inplace=True)

    return out
