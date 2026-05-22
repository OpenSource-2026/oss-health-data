import ast
import re
import numpy as np
import pandas as pd


def safe_parse_dict(value):
    try:
        if isinstance(value, dict):
            return value
        return ast.literal_eval(value)
    except Exception:
        return {}


def get_scalar(data, path_name, default=0.0):
    row = data[data["path"] == path_name]
    if row.empty:
        return default

    try:
        return float(row["example_value"].iloc[0])
    except Exception:
        return default


def get_bool(data, path_name, default=0.0):
    row = data[data["path"] == path_name]
    if row.empty:
        return default

    value = row["example_value"].iloc[0]

    if isinstance(value, bool):
        return float(value)

    if str(value).lower() == "true":
        return 1.0
    if str(value).lower() == "false":
        return 0.0

    return default


def get_datetime(data, path_name):
    row = data[data["path"] == path_name]
    if row.empty:
        return pd.NaT

    return pd.to_datetime(row["example_value"].iloc[0], errors="coerce", utc=True)


def gini(values):
    values = np.array(values, dtype=float)

    if len(values) == 0:
        return 0.0

    if np.all(values == 0):
        return 0.0

    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)

    return (2 * np.sum(index * values)) / (n * np.sum(values)) - (n + 1) / n


def entropy(values):
    values = np.array(values, dtype=float)

    if len(values) == 0 or values.sum() == 0:
        return 0.0

    probs = values / values.sum()
    return float(-np.sum(probs * np.log(probs)))


def make_contributor_features(data):
    rows = data[
        data["path"].astype(str).str.startswith("contributors_url.__api__")
    ].copy()

    if rows.empty:
        return pd.DataFrame([{
            "num_contributors": 0.0,
            "total_contributions": 0.0,
            "top1_contribution_share": 0.0,
            "top5_contribution_share": 0.0,
            "contribution_gini": 0.0,
            "median_contributions": 0.0,
        }])

    rows["obj"] = rows["example_value"].apply(safe_parse_dict)
    contributions = rows["obj"].apply(lambda x: x.get("contributions", 0)).astype(float)

    total = contributions.sum()
    sorted_contrib = contributions.sort_values(ascending=False)

    return pd.DataFrame([{
        "num_contributors": len(contributions),
        "total_contributions": total,
        "top1_contribution_share": sorted_contrib.iloc[0] / total if total > 0 else 0.0,
        "top5_contribution_share": sorted_contrib.head(5).sum() / total if total > 0 else 0.0,
        "contribution_gini": gini(contributions),
        "median_contributions": contributions.median() if len(contributions) > 0 else 0.0,
    }])


def make_deployments_features(data):
    rows = data[
        data["path"].astype(str).str.startswith("deployments_url.__api__")
    ].copy()

    if rows.empty:
        return pd.DataFrame([{
            "num_deployments": 0.0,
            "has_deployments": 0.0,
            "num_unique_refs": 0.0,
            "tag_based_deployment_ratio": 0.0,
            "deployment_recency_days": 0.0,
        }])

    rows["obj"] = rows["example_value"].apply(safe_parse_dict)
    rows["ref"] = rows["obj"].apply(lambda x: x.get("ref", None))
    rows["created_at"] = rows["obj"].apply(lambda x: x.get("created_at", None))

    total = len(rows)

    is_tag_based = rows["ref"].astype(str).str.contains(
        r"^v?\d+\.\d+\.\d+",
        regex=True,
        na=False,
    )

    times = pd.to_datetime(rows["created_at"], errors="coerce", utc=True).dropna()

    if len(times) > 0:
        now = pd.Timestamp.now(tz="UTC")
        recency_days = (now - times.max()).total_seconds() / 86400
    else:
        recency_days = 0.0

    return pd.DataFrame([{
        "num_deployments": total,
        "has_deployments": 1.0,
        "num_unique_refs": rows["ref"].dropna().nunique(),
        "tag_based_deployment_ratio": is_tag_based.sum() / total if total > 0 else 0.0,
        "deployment_recency_days": recency_days,
    }])


def make_events_features(data):
    rows = data[
        data["path"].astype(str).str.startswith("events_url.__api__")
    ].copy()

    event_types = [
        "IssuesEvent",
        "IssueCommentEvent",
        "PullRequestEvent",
        "PushEvent",
        "WatchEvent",
        "ForkEvent",
    ]

    if rows.empty:
        result = {
            "num_events": 0.0,
            "num_unique_event_types": 0.0,
            "dominant_event_type": None,
            "dominant_event_ratio": 0.0,
            "event_type_entropy": 0.0,
            "has_IssuesEvent": 0.0,
            "has_PullRequestEvent": 0.0,
            "has_IssueCommentEvent": 0.0,
            "recent_event_density": 0.0,
        }

        for event_type in event_types:
            result[f"{event_type}_ratio"] = 0.0

        return pd.DataFrame([result])

    rows["obj"] = rows["example_value"].apply(safe_parse_dict)
    rows["event_type"] = rows["obj"].apply(lambda x: x.get("type", None))
    rows["created_at"] = rows["obj"].apply(lambda x: x.get("created_at", None))

    total = len(rows)
    counts = rows["event_type"].dropna().value_counts()
    ratios = counts / total

    probs = ratios.values
    event_entropy = -np.sum(probs * np.log(probs)) if len(probs) > 0 else 0.0

    if len(counts) > 0:
        dominant_type = counts.idxmax()
        dominant_ratio = counts.max() / total
    else:
        dominant_type = None
        dominant_ratio = 0.0

    times = pd.to_datetime(rows["created_at"], errors="coerce", utc=True).dropna()

    if len(times) >= 2:
        period_days = (times.max() - times.min()).total_seconds() / 86400
        recent_event_density = total / period_days if period_days > 0 else float(total)
    elif len(times) == 1:
        recent_event_density = float(total)
    else:
        recent_event_density = 0.0

    event_set = set(rows["event_type"].dropna())

    result = {
        "num_events": total,
        "num_unique_event_types": rows["event_type"].nunique(),
        "dominant_event_type": dominant_type,
        "dominant_event_ratio": dominant_ratio,
        "event_type_entropy": event_entropy,
        "has_IssuesEvent": float("IssuesEvent" in event_set),
        "has_PullRequestEvent": float("PullRequestEvent" in event_set),
        "has_IssueCommentEvent": float("IssueCommentEvent" in event_set),
        "recent_event_density": recent_event_density,
    }

    for event_type in event_types:
        result[f"{event_type}_ratio"] = ratios.get(event_type, 0.0)

    return pd.DataFrame([result])


def make_interest_features(data):
    stars = get_scalar(data, "stargazers_count")
    subscribers = get_scalar(data, "subscribers_count")

    return pd.DataFrame([{
        "stargazers_count": stars,
        "subscribers_count": subscribers,
        "subscribers_to_stars_ratio": subscribers / stars if stars > 0 else 0.0,
    }])


def make_language_features(data):
    rows = data[
        data["path"].astype(str).str.startswith("languages_url.__api__.")
    ].copy()

    if rows.empty:
        return pd.DataFrame([{
            "primary_language_ratio": 0.0,
            "top2_ratio": 0.0,
            "top3_ratio": 0.0,
            "language_entropy": 0.0,
            "minor_lang_ratio": 0.0,
            "infra_ratio": 0.0,
            "markup_ratio": 0.0,
            "is_monolingual": 0.0,
            "compiled_ratio": 0.0,
        }])

    rows["language"] = rows["key"].astype(str)
    rows["bytes"] = pd.to_numeric(rows["example_value"], errors="coerce").fillna(0.0)

    total = rows["bytes"].sum()

    if total <= 0:
        return pd.DataFrame([{
            "primary_language_ratio": 0.0,
            "top2_ratio": 0.0,
            "top3_ratio": 0.0,
            "language_entropy": 0.0,
            "minor_lang_ratio": 0.0,
            "infra_ratio": 0.0,
            "markup_ratio": 0.0,
            "is_monolingual": 0.0,
            "compiled_ratio": 0.0,
        }])

    rows = rows.sort_values("bytes", ascending=False)
    ratios = rows["bytes"] / total

    compiled_langs = {"C", "C++", "Rust", "Go", "Java", "Swift"}
    infra_langs = {"Shell", "Dockerfile", "Makefile", "Meson"}
    markup_langs = {"HTML", "CSS", "XSLT", "Smarty", "Go Template"}

    compiled_ratio = rows.loc[rows["language"].isin(compiled_langs), "bytes"].sum() / total
    infra_ratio = rows.loc[rows["language"].isin(infra_langs), "bytes"].sum() / total
    markup_ratio = rows.loc[rows["language"].isin(markup_langs), "bytes"].sum() / total

    return pd.DataFrame([{
        "primary_language_ratio": ratios.iloc[0],
        "top2_ratio": ratios.head(2).sum(),
        "top3_ratio": ratios.head(3).sum(),
        "language_entropy": entropy(rows["bytes"]),
        "minor_lang_ratio": ratios.iloc[1:].sum() if len(ratios) > 1 else 0.0,
        "infra_ratio": infra_ratio,
        "markup_ratio": markup_ratio,
        "is_monolingual": float(ratios.iloc[0] >= 0.95),
        "compiled_ratio": compiled_ratio,
    }])


def make_tag_features(data):
    rows = data[
        data["path"].astype(str).str.startswith("tags_url.__api__")
    ].copy()

    if rows.empty:
        return pd.DataFrame([{
            "num_tags": 0.0,
            "stable_tag_ratio": 0.0,
            "prerelease_tag_ratio": 0.0,
            "latest_tag_is_stable": 0.0,
            "latest_tag_is_prerelease": 0.0,
            "semver_tag_ratio": 0.0,
            "num_major_versions": 0.0,
            "num_minor_versions": 0.0,
        }])

    rows["obj"] = rows["example_value"].apply(safe_parse_dict)
    rows["tag_name"] = rows["obj"].apply(lambda x: x.get("name", ""))

    total = len(rows)

    prerelease_pattern = r"(dev|rc|alpha|beta|a\d+|b\d+)"
    semver_pattern = r"^v?\d+\.\d+\.\d+"

    is_prerelease = rows["tag_name"].str.contains(
        prerelease_pattern,
        case=False,
        regex=True,
        na=False,
    )

    is_semver = rows["tag_name"].str.contains(
        semver_pattern,
        regex=True,
        na=False,
    )

    is_stable = is_semver & (~is_prerelease)

    versions = rows["tag_name"].str.extract(r"^v?(\d+)\.(\d+)\.(\d+)")

    return pd.DataFrame([{
        "num_tags": total,
        "stable_tag_ratio": is_stable.sum() / total if total > 0 else 0.0,
        "prerelease_tag_ratio": is_prerelease.sum() / total if total > 0 else 0.0,
        "latest_tag_is_stable": float(is_stable.iloc[0]) if total > 0 else 0.0,
        "latest_tag_is_prerelease": float(is_prerelease.iloc[0]) if total > 0 else 0.0,
        "semver_tag_ratio": is_semver.sum() / total if total > 0 else 0.0,
        "num_major_versions": versions[0].dropna().nunique(),
        "num_minor_versions": versions[[0, 1]].dropna().drop_duplicates().shape[0],
    }])


def make_repo_meta_features(data):
    now = pd.Timestamp.now(tz="UTC")

    created_at = get_datetime(data, "created_at")
    updated_at = get_datetime(data, "updated_at")
    pushed_at = get_datetime(data, "pushed_at")

    repo_age_days = (
        (now - created_at).total_seconds() / 86400
        if pd.notna(created_at)
        else 0.0
    )

    last_update_recency_days = (
        (now - updated_at).total_seconds() / 86400
        if pd.notna(updated_at)
        else 0.0
    )

    last_push_recency_days = (
        (now - pushed_at).total_seconds() / 86400
        if pd.notna(pushed_at)
        else 0.0
    )

    update_push_gap_days = (
        abs((updated_at - pushed_at).total_seconds()) / 86400
        if pd.notna(updated_at) and pd.notna(pushed_at)
        else 0.0
    )

    return pd.DataFrame([{
        "repo_age_days": repo_age_days,
        "last_update_recency_days": last_update_recency_days,
        "last_push_recency_days": last_push_recency_days,
        "update_push_gap_days": update_push_gap_days,
        "repo_size": get_scalar(data, "size"),
        "forks_count": get_scalar(data, "forks_count"),
        "open_issues_count": get_scalar(data, "open_issues_count"),
        "network_count": get_scalar(data, "network_count"),
        "has_issues": get_bool(data, "has_issues"),
        "has_projects": get_bool(data, "has_projects"),
        "has_downloads": get_bool(data, "has_downloads"),
        "has_wiki": get_bool(data, "has_wiki"),
        "has_pages": get_bool(data, "has_pages"),
        "has_discussions": get_bool(data, "has_discussions"),
        "archived": get_bool(data, "archived"),
        "disabled": get_bool(data, "disabled"),
        "allow_forking": get_bool(data, "allow_forking"),
        "has_pull_requests": get_bool(data, "has_pull_requests"),
    }])


def make_additional_features(data):
    stars = get_scalar(data, "stargazers_count")
    forks = get_scalar(data, "forks_count")
    subscribers = get_scalar(data, "subscribers_count")
    open_issues = get_scalar(data, "open_issues_count")
    size = get_scalar(data, "size")

    created_at = get_datetime(data, "created_at")
    now = pd.Timestamp.now(tz="UTC")

    repo_age_days = (
        (now - created_at).total_seconds() / 86400
        if pd.notna(created_at)
        else 0.0
    )

    contributor_rows = data[
        data["path"].astype(str).str.startswith("contributors_url.__api__")
    ].copy()

    if contributor_rows.empty:
        num_contributors = 0.0
        total_contributions = 0.0
        top1_contribution_ratio = 0.0
        top3_contribution_ratio = 0.0
        contribution_entropy = 0.0
    else:
        contributor_rows["obj"] = contributor_rows["example_value"].apply(safe_parse_dict)
        contributions = contributor_rows["obj"].apply(
            lambda x: x.get("contributions", 0)
        ).astype(float)

        num_contributors = len(contributions)
        total_contributions = contributions.sum()

        if total_contributions > 0:
            sorted_contrib = contributions.sort_values(ascending=False)
            top1_contribution_ratio = sorted_contrib.iloc[0] / total_contributions
            top3_contribution_ratio = sorted_contrib.head(3).sum() / total_contributions
            contribution_entropy = entropy(contributions)
        else:
            top1_contribution_ratio = 0.0
            top3_contribution_ratio = 0.0
            contribution_entropy = 0.0

    tag_rows = data[
        data["path"].astype(str).str.startswith("tags_url.__api__")
    ].copy()

    event_rows = data[
        data["path"].astype(str).str.startswith("events_url.__api__")
    ].copy()

    if event_rows.empty:
        interaction_ratio = 0.0
        development_ratio = 0.0
        external_interest_event_ratio = 0.0
    else:
        event_rows["obj"] = event_rows["example_value"].apply(safe_parse_dict)
        event_rows["event_type"] = event_rows["obj"].apply(lambda x: x.get("type", None))

        total_events = len(event_rows)

        interaction_ratio = event_rows["event_type"].isin([
            "IssuesEvent",
            "IssueCommentEvent",
        ]).sum() / total_events

        development_ratio = event_rows["event_type"].isin([
            "PushEvent",
            "PullRequestEvent",
        ]).sum() / total_events

        external_interest_event_ratio = event_rows["event_type"].isin([
            "WatchEvent",
            "ForkEvent",
        ]).sum() / total_events

    return pd.DataFrame([{
        "forks_to_stars_ratio": forks / stars if stars > 0 else 0.0,
        "open_issues_to_stars_ratio": open_issues / stars if stars > 0 else 0.0,
        "stars_per_repo_age_day": stars / repo_age_days if repo_age_days > 0 else 0.0,
        "forks_per_repo_age_day": forks / repo_age_days if repo_age_days > 0 else 0.0,
        "top1_contribution_ratio": top1_contribution_ratio,
        "top3_contribution_ratio": top3_contribution_ratio,
        "contribution_entropy": contribution_entropy,
        "contributors_to_stars_ratio": num_contributors / stars if stars > 0 else 0.0,
        "tag_release_velocity": len(tag_rows) / repo_age_days if repo_age_days > 0 else 0.0,
        "interaction_ratio": interaction_ratio,
        "development_ratio": development_ratio,
        "external_interest_event_ratio": external_interest_event_ratio,
        "stars_per_size": stars / size if size > 0 else 0.0,
        "forks_per_size": forks / size if size > 0 else 0.0,
        "issues_per_size": open_issues / size if size > 0 else 0.0,
    }])


def build_all_features(data):
    features = pd.concat([
        make_contributor_features(data),
        make_deployments_features(data),
        make_events_features(data),
        make_interest_features(data),
        make_language_features(data),
        make_tag_features(data),
        make_repo_meta_features(data),
        make_additional_features(data),
    ], axis=1)

    features = features.loc[:, ~features.columns.duplicated()]

    return features