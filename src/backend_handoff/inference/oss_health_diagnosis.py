from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

import warnings
warnings.filterwarnings("ignore", category=PerformanceWarning)

from data.extract_github_repo import build_repo_dataframe
from features.build_features import build_all_features

PACKAGE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PACKAGE_DIR / "models" / "oss_health_best_model.joblib"
FEATURE_PATH = PACKAGE_DIR / "models" / "oss_health_best_features.json"
METADATA_PATH = PACKAGE_DIR / "models" / "oss_health_model_metadata.json"
REFERENCE_DATA_PATH = PACKAGE_DIR / "data" / "reference_dataset.csv"


def _load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_PATH, "r", encoding="utf-8") as f:
        model_features = json.load(f)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        model_metadata = json.load(f)
    return model, model_features, model_metadata


MODEL, MODEL_FEATURES, MODEL_METADATA = _load_artifacts()


def minmax_series(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    min_value = values.min()
    max_value = values.max()
    if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
        return pd.Series(0.5, index=values.index)
    return (values - min_value) / (max_value - min_value)


def to_numeric_columns(data: pd.DataFrame, columns) -> pd.DataFrame:
    out = data.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
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

    out["bus_factor_risk"] = (out["top1_contribution_share"] + out["contribution_gini"]) / 2
    out["distributed_contribution_score"] = out["contribution_entropy"] * (1 - out["top1_contribution_share"])
    out["contributor_depth_score"] = np.log1p(out["num_contributors"]) * np.log1p(out["median_contributions"])

    out["release_maturity_score"] = (
        out["stable_tag_ratio"] * out["semver_tag_ratio"] * out["latest_tag_is_stable"]
    )
    out["active_release_score"] = np.log1p(out["num_tags"]) * out["tag_release_velocity"]
    out["release_recency_score"] = 1 / (1 + out["deployment_recency_days"])
    out["release_quality_score"] = (
        out["release_maturity_score"]
        + out["release_recency_score"]
        + minmax_series(out["active_release_score"].fillna(0))
    ) / 3

    out["collaboration_event_score"] = (
        out["PullRequestEvent_ratio"] + out["IssueCommentEvent_ratio"] + out["IssuesEvent_ratio"]
    )
    out["activity_diversity_score"] = out["event_type_entropy"] * out["num_unique_event_types"]
    out["healthy_activity_score"] = (
        out["interaction_ratio"] + out["development_ratio"] + out["collaboration_event_score"]
    ) / 3
    out["non_external_activity_ratio"] = 1 - out["external_interest_event_ratio"]

    out["adoption_efficiency"] = np.log1p(out["stargazers_count"]) / np.log1p(out["repo_age_days"] + 1)
    out["issue_burden_score"] = out["open_issues_count"] / (np.log1p(out["stargazers_count"]) + eps)
    out["fork_interest_efficiency"] = np.log1p(out["forks_count"]) / np.log1p(out["repo_age_days"] + 1)

    out["governance_openness_score"] = (
        out["has_issues"]
        + out["has_projects"]
        + out["has_wiki"]
        + out["has_discussions"]
        + out["has_pull_requests"]
    ) / 5
    out["negative_repo_state"] = (out["archived"] + out["disabled"]).clip(0, 1)
    out["maintainer_activity_score"] = (
        out["is_recently_pushed_30d"]
        + out["is_recently_updated_90d"]
        + out["push_update_consistency"]
    ) / 3

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


DIMENSION_CONFIG = {
    "community_activity": {
        "label_ko": "커뮤니티 활성도",
        "core_question": "이 프로젝트는 현재 살아 움직이고 있는가?",
        "concepts": "Activity Volume, Responsiveness, Engagement Quality",
        "features": [
            "num_events", "num_unique_event_types", "event_type_entropy", "recent_event_density",
            "has_IssuesEvent", "has_PullRequestEvent", "has_IssueCommentEvent",
            "IssuesEvent_ratio", "IssueCommentEvent_ratio", "PullRequestEvent_ratio",
            "interaction_ratio", "development_ratio", "collaboration_event_score",
            "activity_diversity_score", "healthy_activity_score", "non_external_activity_ratio",
            "external_interest_event_ratio", "dominant_event_ratio",
        ],
    },
    "sustainability": {
        "label_ko": "지속 가능성",
        "core_question": "이 프로젝트는 앞으로도 유지될 수 있는가?",
        "concepts": "Contributor Structure, Diversity, Activity Stability",
        "features": [
            "num_contributors", "total_contributions", "median_contributions", "contribution_entropy",
            "top1_contribution_share", "top5_contribution_share", "contribution_gini",
            "top1_contribution_ratio", "top3_contribution_ratio", "contributors_to_stars_ratio",
            "bus_factor_risk", "distributed_contribution_score", "contributor_depth_score",
            "last_update_recency_days", "last_push_recency_days", "update_push_gap_days",
            "is_recently_pushed_30d", "is_recently_updated_90d", "is_stale_365d",
            "push_update_consistency", "freshness_score", "maintainer_activity_score",
        ],
    },
    "code_quality_reliability": {
        "label_ko": "코드 품질 및 신뢰성",
        "core_question": "이 프로젝트의 산출물은 믿을 수 있는가?",
        "concepts": "Engineering Practice, Defect Signals, Security Signals",
        "features": [
            "semver_tag_ratio", "stable_tag_ratio", "latest_tag_is_stable", "latest_tag_is_prerelease",
            "prerelease_tag_ratio", "release_maturity_score", "release_quality_score",
            "has_pull_requests", "PullRequestEvent_ratio", "development_ratio",
            "open_issues_to_stars_ratio", "issues_per_size", "issue_burden_score",
            "compiled_ratio", "language_entropy",
        ],
    },
    "legal_operational_governance": {
        "label_ko": "법적/운영 거버넌스",
        "core_question": "이 프로젝트는 조직적으로 안전하게 운영되는가?",
        "concepts": "Legal Compliance, Governance Structure",
        "features": [
            "has_issues", "has_projects", "has_downloads", "has_wiki", "has_pages",
            "has_discussions", "allow_forking", "has_pull_requests", "governance_openness_score",
            "archived", "disabled", "negative_repo_state",
        ],
    },
    "project_maturity": {
        "label_ko": "프로젝트 성숙도",
        "core_question": "이 프로젝트는 성숙한 운영 체계를 갖추었는가?",
        "concepts": "Release Engineering, Adoption/Popularity, Lifecycle/Scale",
        "features": [
            "num_tags", "num_major_versions", "num_minor_versions", "tag_release_velocity",
            "num_deployments", "has_deployments", "num_unique_refs", "tag_based_deployment_ratio",
            "deployment_recency_days", "release_recency_score", "active_release_score",
            "release_quality_score", "stargazers_count", "subscribers_count", "forks_count",
            "network_count", "stars_per_repo_age_day", "forks_per_repo_age_day", "stars_per_size",
            "forks_per_size", "adoption_efficiency", "fork_interest_efficiency", "repo_age_days", "repo_size",
        ],
    },
}

NEGATIVE_SCORE_FEATURES = {
    "top1_contribution_share", "top5_contribution_share", "contribution_gini",
    "top1_contribution_ratio", "top3_contribution_ratio", "bus_factor_risk",
    "deployment_recency_days", "prerelease_tag_ratio", "latest_tag_is_prerelease",
    "last_update_recency_days", "last_push_recency_days", "update_push_gap_days",
    "is_stale_365d", "open_issues_to_stars_ratio", "issues_per_size",
    "issue_burden_score", "dominant_event_ratio", "external_interest_event_ratio",
    "archived", "disabled", "negative_repo_state",
}


FEATURE_LABELS = {
    "num_events": "최근 활동량",
    "num_unique_event_types": "활동 유형 다양성",
    "event_type_entropy": "활동 분산도",
    "recent_event_density": "최근 활동 밀도",
    "has_IssuesEvent": "이슈 활동 존재",
    "has_PullRequestEvent": "Pull Request 활동 존재",
    "has_IssueCommentEvent": "이슈 댓글 활동 존재",
    "IssuesEvent_ratio": "이슈 생성 비율",
    "IssueCommentEvent_ratio": "이슈 댓글 비율",
    "PullRequestEvent_ratio": "Pull Request 비율",
    "interaction_ratio": "커뮤니티 상호작용 비율",
    "development_ratio": "개발 중심 활동 비율",
    "collaboration_event_score": "협업 이벤트 점수",
    "activity_diversity_score": "활동 다양성 점수",
    "healthy_activity_score": "건강한 활동 점수",
    "non_external_activity_ratio": "실제 내부 활동 비율",
    "external_interest_event_ratio": "외부 관심 이벤트 의존도",
    "dominant_event_ratio": "특정 이벤트 쏠림 정도",

    "num_contributors": "기여자 수",
    "total_contributions": "전체 기여량",
    "median_contributions": "기여자별 중앙 기여량",
    "contribution_entropy": "기여 분산도",
    "top1_contribution_share": "최상위 기여자 의존도",
    "top5_contribution_share": "상위 5명 기여자 의존도",
    "contribution_gini": "기여 불균형",
    "top1_contribution_ratio": "최상위 기여자 비중",
    "top3_contribution_ratio": "상위 3명 기여자 비중",
    "contributors_to_stars_ratio": "관심도 대비 기여자 비율",
    "bus_factor_risk": "Bus factor 위험도",
    "distributed_contribution_score": "분산 기여 점수",
    "contributor_depth_score": "기여자 기반 깊이",

    "last_update_recency_days": "마지막 업데이트 경과일",
    "last_push_recency_days": "마지막 push 경과일",
    "update_push_gap_days": "업데이트와 push 간격",
    "is_recently_pushed_30d": "최근 30일 내 push 여부",
    "is_recently_updated_90d": "최근 90일 내 업데이트 여부",
    "is_stale_365d": "1년 이상 비활성 여부",
    "push_update_consistency": "push와 update 일관성",
    "freshness_score": "최신성 점수",
    "maintainer_activity_score": "유지보수 활동 점수",

    "semver_tag_ratio": "Semantic version 사용률",
    "stable_tag_ratio": "안정 버전 태그 비율",
    "latest_tag_is_stable": "최신 태그 안정성",
    "latest_tag_is_prerelease": "최신 태그 pre-release 여부",
    "prerelease_tag_ratio": "pre-release 태그 비율",
    "release_maturity_score": "릴리즈 성숙도",
    "release_quality_score": "릴리즈 품질 점수",
    "num_tags": "릴리즈 태그 수",
    "num_major_versions": "major 버전 수",
    "num_minor_versions": "minor 버전 수",
    "tag_release_velocity": "릴리즈 속도",
    "num_deployments": "배포 기록 수",
    "has_deployments": "배포 기록 존재",
    "num_unique_refs": "배포 ref 다양성",
    "tag_based_deployment_ratio": "태그 기반 배포 비율",
    "deployment_recency_days": "마지막 배포 경과일",
    "release_recency_score": "릴리즈 최신성 점수",
    "active_release_score": "활발한 릴리즈 점수",

    "open_issues_to_stars_ratio": "관심도 대비 열린 이슈 부담",
    "issues_per_size": "코드 규모 대비 이슈 부담",
    "issue_burden_score": "이슈 부담 점수",
    "compiled_ratio": "컴파일 언어 비율",
    "language_entropy": "언어 구성 다양성",

    "has_issues": "이슈 기능 활성화",
    "has_projects": "프로젝트 관리 기능 활성화",
    "has_downloads": "다운로드 기능 활성화",
    "has_wiki": "위키 기능 활성화",
    "has_pages": "GitHub Pages 활성화",
    "has_discussions": "토론 기능 활성화",
    "allow_forking": "Fork 허용 여부",
    "has_pull_requests": "Pull Request 운영 여부",
    "governance_openness_score": "운영 개방성 점수",
    "archived": "아카이브 상태",
    "disabled": "비활성화 상태",
    "negative_repo_state": "부정적 repository 상태",

    "stargazers_count": "Star 수",
    "subscribers_count": "구독자 수",
    "forks_count": "Fork 수",
    "network_count": "네트워크 규모",
    "stars_per_repo_age_day": "기간 대비 Star 증가율",
    "forks_per_repo_age_day": "기간 대비 Fork 증가율",
    "stars_per_size": "코드 규모 대비 Star",
    "forks_per_size": "코드 규모 대비 Fork",
    "adoption_efficiency": "도입 효율성",
    "fork_interest_efficiency": "Fork 관심도 효율성",
    "repo_age_days": "프로젝트 운영 기간",
    "repo_size": "Repository 규모",
}

FEATURE_DESCRIPTIONS = {
    "num_events": "최근 GitHub 이벤트가 많아 프로젝트 활동량이 높게 관찰됩니다.",
    "num_unique_event_types": "이슈, PR, 댓글, push 등 여러 유형의 활동이 함께 나타납니다.",
    "event_type_entropy": "활동이 한 유형에만 치우치지 않고 비교적 다양하게 분포합니다.",
    "recent_event_density": "짧은 기간 안에 활동이 밀도 있게 발생하고 있습니다.",
    "has_IssuesEvent": "이슈 기반 논의가 관찰됩니다.",
    "has_PullRequestEvent": "Pull Request 기반 개발 흐름이 관찰됩니다.",
    "has_IssueCommentEvent": "이슈 댓글을 통한 상호작용이 관찰됩니다.",
    "IssueCommentEvent_ratio": "이슈 댓글 비중이 높아 커뮤니티 논의가 활발한 편입니다.",
    "PullRequestEvent_ratio": "Pull Request 활동 비중이 높아 협업 개발 흐름이 보입니다.",
    "interaction_ratio": "이슈와 댓글 중심의 상호작용이 활발합니다.",
    "development_ratio": "push와 PR 중심의 개발 활동이 유지되고 있습니다.",
    "collaboration_event_score": "이슈, 댓글, PR을 통한 협업 신호가 강합니다.",
    "activity_diversity_score": "활동량과 활동 다양성이 함께 높게 나타납니다.",
    "healthy_activity_score": "상호작용과 개발 활동을 종합했을 때 활동성이 양호합니다.",
    "non_external_activity_ratio": "단순 관심 이벤트보다 실제 개발/상호작용 활동 비중이 높습니다.",
    "external_interest_event_ratio": "Watch/Fork 같은 외부 관심 이벤트에 비해 실제 상호작용이 부족할 수 있습니다.",
    "dominant_event_ratio": "특정 이벤트 유형에 활동이 집중되어 다양성이 낮을 수 있습니다.",

    "num_contributors": "참여 contributor 수가 많아 유지보수 기반이 넓습니다.",
    "total_contributions": "누적 contribution 규모가 큽니다.",
    "median_contributions": "일반 contributor의 기여 수준도 일정하게 유지됩니다.",
    "contribution_entropy": "기여가 여러 contributor에게 분산되어 있습니다.",
    "top1_contribution_share": "최상위 contributor 한 명에게 기여가 집중될 수 있습니다.",
    "top5_contribution_share": "상위 소수 contributor에게 기여가 집중될 수 있습니다.",
    "contribution_gini": "기여 분포의 불균형이 커질 수 있습니다.",
    "top1_contribution_ratio": "최상위 contributor 의존도가 높을 수 있습니다.",
    "top3_contribution_ratio": "상위 3명 contributor 의존도가 높을 수 있습니다.",
    "contributors_to_stars_ratio": "외부 관심도에 비해 실제 contributor 기반이 약할 수 있습니다.",
    "bus_factor_risk": "소수 maintainer 의존으로 bus factor 위험이 있습니다.",
    "distributed_contribution_score": "기여가 비교적 넓게 분산되어 지속 가능성에 긍정적입니다.",
    "contributor_depth_score": "contributor 수와 기여 깊이가 함께 양호합니다.",

    "last_update_recency_days": "최근 업데이트가 오래되어 유지보수 신호가 약할 수 있습니다.",
    "last_push_recency_days": "최근 push가 오래되어 개발 활동이 정체되었을 수 있습니다.",
    "update_push_gap_days": "업데이트와 실제 push 흐름 사이의 간격이 큽니다.",
    "is_recently_pushed_30d": "최근 30일 내 push가 있어 최신 개발 활동이 확인됩니다.",
    "is_recently_updated_90d": "최근 90일 내 repository 업데이트가 확인됩니다.",
    "is_stale_365d": "1년 이상 주요 활동이 없을 가능성이 있습니다.",
    "push_update_consistency": "push와 update 흐름이 비교적 일관됩니다.",
    "freshness_score": "최근 update와 push 기준 최신성이 좋습니다.",
    "maintainer_activity_score": "최근 유지보수 활동 신호가 양호합니다.",

    "semver_tag_ratio": "semantic versioning을 잘 따르는 릴리즈 체계가 보입니다.",
    "stable_tag_ratio": "안정 버전 릴리즈 비중이 높습니다.",
    "latest_tag_is_stable": "최신 릴리즈가 안정 버전으로 보입니다.",
    "latest_tag_is_prerelease": "최신 릴리즈가 pre-release라 안정성 판단에 주의가 필요합니다.",
    "prerelease_tag_ratio": "pre-release 비중이 높아 안정 릴리즈와 구분해 볼 필요가 있습니다.",
    "release_maturity_score": "stable release와 semver 기반 릴리즈 성숙도가 좋습니다.",
    "release_quality_score": "릴리즈 안정성, 최신성, 활동성이 종합적으로 양호합니다.",
    "num_tags": "릴리즈 태그가 충분히 누적되어 있습니다.",
    "tag_release_velocity": "프로젝트 기간 대비 릴리즈가 꾸준히 발생합니다.",
    "num_deployments": "배포 기록이 충분히 관찰됩니다.",
    "has_deployments": "배포 활동이 확인됩니다.",
    "deployment_recency_days": "최근 배포가 오래되어 배포 최신성이 약할 수 있습니다.",
    "release_recency_score": "최근 릴리즈/배포 신호가 좋습니다.",
    "active_release_score": "릴리즈 규모와 속도를 함께 볼 때 활동성이 좋습니다.",

    "open_issues_to_stars_ratio": "관심도 대비 열린 이슈 부담이 큽니다.",
    "issues_per_size": "repository 규모 대비 이슈 부담이 큽니다.",
    "issue_burden_score": "열린 이슈 부담이 신뢰성 리스크로 작용할 수 있습니다.",
    "compiled_ratio": "컴파일 언어 비중이 있어 구조적 복잡도 또는 엔지니어링 성격이 강합니다.",
    "language_entropy": "언어 구성이 다양해 구조적 복잡도가 높을 수 있습니다.",

    "has_issues": "이슈 기반 운영이 가능하도록 열려 있습니다.",
    "has_projects": "프로젝트 관리 기능이 활성화되어 있습니다.",
    "has_downloads": "다운로드 기능이 활성화되어 있습니다.",
    "has_wiki": "위키 기반 문서화/운영 공간이 열려 있습니다.",
    "has_pages": "GitHub Pages 기반 공개 문서 또는 사이트 운영이 가능합니다.",
    "has_discussions": "커뮤니티 토론 기능이 활성화되어 있습니다.",
    "allow_forking": "외부 fork와 재사용이 허용됩니다.",
    "has_pull_requests": "Pull Request 기반 협업 운영이 가능합니다.",
    "governance_openness_score": "운영 기능들이 전반적으로 개방되어 있습니다.",
    "archived": "repository가 archived 상태일 수 있어 운영 지속성에 리스크가 있습니다.",
    "disabled": "repository가 disabled 상태일 수 있어 접근성과 운영에 리스크가 있습니다.",
    "negative_repo_state": "archived 또는 disabled 같은 부정적 상태 신호가 있습니다.",

    "stargazers_count": "Star 수가 많아 외부 관심도가 높습니다.",
    "subscribers_count": "구독자 수가 많아 지속 관찰하는 사용자가 많습니다.",
    "forks_count": "Fork 수가 많아 재사용과 확산 신호가 강합니다.",
    "network_count": "네트워크 규모가 커 ecosystem 확산 신호가 있습니다.",
    "stars_per_repo_age_day": "운영 기간 대비 Star 증가 속도가 좋습니다.",
    "forks_per_repo_age_day": "운영 기간 대비 Fork 증가 속도가 좋습니다.",
    "stars_per_size": "코드 규모 대비 관심도가 높습니다.",
    "forks_per_size": "코드 규모 대비 재사용 신호가 높습니다.",
    "adoption_efficiency": "프로젝트 기간 대비 adoption 신호가 좋습니다.",
    "fork_interest_efficiency": "프로젝트 기간 대비 fork 관심도가 좋습니다.",
    "repo_age_days": "프로젝트 운영 기간이 길어 lifecycle이 성숙한 편입니다.",
    "repo_size": "repository 규모가 커 프로젝트 축적도가 높습니다.",
}


def feature_display_name(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


def feature_explanation(feature: str, score: float | None = None) -> str:
    if feature in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[feature]
    if score is not None and score >= 70:
        return "reference dataset과 비교했을 때 긍정적인 신호로 나타납니다."
    if score is not None and score < 40:
        return "reference dataset과 비교했을 때 개선이 필요한 신호로 나타납니다."
    return "reference dataset 대비 중간 수준의 신호입니다."


def format_feature_insights(detail: pd.DataFrame, ascending: bool, n: int = 3):
    if detail.empty:
        return []
    selected = detail.sort_values("feature_score", ascending=ascending).head(n)
    items = []
    for row in selected.to_dict("records"):
        feature = row["feature"]
        score = float(row["feature_score"])
        items.append({
            "feature": feature,
            "label": feature_display_name(feature),
            "score": round(score, 2),
            "description": feature_explanation(feature, score),
        })
    return items


def load_reference_features() -> pd.DataFrame:
    reference = pd.read_csv(REFERENCE_DATA_PATH)
    reference = add_engineered_features(reference)
    for feature in MODEL_FEATURES:
        if feature not in reference.columns:
            reference[feature] = np.nan
    for feature in MODEL_FEATURES:
        reference[feature] = pd.to_numeric(reference[feature], errors="coerce")
    return reference


REFERENCE_DF = load_reference_features()


def percentile_score(value: Any, reference_values: pd.Series, higher_is_better: bool = True) -> float:
    values = pd.to_numeric(reference_values, errors="coerce").dropna().values.astype(float)
    if len(values) == 0 or pd.isna(value):
        return np.nan
    percentile = (values <= float(value)).mean()
    if not higher_is_better:
        percentile = 1 - percentile
    return float(np.clip(percentile * 100, 0, 100))


def score_to_grade(score: float) -> str:
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 55:
        return "Moderate"
    if score >= 40:
        return "Weak"
    return "Risk"


def parse_github_repo_url(repo_url_or_full_name: str) -> str:
    text = repo_url_or_full_name.strip()
    if text.startswith("http"):
        match = re.search(r"github\.com[:/]([^/]+)/([^/#?]+)", text)
        if not match:
            raise ValueError("Invalid GitHub repository URL")
        owner = match.group(1)
        repo = match.group(2).replace(".git", "")
        return f"{owner}/{repo}"
    if re.match(r"^[^/]+/[^/]+$", text):
        return text
    raise ValueError("Input must be a GitHub URL or owner/repo string")


def build_single_repo_features(repo_url_or_full_name: str) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    full_name = parse_github_repo_url(repo_url_or_full_name)
    raw_repo_df = build_repo_dataframe(full_name)
    feature_df = build_all_features(raw_repo_df)
    feature_df.insert(0, "repo_name", full_name)
    feature_df = add_engineered_features(feature_df)
    for feature in MODEL_FEATURES:
        if feature not in feature_df.columns:
            feature_df[feature] = np.nan
    return full_name, raw_repo_df, feature_df


def score_dimension(repo_features: pd.DataFrame, dimension_key: str) -> Tuple[float, pd.DataFrame]:
    config = DIMENSION_CONFIG[dimension_key]
    rows = []
    for feature in config["features"]:
        if feature not in repo_features.columns or feature not in REFERENCE_DF.columns:
            continue
        value = repo_features[feature].iloc[0]
        higher_is_better = feature not in NEGATIVE_SCORE_FEATURES
        score = percentile_score(value, REFERENCE_DF[feature], higher_is_better=higher_is_better)
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
    return float(np.clip(detail["feature_score"].mean(), 0, 100)), detail


def make_dimension_comment(score: float, dimension_key: str, detail: pd.DataFrame) -> Dict[str, Any]:
    config = DIMENSION_CONFIG[dimension_key]
    grade = score_to_grade(score)
    strength_features = format_feature_insights(detail, ascending=False, n=3)
    risk_features = format_feature_insights(detail, ascending=True, n=3)

    if grade in ["Excellent", "Good"]:
        summary = f"{config['label_ko']} 점수는 {score:.1f}점으로 양호하다."
    elif grade == "Moderate":
        summary = f"{config['label_ko']} 점수는 {score:.1f}점으로 중간 수준이다."
    else:
        summary = f"{config['label_ko']} 점수는 {score:.1f}점으로 개선이 필요하다."

    return {
        "dimension": dimension_key,
        "dimension_label": config["label_ko"],
        "score": float(score),
        "grade": grade,
        "core_question": config["core_question"],
        "concepts": config["concepts"],
        "summary": summary,
        "strength_features": strength_features,
        "risk_features": risk_features,
    }


def diagnose_repository(repo_url_or_full_name: str) -> Dict[str, Any]:
    full_name, raw_repo_df, repo_features = build_single_repo_features(repo_url_or_full_name)
    model_input = repo_features[MODEL_FEATURES].copy()
    for col in MODEL_FEATURES:
        model_input[col] = pd.to_numeric(model_input[col], errors="coerce")

    healthy_probability = float(MODEL.predict_proba(model_input)[0, 1])
    overall_score = healthy_probability * 100

    dimension_rows = []
    for dimension_key in DIMENSION_CONFIG:
        score, detail = score_dimension(repo_features, dimension_key)
        dimension_rows.append(make_dimension_comment(score, dimension_key, detail))

    return {
        "repo_name": full_name,
        "overall_score": round(overall_score, 2),
        "healthy_probability": round(healthy_probability, 4),
        "overall_grade": score_to_grade(overall_score),
        "model_name": MODEL_METADATA.get("best_model_name"),
        "target": MODEL_METADATA.get("target_col"),
        "dimension_scores": [
            {
                "dimension": row["dimension"],
                "label": row["dimension_label"],
                "score": round(row["score"], 2),
                "grade": row["grade"],
                "core_question": row["core_question"],
                "concepts": row["concepts"],
                "summary": row["summary"],
                "strength_features": row["strength_features"],
                "risk_features": row["risk_features"],
            }
            for row in dimension_rows
        ],
    }
