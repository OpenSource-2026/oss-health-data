# active reference 평가/drop/replace

from __future__ import annotations
import pandas as pd
from five_das import calculate_reference_5das_snapshot
from pathlib import Path
import json
from dataclasses import asdict, dataclass


class ReferenceUpdateResult:
    reference_changed: bool
    lower_bound: float
    retired_repos: list[str]
    promoted_repos: list[str]
    active_reference_path: str
    candidate_pool_path: str
    output_reference_path: str

def find_below(
    active: pd.DataFrame,
    lower_bound: float,
)-> pd.DataFrame:
    out = active.copy()

    if "below_bound_count" not in out.columns:
        out["below_bound_count"] = 0

    below = out["five_das"] < lower_bound
    out.loc[below, "below_bound_count"] += 1
    out.loc[~below, "below_bound_count"] = 0
    # 3번 연속 Lower bound 미만이면 retire 후보
    return out

def select_retired_repos(
    active: pd.DataFrame,
    max_retire_ratio: float = 0.10,
    hysteresis_count: int=3,
) -> list[str]:
    retire_candidates = active[active["below_bound_count"] >= hysteresis_count]
    retire_candidates = retire_candidates.sort_values("five_das", ascending=True)
    max_retire = int(len(active)*max_retire_ratio)
    retire_candidates = retire_candidates.head(max_retire)
    return retire_candidates["repo_name"].tolist()
    
def promote_from_candidate_pool(
        active: pd.DataFrame,
        candidate_pool: pd.DataFrame,
        retired_repos: list[str],
) -> pd.DataFrame: 
    if not retired_repos:
        return active, candidate_pool, []
    active_after_drop = active[~active["repo_name"].isin(retired_repos)].copy()
    needed=len(retired_repos)
    candidate_pool = candidate_pool.sort_value("five_das", ascending=False)
    promoted = candidate_pool.head(needed)
    remaining_pool = candidate_pool.iloc[needed:].copy()
    promoted["below_bound_count"]=0
    next_active = pd.concat([active_after_drop, promoted], ignore_index=True)
    return next_active, remaining_pool, promoted["repo_name"].tolist()

def run_reference_update(
    active_reference_path: str,
    candidate_pool_path: str,
    reference_feature_path: str,
    output_reference_path: str,
    output_candidate_pool_path: str,
    output_report_path: str,
) -> ReferenceUpdateResult:

    candidate_pool = pd.read_csv(candidate_pool_path)
    active_with_5das, lower_bound = calculate_reference_5das_snapshot(active_reference_path, reference_feature_path)
    active_scored = find_below(active_scored, lower_bound)
    retired_repos = select_retired_repos(active_scored)
    if candidate_pool.empty and retired_repos:
        raise ValueError("no candidate to replace")
    next_active, next_candidate_pool, promoted_repos = promote_from_candidate_pool(
        active=active_scored,
        candidate_pool=candidate_pool,
        retired_repos=retired_repos,
    )
    Path(output_reference_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_candidate_pool_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_report_path).parent.mkdir(parents=True, exist_ok=True)

    next_active.to_csv(output_reference_path, index=False)
    next_candidate_pool.to_csv(output_candidate_pool_path, index=False)

    result = ReferenceUpdateResult(
        reference_changed=bool(retired_repos),
        lower_bound=lower_bound,
        retired_repos=retired_repos,
        promoted_repos=promoted_repos,
        active_reference_path=active_reference_path,
        candidate_pool_path=candidate_pool_path,
        output_reference_path=output_reference_path,
    )

    with open(output_report_path, "w", enxoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)
    return result