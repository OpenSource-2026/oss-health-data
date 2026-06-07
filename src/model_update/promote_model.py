from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


MODEL_ARTIFACTS = [
    "oss_health_best_model.joblib",
    "oss_health_best_features.json",
    "oss_health_meta_model.joblib",
    "oss_health_meta_features.json",
    "oss_health_model_metadata.json",
]

@dataclass
class PromotionResult:
    promoted: bool
    model_version: str
    trigger_reason: str
    champion_path: str
    archived_champion_path: str | None
    backend_handoff_models_path: str | None
    metadata_path: str

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(project_root: str | Path, path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(project_root) / path

def copy_file_set(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    for filename in MODEL_ARTIFACTS:
        source = source_dir / filename
        if not source.exists():
            raise FileNotFoundError(f"Missing model artifact: {source}")

        shutil.copy2(source, target_dir / filename)

def archive_existing_champion(
    champion_path: Path,
    archive_root: Path,
    model_version: str,
) -> str | None:
    if not champion_path.exists():
        return None

    archived_path = archive_root / f"champion_before_{model_version}"

    if archived_path.exists():
        shutil.rmtree(archived_path)

    archived_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(champion_path, archived_path)

    return str(archived_path)

def promote_challenger_to_champion(
    project_root: str,
    challenger_path: str,
    champion_path: str,
    archive_root: str,
    model_version: str,
    trigger_reason: str,
    evaluation_report_path: str,
    backend_handoff_models_path: str | None = None,
) -> PromotionResult:
    project_root_path = Path(project_root)

    challenger = resolve_path(project_root_path, challenger_path)
    champion = resolve_path(project_root_path, champion_path)
    archive = resolve_path(project_root_path, archive_root) 

    if not challenger.exists():
        raise FileNotFoundError(f"Challenger path does not exist: {challenger}")

    archived_champion_path = archive_existing_champion(
        champion_path=champion,
        archive_root=archive,
        model_version=model_version,
    )

    if champion.exists():
        shutil.rmtree(champion)
    
    shutil.copytree(challenger, champion)

    metadata_path = champion / "model_version.json"
    promotion_metadata = {
        "event_type": "model_promoted",
        "model_version": model_version,
        "trigger_reason": trigger_reason,
        "promoted_at": utc_now(),
        "champion_path": str(champion),
        "source_challenger_path": str(challenger),
        "archived_champion_path": archived_champion_path,
        "evaluation_report_path": evaluation_report_path,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(promotion_metadata, f, ensure_ascii=False, indent=2)

    resolved_backend_handoff_path = None

    if backend_handoff_models_path:
        backend_target = resolve_path(project_root_path, backend_handoff_models_path)
        copy_file_set(champion, backend_target)
        resolved_backend_handoff_path = str(backend_target)

    return PromotionResult(
        promoted=True,
        model_version=model_version,
        trigger_reason=trigger_reason,
        champion_path=str(champion),
        archived_champion_path=archived_champion_path,
        backend_handoff_models_path=resolved_backend_handoff_path,
        metadata_path=str(metadata_path),
    )
