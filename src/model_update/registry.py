# model/reference registry
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def load_registry(path: str) -> dict[str, Any]:
    registry_path = Path(path)
    if not registry_path.exists():
        return {
            "production_version": None,
            "production_reference": None,
            "updated_at": None,
            "models": {},
            "references": {},
        }
    with open(registry_path, encoding="utf-8") as f:
        return json.load(f)

def save_registry(registry: dict[str, Any], path: str) -> None:
    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)

def promote_model(
    registry_path: str,
    model_version: str,
    model_path: str,
    features_path: str,
    metadata_path: str,
    reference_version: str,
    reference_path: str,
    metrics: dict[str, float],
) -> dict[str, Any]:
    registry = load_registry(registry_path)

    previous = registry.get("production_version")
    if previous and previous in registry["models"]:
        registry["models"][previous]["status"] = "archived"

    registry["models"][model_version] = {
        "version": model_version,
        "status": "production",
        "model_path": model_path,
        "features_path": features_path,
        "metadata_path": metadata_path,
        "reference_version": reference_version,
        "metrics": metrics,
        "promoted_at": utc_now(),
    }
    registry["references"][reference_version] = {
        "version": reference_version,
        "path": reference_path,
        "status": "production",
        "promoted_at": utc_now()
    }
    registry["production_version"] = model_version
    registry["production_reference"] = reference_version
    registry["updated_at"] = utc_now()

    save_registry(registry, registry_path)
    return registry

def write_backend_handoff_event(
    output_path: str,
    registry_path: str,
    model_version: str, 
    reference_version: str,
) -> None:
    event = {
        "event": "model_promoted",
        "model_version": model_version,
        "reference_version": reference_version,
        "registry_path": registry_path,
        "created_at": utc_now(),
        "owner": "data-pipeline",
        "consumer": "backend",
        "note": "Backend reload implementation is handled by backend team.",
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(event, f, ensure_ascii=False, indent=2)