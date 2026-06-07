from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class BackendAlertResult:
    sent: bool
    event_type: str
    model_version: str
    webhook_url: str | None
    status_code: int | None
    payload_path: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def send_backend_model_promoted_alert(
    webhook_url: str | None,
    model_version: str,
    trigger_reason: str,
    champion_path: str,
    metadata_path: str,
    backend_handoff_models_path: str | None,
    payload_path: str,
    timeout_seconds: int = 10,
) -> BackendAlertResult:
    payload = {
        "event_type": "model_promoted",
        "project": "oss-health",
        "model_version": model_version,
        "trigger_reason": trigger_reason,
        "champion_path": champion_path,
        "metadata_path": metadata_path,
        "backend_handoff_models_path": backend_handoff_models_path,
        "created_at": utc_now(),
        "producer": "data-pipeline",
        "consumer": "backend",
    }

    save_json(payload, payload_path)

    if not webhook_url:
        return BackendAlertResult(
            sent=False,
            event_type="model_promoted",
            model_version=model_version,
            webhook_url=None,
            status_code=None,
            payload_path=str(payload_path),
        )

    request = urllib.request.Request(
        webhook_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        status_code = response.status

    if status_code >= 300:
        raise RuntimeError(f"Backend webhook failed with status={status_code}")

    return BackendAlertResult(
        sent=True,
        event_type="model_promoted",
        model_version=model_version,
        webhook_url=webhook_url,
        status_code=status_code,
        payload_path=str(payload_path),
    )