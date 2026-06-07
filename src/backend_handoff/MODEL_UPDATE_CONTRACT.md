# OSS Health 모델 업데이트 연동 계약서

이 문서는 백엔드가 OSS Health 모델 업데이트 파이프라인과 연동할 때 필요한 계약을 정리한다.

데이터 파이프라인은 reference 관리, drift 탐지, challenger 재학습, champion-challenger 평가, 모델 promotion까지 담당한다.
백엔드는 모델을 직접 학습하지 않는다. 백엔드는 `model_promoted` 이벤트를 받은 뒤, 지정된 모델 artifact를 다시 로드하면 된다.

## 1. 백엔드가 사용해야 하는 모델 위치

백엔드에서 inference에 사용할 최신 모델 artifact는 아래 경로에 저장된다.

```text
src/backend_handoff/models/
```

필수 파일은 다음 5개다.

```text
oss_health_best_model.joblib
oss_health_best_features.json
oss_health_meta_model.joblib
oss_health_meta_features.json
oss_health_model_metadata.json
```

각 파일의 의미는 다음과 같다.

| 파일 | 설명 |
| --- | --- |
| `oss_health_best_model.joblib` | healthy probability를 예측하는 base model. 현재 best model 구조는 Logistic Regression pipeline이다. |
| `oss_health_best_features.json` | base model 입력 feature 순서. 백엔드는 반드시 이 순서를 유지해야 한다. |
| `oss_health_meta_model.joblib` | base model logit과 5개 dimension score를 받아 최종 health score를 보정하는 Ridge meta model. |
| `oss_health_meta_features.json` | meta model 입력 feature 순서. |
| `oss_health_model_metadata.json` | 모델 버전, 학습 기준, metric, feature 정보 등 metadata. |

## 2. 모델 업데이트 이벤트

백엔드는 drift 감지 이벤트를 받지 않는다.

백엔드가 받는 이벤트는 challenger가 평가를 통과하고 실제 champion으로 승격된 뒤 발생하는 `model_promoted` 이벤트다.

즉 백엔드 reload 조건은 다음과 같다.

```text
reference update 또는 feature drift 감지
-> challenger 모델 재학습
-> champion-challenger evaluation
-> challenger가 기준 통과
-> model promotion
-> backend에 model_promoted webhook 전송
-> backend가 최신 모델 reload
```

## 3. Webhook endpoint

백엔드는 아래와 같은 내부 endpoint를 제공하면 된다.

```text
POST /internal/model-promoted
```

endpoint 이름은 백엔드 구현에 맞게 바꿀 수 있다. 단, Airflow/Data pipeline의 `oss_health_backend_webhook_url` 변수와 일치해야 한다.

## 4. Webhook payload

예시 payload는 다음과 같다.

```json
{
  "event_type": "model_promoted",
  "project": "oss-health",
  "model_version": "20260608_031522",
  "trigger_reason": "feature_drift",
  "champion_path": "src/model_registry/champion",
  "metadata_path": "src/model_registry/champion/model_version.json",
  "backend_handoff_models_path": "src/backend_handoff/models",
  "created_at": "2026-06-08T03:15:22+00:00",
  "producer": "data-pipeline",
  "consumer": "backend"
}
```

필드 설명은 다음과 같다.

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `event_type` | string | 항상 `model_promoted`. |
| `project` | string | 프로젝트 식별자. |
| `model_version` | string | 새로 promote된 모델 버전. UTC 기준 `YYYYMMDD_HHMMSS` 형식. |
| `trigger_reason` | string | 모델 업데이트 원인. `reference_changed` 또는 `feature_drift`. |
| `champion_path` | string | data pipeline 내부 champion registry 경로. |
| `metadata_path` | string | promotion metadata 경로. |
| `backend_handoff_models_path` | string | 백엔드가 reload해야 하는 모델 artifact 경로. |
| `created_at` | string | 이벤트 생성 시각. |
| `producer` | string | 이벤트 생산자. 항상 `data-pipeline`. |
| `consumer` | string | 이벤트 소비자. 항상 `backend`. |

## 5. 백엔드 처리 로직

백엔드는 webhook을 받으면 다음 순서로 처리한다.

1. `event_type == "model_promoted"`인지 확인한다.
2. payload의 `model_version`을 현재 메모리에 로드된 모델 버전과 비교한다.
3. 새 버전이면 `backend_handoff_models_path`에서 모델 파일 5개를 다시 로드한다.
4. reload에 성공하면 이후 inference 요청부터 새 모델을 사용한다.
5. reload에 실패하면 기존 메모리 모델을 유지하고 에러 로그를 남긴다.

백엔드 처리 예시는 다음과 같다.

```python
from pathlib import Path
import json
import joblib

_loaded_model_version = None
_loaded_artifacts = None


def load_oss_health_artifacts(models_dir: str):
    models_path = Path(models_dir)

    with open(models_path / "oss_health_best_features.json", encoding="utf-8") as f:
        base_features = json.load(f)

    with open(models_path / "oss_health_meta_features.json", encoding="utf-8") as f:
        meta_features = json.load(f)

    with open(models_path / "oss_health_model_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    return {
        "base_model": joblib.load(models_path / "oss_health_best_model.joblib"),
        "base_features": base_features,
        "meta_model": joblib.load(models_path / "oss_health_meta_model.joblib"),
        "meta_features": meta_features,
        "metadata": metadata,
    }


def handle_model_promoted_event(payload: dict):
    global _loaded_model_version, _loaded_artifacts

    if payload.get("event_type") != "model_promoted":
        return {"reloaded": False, "reason": "ignored_event_type"}

    next_version = payload["model_version"]
    if next_version == _loaded_model_version:
        return {"reloaded": False, "reason": "same_model_version"}

    models_dir = payload["backend_handoff_models_path"]
    next_artifacts = load_oss_health_artifacts(models_dir)

    _loaded_artifacts = next_artifacts
    _loaded_model_version = next_version

    return {"reloaded": True, "model_version": next_version}
```

실제 백엔드에서는 전역 변수 대신 application state, dependency container, cache layer 등을 사용할 수 있다.

## 6. 장애 처리 원칙

모델 reload 실패 시 백엔드는 기존 모델을 계속 사용해야 한다.

권장 정책은 다음과 같다.

```text
새 모델 로드 성공 -> 새 모델로 교체
새 모델 로드 실패 -> 기존 모델 유지 + 에러 로그/모니터링
```

모델 파일 중 하나라도 없거나 로드에 실패하면 부분 교체하지 않는다.

## 7. 백엔드가 하지 않아도 되는 일

백엔드는 아래 작업을 하지 않는다.

- drift 탐지
- reference dataset 교체 판단
- candidate reference 검색
- 모델 재학습
- champion-challenger evaluation
- model promotion 판단

위 작업은 모두 data pipeline 책임이다.

백엔드 책임은 `model_promoted` 이벤트를 받은 뒤 최신 artifact를 안전하게 reload하는 것이다.

## 8. Airflow 변수

Data pipeline에서 webhook을 보내려면 Airflow에 아래 변수가 설정되어야 한다.

```bash
airflow variables set oss_health_project_root /Users/carolyn/Desktop/opensource/data/oss-health-data
airflow variables set oss_health_backend_webhook_url http://localhost:8080/internal/model-promoted
```

백엔드 endpoint가 바뀌면 `oss_health_backend_webhook_url`만 수정하면 된다.
