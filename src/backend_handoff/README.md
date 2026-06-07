# OSS Health Backend Handoff

이 폴더는 백엔드가 GitHub 오픈소스 저장소의 health score를 계산하고, data pipeline에서 promote된 최신 모델을 reload하기 위해 필요한 파일을 제공한다.

## 구성

```text
backend_handoff/
  models/
    oss_health_best_model.joblib
    oss_health_best_features.json
    oss_health_meta_model.joblib
    oss_health_meta_features.json
    oss_health_model_metadata.json
  data/
    reference_dataset.csv
    extract_github_repo.py
  features/
    build_features.py
    engineered_features.py
  inference/
    oss_health_diagnosis.py
  examples/
    smoke_test.py
    model_promoted_payload.example.json
  requirements.txt
  API_CONTRACT.md
  MODEL_UPDATE_CONTRACT.md
  .env.example
```

## 백엔드가 읽어야 하는 문서

- `API_CONTRACT.md`: 저장소 진단 API request/response 계약
- `MODEL_UPDATE_CONTRACT.md`: 모델 promotion webhook과 reload 계약

## 설치

```bash
cd src/backend_handoff
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

운영 환경에서는 GitHub API rate limit을 피하기 위해 token을 설정한다.

```bash
export GITHUB_TOKEN=ghp_your_token_here
```

## Smoke Test

```bash
python examples/smoke_test.py https://github.com/pandas-dev/pandas
```

## 진단 API에서 사용하는 함수

```python
from inference.oss_health_diagnosis import diagnose_repository

result = diagnose_repository("https://github.com/pandas-dev/pandas")
```

## 모델 reload 연동 요약

백엔드는 drift 감지 알림을 받는 것이 아니라, 최종적으로 검증된 모델이 promote된 뒤 `model_promoted` webhook을 받는다.

Webhook을 받으면 아래 경로의 모델 artifact를 reload하면 된다.

```text
src/backend_handoff/models/
```

필수 모델 artifact는 다음 5개다.

```text
oss_health_best_model.joblib
oss_health_best_features.json
oss_health_meta_model.joblib
oss_health_meta_features.json
oss_health_model_metadata.json
```

자세한 reload 계약은 `MODEL_UPDATE_CONTRACT.md`를 참고한다.

## 결과 의미

- `overall_score`: 최종 OSS health score. 0-100 범위.
- `healthy_probability`: base classifier의 healthy class probability.
- `dimension_scores`: reference dataset percentile 기반의 5개 진단 차원 점수.

5개 진단 차원은 다음과 같다.

1. 커뮤니티 활성도
2. 지속 가능성
3. 코드 품질 및 신뢰성
4. 법적/운영 거버넌스
5. 프로젝트 성숙도

## 운영 주의사항

- 백엔드는 `models/oss_health_best_features.json`의 feature 순서를 반드시 유지해야 한다.
- 모델 artifact는 sklearn-compatible pipeline이다.
- 모델 파일은 5개를 한 세트로 취급한다. 일부 파일만 교체하면 안 된다.
- reload 실패 시 기존 메모리 모델을 유지해야 한다.
- 반복 조회되는 repository feature extraction 결과는 cache하는 것이 좋다.

## 사용자 표시용 feature 설명

API response의 strength/risk item은 내부 feature key와 사용자 표시용 설명을 함께 제공한다.

- `feature`: 내부 feature key
- `label`: 한국어 표시명
- `score`: reference percentile score
- `description`: 사용자용 해석 문장
