# OSS Health Data Pipeline

GitHub 오픈소스 repository의 운영 건강성(OSS Health)을 정량적으로 평가하기 위한 데이터 수집, feature engineering, scoring, 모델 학습, drift 기반 모델 업데이트, backend handoff 파이프라인이다.

이 레포는 단순히 star 수나 fork 수가 많은 repository를 좋은 프로젝트로 판단하지 않는다. 오픈소스 프로젝트의 건강성을 커뮤니티 활동, contributor 지속 가능성, release 성숙도, 운영 거버넌스, 프로젝트 규모와 생애주기 등 여러 신호를 결합해 평가한다.

최종 목표는 다음 두 가지다.

1. GitHub repository URL을 입력하면 OSS Health Score와 5개 진단 차원별 해석을 제공한다.
2. 시간이 지나면서 reference dataset이나 feature 분포가 변하면, Airflow 기반 파이프라인이 이를 감지하고 검증된 challenger 모델만 champion으로 승격한 뒤 backend에 reload 이벤트를 전달한다.

## 핵심 요약

이 프로젝트의 핵심 설계는 다음과 같다.

```text
GitHub data 수집
-> feature engineering
-> pseudo health target 생성
-> base model 학습
-> 5개 dimension score 계산
-> Ridge meta model 기반 final score 보정
-> repository diagnosis
-> reference/drift monitoring
-> challenger 재학습
-> champion-challenger evaluation
-> model promotion
-> backend handoff
```

현재 모델 구조는 다음과 같다.

| 단계 | 모델/로직 | 역할 |
| --- | --- | --- |
| Base model | Logistic Regression pipeline | repository feature로 `new_label` healthy 여부를 예측 |
| Dimension scoring | percentile 기반 scoring | 5개 평가 차원별 설명 가능한 점수 산출 |
| Meta model | Ridge Regression pipeline | base probability logit과 dimension score를 결합해 최종 OSS health score 보정 |
| Update pipeline | Airflow + drift/reference monitoring | reference 품질과 feature drift를 감지하고 검증된 모델만 업데이트 |

## 평가 차원

OSS Health는 5개 평가 차원으로 설명된다.

| 평가 차원 | 핵심 질문 | 주요 신호 |
| --- | --- | --- |
| 커뮤니티 활성도 | 이 프로젝트는 현재 살아 움직이고 있는가? | event 수, issue/PR/comment 활동, interaction ratio, activity diversity |
| 지속 가능성 | 이 프로젝트는 앞으로도 유지될 수 있는가? | contributor 수, contribution 분산, bus factor risk, maintainer activity |
| 코드 품질 및 신뢰성 | 이 프로젝트의 산출물은 믿을 수 있는가? | semantic versioning, stable release, PR activity, issue burden |
| 법적/운영 거버넌스 | 이 프로젝트는 조직적으로 안전하게 운영되는가? | issues/projects/wiki/discussions 활성화, archived/disabled 상태, governance openness |
| 프로젝트 성숙도 | 이 프로젝트는 성숙한 운영 체계를 갖추었는가? | tag/release 수, deployment, stars/forks, repo age, repo size |

일부 품질 지표는 GitHub metadata만으로 직접 관측하기 어렵다. 예를 들어 test coverage, CI pass rate, vulnerability advisory, CLA, code of conduct 등은 현재 dataset에 직접 포함되어 있지 않다. 따라서 이 프로젝트는 release practice, issue burden, PR activity, repository option, archived/disabled 상태와 같은 관측 가능한 proxy feature를 사용한다.

## 전체 디렉토리 구조

```text
oss-health-data/
├── README.md
├── requirements.txt
├── pyproject.toml
├── dags/
│   └── oss_health_update_dag.py
├── data/
│   ├── FEATURE/
│   ├── dags/
│   └── repo/
├── data_extraction/
└── src/
    ├── 0_DATASET.ipynb
    ├── 1_FEATURE.ipynb
    ├── 2_MODEL.ipynb
    ├── 3_REPOSITORY_DIAGNOSIS.ipynb
    ├── train_meta_model.py
    ├── data/
    │   ├── extract_github_repo.py
    │   └── main/dataset.csv
    ├── features/
    │   └── build_features.py
    ├── scoring/
    │   ├── dimension_scores.py
    │   └── pseudo_labels.py
    ├── models/
    │   ├── oss_health_best_model.joblib
    │   ├── oss_health_best_features.json
    │   ├── oss_health_meta_model.joblib
    │   ├── oss_health_meta_features.json
    │   └── oss_health_model_metadata.json
    ├── monitoring/
    │   ├── drift_detector.py
    │   └── drift_thresholds.py
    ├── monitoring_store/
    │   ├── drift_history.csv
    │   └── drift_thresholds.json
    ├── reference/
    │   ├── candidate_search.py
    │   ├── five_das.py
    │   └── reference_manager.py
    ├── reference_store/
    │   ├── active/
    │   ├── candidates/
    │   └── reports/
    ├── model_update/
    │   ├── backend_alert.py
    │   ├── evaluate_challenger.py
    │   ├── promote_model.py
    │   ├── registry.py
    │   ├── run_update_pipeline.py
    │   └── train_challenger.py
    ├── outputs/
    │   ├── 2_model/
    │   ├── 3_repository_diagnosis/
    │   └── model_update/
    └── backend_handoff/
        ├── README.md
        ├── API_CONTRACT.md
        ├── MODEL_UPDATE_CONTRACT.md
        ├── models/
        ├── inference/
        ├── features/
        ├── data/
        └── examples/
```

## 데이터 생성 흐름

### 1. Seed dataset 구성

초기 dataset은 `src/0_DATASET.ipynb`에서 구성된다.

healthy repository는 오픈소스 생태계에서 널리 사용되고, 활발히 유지되며, 커뮤니티와 release 운영이 비교적 안정적인 repository를 중심으로 구성한다. unhealthy repository는 archived, deprecated, stale, tutorial, awesome list, boilerplate, old ecosystem 등 운영 건강성이 낮거나 일반적인 제품형 오픈소스 프로젝트와 성격이 다른 repository를 포함한다.

초기 `label`은 사람이 구성한 seed 기준이다. 다만 이 label만으로 최종 모델을 바로 학습하지 않고, 이후 feature-family score와 dimension score 기반 pseudo scoring framework를 통해 더 일관적인 target을 만든다.

### 2. Feature engineering

`src/features/build_features.py`와 `src/backend_handoff/features/engineered_features.py`에서 GitHub API 기반 feature를 생성한다.

주요 feature family는 다음과 같다.

- contributor features
- deployment features
- event/activity features
- interest/popularity features
- language structure features
- tag/release features
- repository metadata features
- additional ratio/entropy/recency features

예시 feature는 다음과 같다.

```text
num_contributors
top1_contribution_share
contribution_gini
num_events
event_type_entropy
PullRequestEvent_ratio
stargazers_count
forks_count
language_entropy
repo_age_days
last_push_recency_days
semver_tag_ratio
release_quality_score
governance_openness_score
maintainer_activity_score
```

## Target 설계

이 프로젝트는 객관적 정답 label이 존재하기 어려운 OSS health 문제를 다룬다. 따라서 target은 단순 수동 label이 아니라 pseudo scoring framework로 생성한다.

### 1. 7개 feature-family score

먼저 feature를 7개 family로 나누고 각 family별 score를 만든다.

```text
community_activity_score
contributor_sustainability_score
release_engineering_score
popularity_adoption_score
language_structure_score
maintenance_score
governance_score
```

각 feature는 0-1 min-max scaling을 적용한다. 건강하지 않은 신호는 방향을 반전한다. 예를 들어 `last_push_recency_days`, `archived`, `disabled`, `top1_contribution_share`, `contribution_gini`처럼 값이 클수록 위험한 feature는 `1 - scaled_value`로 처리한다.

그 다음 family별 feature score 평균을 내고, 최종적으로 7개 family score의 평균을 `oss_health_score`로 정의한다.

```text
oss_health_score = mean(7 feature-family scores)
```

### 2. new_label 생성

`oss_health_score`의 median을 threshold로 사용해 binary label을 만든다.

```text
new_label = 1 if oss_health_score >= median(oss_health_score)
new_label = 0 otherwise
```

이 방식은 임의의 절대 기준을 두기보다, 현재 reference universe 안에서 상대적으로 건강한 repository와 그렇지 않은 repository를 나누는 방식이다.

### 3. 왜 pseudo target인가

OSS health는 정답 label이 명확히 존재하지 않는다. 따라서 이 프로젝트는 다음 논리를 사용한다.

- seed dataset은 사람이 해석 가능한 기준으로 구성한다.
- feature-family score는 GitHub metadata에서 관측 가능한 운영 건강성 신호를 반영한다.
- median threshold는 dataset 내부 분포를 기준으로 binary target을 만든다.
- 모델은 이 pseudo target을 학습해 새로운 repository에 대해 일관된 scoring을 제공한다.

즉 이 모델은 절대적 진실을 맞히는 classifier가 아니라, 해석 가능한 OSS health scoring framework를 학습한 모델이다.

## 모델링 구조

### 1. Base model

Base model은 `new_label`을 예측한다.

현재 best model은 Logistic Regression pipeline이다.

```text
SimpleImputer(strategy="median")
-> StandardScaler
-> LogisticRegression(C=1.2357, penalty="l2", class_weight="balanced", solver="liblinear")
```

Logistic Regression을 선택한 이유는 다음과 같다.

- tabular feature와 잘 맞는다.
- regularized linear classifier라 작은 dataset에서 과적합 위험이 낮다.
- coefficient 기반 해석이 가능하다.
- feature contribution을 설명하기 쉽다.
- 여러 모델과 비교했을 때 cross-validation ROC-AUC와 holdout ROC-AUC가 안정적이었다.

현재 metadata 기준 주요 성능은 다음과 같다.

```text
holdout accuracy : 0.9398
holdout precision: 0.9512
holdout recall   : 0.9286
holdout f1       : 0.9398
holdout roc_auc  : 0.9820
cv roc_auc mean  : 0.9919
```

### 2. 5개 dimension score

최종 사용자에게 보여주는 차원별 점수는 `src/scoring/dimension_scores.py`에서 계산한다.

각 repository feature를 reference dataset의 percentile 기준으로 변환하고, dimension별 관련 feature score를 평균낸다.

```text
community_activity_score
sustainability_score
code_quality_reliability_score
legal_operational_governance_score
project_maturity_score
```

이 점수는 모델 probability가 아니라 reference percentile 기반의 설명 가능한 진단 점수다.

### 3. Meta model

Meta model은 최종 OSS health score를 보정한다.

입력은 다음과 같다.

```text
base model healthy probability를 logit으로 변환한 값
+ 5개 dimension score
```

정확히는 다음 feature를 사용한다.

```text
raw_model_logit
community_activity_score
sustainability_score
code_quality_reliability_score
legal_operational_governance_score
project_maturity_score
```

Target은 `oss_health_score`다.

Meta model은 Ridge Regression을 사용한다.

Ridge를 선택한 이유는 다음과 같다.

- 최종 점수는 연속값이므로 regression이 적합하다.
- 입력 feature 수가 적어 복잡한 모델이 필요하지 않다.
- dimension score 간 상관관계가 있을 수 있어 L2 regularization이 안정적이다.
- linear model이라 어떤 차원이 최종 점수에 얼마나 기여했는지 설명 가능하다.
- 작은 dataset에서 tree/boosting meta model보다 과적합 위험이 낮다.

현재 meta model metric은 다음과 같다.

```text
mae : 0.0154
rmse: 0.0212
r2  : 0.8956
```

## Repository diagnosis 흐름

특정 repository를 진단할 때의 흐름은 다음과 같다.

```text
repository URL 입력
-> GitHub API data 수집
-> raw feature 생성
-> engineered feature 생성
-> base model healthy probability 계산
-> 5개 dimension score 계산
-> meta model final score 계산
-> strength/risk feature 해석
-> JSON response 생성
```

Backend에서 사용할 수 있는 inference 함수는 다음과 같다.

```python
from backend_handoff.inference.oss_health_diagnosis import diagnose_repository

result = diagnose_repository("https://github.com/pandas-dev/pandas")
```

Smoke test는 다음과 같이 실행한다.

```bash
cd src/backend_handoff
python examples/smoke_test.py https://github.com/pandas-dev/pandas
```

## Reference 관리와 5DAS

모델 업데이트 파이프라인에서는 reference dataset의 품질을 유지하기 위해 5DAS를 사용한다.

5DAS는 5 Dimension Average Score의 약자다.

```text
5DAS = mean(
  community_activity,
  sustainability,
  code_quality_reliability,
  legal_operational_governance,
  project_maturity
)
```

Reference update 로직은 `src/reference/reference_manager.py`에서 관리한다.

흐름은 다음과 같다.

```text
active reference 재평가
-> 각 reference repo의 5DAS 계산
-> active reference 5DAS의 Q10을 lower bound로 설정
-> lower bound 미만 repo는 below_bound_count 증가
-> 3회 연속 미달 시 retire 후보
-> retire cap 적용
-> candidate pool에서 5DAS 높은 repo를 promote
-> active reference와 candidate pool 상태 저장
```

Q10 lower bound를 사용하는 이유는 다음과 같다.

- minimum은 outlier 하나에 지나치게 민감하다.
- median은 기준이 너무 높아 정상 reference까지 과도하게 제거할 수 있다.
- Q10은 reference set 하위 품질을 관리하면서도 일시적 변동에 덜 민감하다.

3회 hysteresis를 두는 이유는 다음과 같다.

- GitHub 활동 지표는 주기적으로 흔들릴 수 있다.
- 한 번 낮아졌다고 바로 reference에서 제거하면 false positive가 증가한다.
- 3회 연속 미달일 때만 제거하면 일시적 변동과 구조적 품질 하락을 구분할 수 있다.

Candidate pool은 Kubernetes replica처럼 active reference가 빠졌을 때 즉시 대체 가능한 standby reference 역할을 한다.

## Drift 탐지

Feature drift는 `src/monitoring/drift_detector.py`에서 감지한다.

비교 대상은 다음과 같다.

```text
reference: 기존 학습 feature dataset
current  : 최신 수집/current batch feature dataset
```

Drift 대상 feature는 base model 입력 feature인 `src/models/oss_health_best_features.json`이다.

feature type별 drift 계산 방식은 다음과 같다.

| feature type | method | 설명 |
| --- | --- | --- |
| binary feature | binary ratio difference | reference와 current의 1 비율 차이 |
| numeric feature | PSI | reference 분포 bin 기준 current 분포 변화 측정 |

Threshold는 고정값만 사용하지 않는다. `src/monitoring/drift_thresholds.py`에서 drift history가 충분히 쌓이면 과거 drift score의 quantile을 기준으로 dynamic threshold를 계산한다.

Cold start 구간에서는 기본 threshold를 사용한다.

```text
overall_threshold = 0.15
drifted_ratio_threshold = 0.30
high_count_threshold = 5
```

history가 충분히 쌓이면 Q95 기반 threshold를 사용한다.

## 모델 업데이트 파이프라인

모델 업데이트는 `src/model_update/run_update_pipeline.py`가 orchestration한다.

전체 흐름은 다음과 같다.

```text
1. active reference update
2. reference_changed 여부 확인
3. reference가 바뀌지 않았으면 feature drift check
4. reference_changed 또는 drift_detected이면 challenger 학습
5. 기존 champion과 challenger 비교
6. challenger가 기준 통과 시 champion으로 promote
7. src/model_registry/champion 업데이트
8. src/backend_handoff/models 업데이트
9. src/models 업데이트
10. backend에 model_promoted webhook 전송
```

중요한 원칙은 다음과 같다.

```text
drift 감지 != 운영 모델 즉시 교체
```

Drift는 재학습 필요성을 알리는 신호일 뿐이다. 실제 운영 모델 교체는 challenger가 champion 대비 성능과 score 안정성 검증을 통과한 뒤에만 수행한다.

### Challenger training

`src/model_update/train_challenger.py`는 기존 best model 구조를 유지하고, 새 데이터에 대해 파라미터만 재학습한다.

즉 모델 탐색을 다시 하지 않는다.

```text
기존 best model 구조 clone
-> 새 dataset으로 fit
-> Ridge meta model 재학습
-> challenger artifact 저장
```

### Champion-challenger evaluation

`src/model_update/evaluate_challenger.py`는 champion과 challenger를 비교한다.

평가 기준은 다음과 같다.

- base ROC-AUC가 기존 champion 대비 크게 하락하지 않을 것
- base F1이 기존 champion 대비 크게 하락하지 않을 것
- meta MAE가 악화되지 않을 것
- meta RMSE가 악화되지 않을 것

Champion metric은 `final_cv_summary`를 우선 사용하고, 없으면 holdout metric으로 fallback한다.

### Model promotion

`src/model_update/promote_model.py`는 검증을 통과한 challenger를 champion으로 승격한다.

Promotion 성공 시 다음 경로가 함께 업데이트된다.

```text
src/model_registry/champion
src/backend_handoff/models
src/models
```

각 경로의 의미는 다음과 같다.

| 경로 | 역할 |
| --- | --- |
| `src/model_registry/champion` | 데이터 파이프라인 내부 최신 champion registry |
| `src/backend_handoff/models` | 백엔드가 reload해야 하는 모델 artifact |
| `src/models` | 다음 data pipeline 실행 때 기준이 되는 최신 모델 |

## Airflow DAG

Airflow DAG는 `dags/oss_health_update_dag.py`에 있다.

기본 스케줄은 매주 월요일 오전 3시, Asia/Seoul 기준이다.

```text
schedule = "0 3 * * 1"
```

Airflow 변수는 다음과 같이 설정한다.

```bash
airflow variables set oss_health_project_root /Users/carolyn/Desktop/opensource/data/oss-health-data
airflow variables set oss_health_backend_webhook_url http://localhost:8080/internal/model-promoted
```

DAG는 복잡한 로직을 직접 들고 있지 않고, `run_update_pipeline.py`를 실행하는 역할을 한다.

## Backend handoff

Backend 연동 파일은 `src/backend_handoff/`에 정리되어 있다.

백엔드가 우선 읽어야 할 문서는 다음 두 개다.

```text
src/backend_handoff/API_CONTRACT.md
src/backend_handoff/MODEL_UPDATE_CONTRACT.md
```

백엔드는 drift alert를 받지 않는다. 백엔드는 최종적으로 검증된 모델이 promote된 뒤 `model_promoted` webhook을 받는다.

Webhook payload 예시는 다음 파일에 있다.

```text
src/backend_handoff/examples/model_promoted_payload.example.json
```

Backend reload 대상 모델 파일은 다음 위치에 있다.

```text
src/backend_handoff/models/
```

## 실행 방법

### 1. 환경 설정

```bash
cd /Users/carolyn/Desktop/opensource/data/oss-health-data
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

GitHub API를 사용할 경우 token을 설정한다.

```bash
export GITHUB_TOKEN=ghp_your_token_here
```

### 2. 기존 모델/진단 smoke test

```bash
cd src/backend_handoff
python examples/smoke_test.py https://github.com/pandas-dev/pandas
```

### 3. 모델 업데이트 파이프라인 로컬 테스트

current batch feature가 아직 없다면 기존 학습 feature를 복사해 안전 테스트할 수 있다.

```bash
cd /Users/carolyn/Desktop/opensource/data/oss-health-data
mkdir -p src/outputs/model_update
cp src/outputs/2_model/final_training_dataset.csv src/outputs/model_update/current_batch_features.csv
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 src/model_update/run_update_pipeline.py
```

정상적인 동일 데이터 테스트라면 보통 다음처럼 종료된다.

```text
retrain_required: False
model_promoted: False
trigger_reason: none
```

### 4. Backend webhook 포함 실행

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 src/model_update/run_update_pipeline.py   --backend-webhook-url http://localhost:8080/internal/model-promoted
```

Webhook은 challenger가 실제로 promote된 경우에만 전송된다.

## 주요 산출물

| 경로 | 설명 |
| --- | --- |
| `src/data/main/dataset.csv` | 초기 main dataset |
| `src/outputs/2_model/final_training_dataset.csv` | base model 학습에 사용한 최종 feature dataset |
| `src/oss_health_score_labeled_train.csv` | `oss_health_score`, `new_label`이 포함된 scoring dataset |
| `src/models/` | data pipeline 기준 최신 모델 artifact |
| `src/backend_handoff/models/` | backend reload 대상 모델 artifact |
| `src/reference_store/active/reference_latest.csv` | 현재 active reference dataset |
| `src/reference_store/candidates/candidate_pool.csv` | standby candidate reference pool |
| `src/monitoring_store/drift_history.csv` | drift score history |
| `src/monitoring_store/drift_thresholds.json` | dynamic drift threshold snapshot |
| `src/outputs/model_update/pipeline_report.json` | 최신 update pipeline 실행 결과 |

## 설계상 중요한 판단

### 왜 Airflow인가

이 파이프라인은 request-time inference가 아니라 주기적으로 실행되는 batch workflow다.

필요한 기능은 다음과 같다.

- 정해진 주기 실행
- reference update, drift check, retrain, evaluation, promotion의 dependency 관리
- 실패 시 retry
- 실행 log와 report 보존
- backend webhook 전송 전후 상태 추적

따라서 단순 cron보다 Airflow가 적합하다. Kafka나 streaming system은 실시간 이벤트 처리에는 강하지만, 이 프로젝트의 핵심은 주기적 batch monitoring과 검증 기반 model promotion이므로 현재 범위에서는 과하다.

### 왜 drift 감지 후 바로 모델을 교체하지 않는가

Drift는 모델이 낡았을 가능성을 알려주는 신호이지, 새 모델이 더 낫다는 증거는 아니다.

따라서 이 프로젝트는 다음 원칙을 사용한다.

```text
Drift detected -> retrain candidate
Candidate better or stable -> promote
Candidate worse -> keep champion
```

이 구조는 오픈소스 생태계의 유동성을 반영하면서도 무분별한 모델 교체를 막는다.

### 왜 reference update가 model update보다 먼저인가

모델은 reference dataset을 기준으로 score와 target을 만든다. reference 자체가 낡거나 품질이 떨어지면, 그 위에서 drift를 계산하거나 모델을 재학습해도 기준이 흔들린다.

따라서 업데이트 순서는 다음이 맞다.

```text
reference update -> model update decision
```

Reference가 변경되면 scoring 기준이 바뀐 것이므로 drift 여부와 관계없이 challenger 재학습 대상이 된다.

## Git commit convention

권장 commit message 형식은 다음과 같다.

```text
feat: add active reference update management
feat: add model update pipeline orchestration
feat: add champion challenger evaluation
feat: add model promotion artifact sync
feat: add Airflow DAG for model update pipeline
fix: fix meta model feature names
docs: add backend model update handoff contract
```

## 한계와 향후 개선

현재 파이프라인은 GitHub metadata 기반으로 OSS health를 평가한다. 따라서 다음 신호는 아직 제한적으로만 반영된다.

- test coverage
- CI/CD pass rate
- security advisory
- dependency vulnerability
- license compatibility
- code of conduct
- maintainer response time의 정밀 측정

향후 개선 방향은 다음과 같다.

- GitHub GraphQL API 기반 더 정교한 activity window 수집
- license/security/governance 문서 존재 여부 feature 추가
- current batch 자동 수집 job 추가
- candidate reference search의 proxy feature ranking 고도화
- model registry metadata와 backend reload 상태 양방향 확인
- Airflow task를 BashOperator 단일 실행에서 task 단위 DAG로 세분화

## 프로젝트 설명용 요약

이 프로젝트는 GitHub 오픈소스 repository의 건강성을 평가하기 위해 metadata 기반 feature를 수집하고, 해석 가능한 pseudo scoring framework로 `oss_health_score`와 `new_label`을 생성한 뒤, Logistic Regression base model과 Ridge meta model을 결합해 최종 OSS Health Score를 산출한다.

또한 오픈소스 생태계의 시간적 변화를 반영하기 위해 reference dataset 품질을 5DAS로 관리하고, feature drift를 PSI와 binary ratio difference로 감지하며, drift나 reference 변화가 발생하면 기존 best model 구조를 유지한 challenger를 재학습한다. Challenger는 champion 대비 성능과 score 안정성 검증을 통과한 경우에만 promote되고, 이때 backend에는 `model_promoted` webhook만 전달된다.

따라서 이 레포는 단순 모델 학습 코드가 아니라, 데이터 기준 관리, drift monitoring, 검증 기반 model promotion, backend handoff까지 포함한 end-to-end OSS health data pipeline이다.
