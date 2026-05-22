
# OSS Health Data

GitHub Open Source Repository의 운영 건강성(OSS Health)을 정량적으로 평가하기 위한 데이터 수집, feature engineering, 모델링, repository 진단 파이프라인입니다.

이 프로젝트는 특정 GitHub repository URL을 입력하면 해당 repository의 전반적인 OSS health score와 5개 평가 차원별 진단 결과를 제공하는 것을 목표로 합니다.

평가 차원은 다음 5가지입니다.

1. 커뮤니티 활성도
2. 지속 가능성
3. 코드 품질 및 신뢰성
4. 법적/운영 거버넌스
5. 프로젝트 성숙도

최종 모델은 repository metadata, contributors, events, tags, deployments, languages, popularity, maintenance signal 등을 기반으로 `new_label`을 예측하며, binary threshold 적용 전의 positive-class probability를 0~100점으로 변환하여 총괄 OSS Health Score로 사용합니다.

---

## 1. 프로젝트 개요

본 프로젝트는 GitHub repository의 여러 활동 지표를 수집하고, 이를 기반으로 오픈소스 프로젝트의 건강성을 평가하는 분석 및 모델링 pipeline입니다.

단순히 star 수나 fork 수만으로 repository의 상태를 판단하지 않고, 다음과 같은 다차원적 신호를 함께 사용합니다.

- 최근 활동량
- contributor 구조
- release / deployment 성숙도
- issue / pull request / comment 기반 interaction
- repository freshness
- semantic versioning 사용 여부
- governance option 활성화 여부
- archived / disabled 상태
- adoption / popularity signal
- language diversity 및 repository 규모

최종적으로는 다음 기능을 지원합니다.

- GitHub repository data extraction
- OSS health feature generation
- health score 기반 `new_label` 생성
- feature importance, SHAP, ablation 분석
- 최적 feature set 및 model 탐색
- hyperparameter tuning
- backend inference용 model artifact 저장
- 특정 repository URL 입력 기반 health diagnosis

---

## 2. 핵심 평가 차원

### 2.1 커뮤니티 활성도

핵심 질문:

```text
이 프로젝트는 현재 살아 움직이고 있는가?
```

세부 개념:

```text
Activity Volume, Responsiveness, Engagement Quality
```

주요 signal:

- GitHub event 수
- event type 다양성
- issue / issue comment 활동
- pull request 활동
- development event 비율
- interaction ratio
- activity diversity

이 차원은 repository가 실제로 활발히 운영되고 있는지 확인합니다.

---

### 2.2 지속 가능성

핵심 질문:

```text
이 프로젝트는 앞으로도 유지될 수 있는가?
```

세부 개념:

```text
Contributor Structure, Diversity, Activity Stability
```

주요 signal:

- contributor 수
- 전체 contribution 수
- contribution entropy
- top contributor 의존도
- contribution gini
- bus factor risk
- 최근 push / update 여부
- maintainer activity

이 차원은 프로젝트가 소수 maintainer에게 과도하게 의존하는지, 장기적으로 유지될 수 있는 contributor 기반을 가지고 있는지 평가합니다.

---

### 2.3 코드 품질 및 신뢰성

핵심 질문:

```text
이 프로젝트의 산출물은 믿을 수 있는가?
```

세부 개념:

```text
Engineering Practice, Defect Signals, Security Signals
```

주요 signal:

- semantic versioning 사용률
- stable release 비율
- latest tag 안정성
- release quality score
- pull request 기반 개발 활동
- issue burden
- open issues 대비 popularity
- language diversity

현재 dataset에는 test coverage, CI pass rate, vulnerability advisory 같은 직접적인 코드 품질 feature가 포함되어 있지 않습니다. 따라서 이 차원은 release practice, issue burden, PR activity 등 GitHub metadata 기반 proxy feature로 평가합니다.

---

### 2.4 법적/운영 거버넌스

핵심 질문:

```text
이 프로젝트는 조직적으로 안전하게 운영되는가?
```

세부 개념:

```text
Legal Compliance, Governance Structure
```

주요 signal:

- issues 기능 활성화
- projects 기능 활성화
- wiki 기능 활성화
- discussions 기능 활성화
- pull requests 운영 여부
- forking 허용 여부
- archived 상태
- disabled 상태
- governance openness score

현재 dataset에는 license validation, CLA, security policy, code of conduct 같은 법적/운영 문서 feature가 직접 포함되어 있지 않습니다. 따라서 repository 운영 기능과 상태 정보를 proxy로 사용합니다.

---

### 2.5 프로젝트 성숙도

핵심 질문:

```text
이 프로젝트는 성숙한 운영 체계를 갖추었는가?
```

세부 개념:

```text
Release Engineering, Adoption/Popularity, Lifecycle/Scale
```

주요 signal:

- tag 수
- major / minor version 수
- release velocity
- deployment 기록
- release recency
- stars / forks / subscribers
- adoption efficiency
- repository age
- repository size

이 차원은 프로젝트가 release 체계, adoption signal, 운영 규모를 갖춘 성숙한 프로젝트인지 평가합니다.

---

## 3. 전체 폴더 구조

주요 파일 중심의 구조는 다음과 같습니다.

```text
oss-health-data/
├── README.md
├── pyproject.toml
├── requirements.txt
├── data_extraction/
│   └── data_extraction.ipynb
│
└── src/
    ├── 0_DATASET.ipynb
    ├── 1_FEATURE.ipynb
    ├── 2_MODEL.ipynb
    ├── 3_REPOSITORY_DIAGNOSIS.ipynb
    ├── __init__.py
    ├── addFeature.parquet
    ├── oss_health_score_labeled_full.csv
    ├── oss_health_score_labeled_train.csv
    │
    ├── data/
    │   ├── extract_github_repo.py
    │   ├── main/
    │   │   └── dataset.csv
    │   └── repo/
    │       ├── healthy_repos/
    │       ├── unhealthy_repos/
    │       └── else/
    │
    ├── features/
    │   └── build_features.py
    │
    ├── models/
    │   ├── oss_health_best_model.joblib
    │   ├── oss_health_best_features.json
    │   └── oss_health_model_metadata.json
    │
    ├── outputs/
    │   ├── 2_model/
    │   ├── 3_repository_diagnosis/
    │   ├── label_ablation_result.csv
    │   ├── label_linear_or_analysis.csv
    │   ├── label_model_performance.csv
    │   ├── label_nonlinear_importance.csv
    │   ├── label_shap_importance.csv
    │   ├── label_univariate_analysis.csv
    │   ├── new_label_ablation_result.csv
    │   ├── new_label_linear_or_analysis.csv
    │   ├── new_label_model_performance.csv
    │   ├── new_label_nonlinear_importance.csv
    │   ├── new_label_univariate_analysis.csv
    │   ├── oss_health_final_modeling_dataset.csv
    │   ├── oss_health_rejected_features.csv
    │   └── oss_health_selected_features.csv
    │
    └── backend_handoff/
        ├── README.md
        ├── API_CONTRACT.md
        ├── requirements.txt
        ├── .env.example
        ├── models/
        │   ├── oss_health_best_model.joblib
        │   ├── oss_health_best_features.json
        │   └── oss_health_model_metadata.json
        ├── data/
        │   ├── reference_dataset.csv
        │   └── extract_github_repo.py
        ├── features/
        │   └── build_features.py
        ├── inference/
        │   └── oss_health_diagnosis.py
        └── examples/
            └── smoke_test.py
```

---

## 4. 주요 디렉토리 설명

### 4.1 `data_extraction/`

```text
data_extraction/data_extraction.ipynb
```

GitHub API 응답 구조를 관찰하고, repository raw data가 어떤 형태로 수집되는지 확인하기 위한 샘플 관찰용 notebook입니다.

이 notebook은 최종 pipeline의 핵심 실행 파일이라기보다, 초기 API 구조 탐색과 데이터 형태 확인을 위한 참고용 파일입니다.

---

### 4.2 `src/data/`

GitHub API 기반 data extraction 로직과 수집된 dataset을 저장합니다.

#### `src/data/extract_github_repo.py`

GitHub repository raw data를 수집하는 모듈입니다.

주요 역할:

- GitHub repository 기본 metadata 요청
- contributors, languages, tags, events, deployments, subscribers API 호출
- nested API response flattening
- feature extraction 이전 단계의 raw dataframe 생성

주요 함수:

```python
build_repo_dataframe(full_name)
```

예시:

```python
from data.extract_github_repo import build_repo_dataframe

raw_df = build_repo_dataframe("pandas-dev/pandas")
```

#### `src/data/main/dataset.csv`

최종 학습에 사용한 main dataset입니다.

각 row는 하나의 GitHub repository를 의미하며, feature columns와 기존 `label`을 포함합니다.

#### `src/data/repo/healthy_repos/`

healthy repository로 구성된 개별 feature CSV들이 저장되어 있습니다.

#### `src/data/repo/unhealthy_repos/`

unhealthy repository로 구성된 개별 feature CSV들이 저장되어 있습니다.

#### `src/data/repo/else/`

추가 수집 결과, 실패 repository 목록, 중간 병합 파일 등이 저장되어 있습니다.

---

### 4.3 `src/features/`

Feature engineering의 핵심 Python 모듈을 포함합니다.

#### `src/features/build_features.py`

GitHub API raw dataframe을 모델링 가능한 feature dataframe으로 변환합니다.

주요 역할:

- contributor feature 생성
- deployment feature 생성
- event feature 생성
- popularity / interest feature 생성
- language feature 생성
- tag / release feature 생성
- repository metadata feature 생성
- additional ratio / entropy / velocity feature 생성

주요 함수:

```python
build_all_features(data)
```

예시:

```python
from features.build_features import build_all_features

feature_df = build_all_features(raw_df)
```

이 함수는 backend inference에서도 그대로 재사용됩니다.

---

### 4.4 `src/models/`

최종 학습된 모델 artifact를 저장합니다.

```text
src/models/
├── oss_health_best_model.joblib
├── oss_health_best_features.json
└── oss_health_model_metadata.json
```

#### `oss_health_best_model.joblib`

최종 학습된 sklearn-compatible pipeline입니다.

현재 저장된 모델은 tuning까지 완료된 최종 모델입니다.

#### `oss_health_best_features.json`

최종 모델이 입력으로 사용하는 feature list입니다.

Backend inference 시 반드시 이 feature 순서를 유지해야 합니다.

#### `oss_health_model_metadata.json`

모델 학습 관련 metadata입니다.

포함 정보:

- target column
- best model name
- best dataset name
- selected feature list
- holdout metrics
- cross validation summary
- best hyperparameters

현재 최종 모델의 holdout 성능은 다음과 같습니다.

```text
accuracy : 0.9398
precision: 0.9512
recall   : 0.9286
f1       : 0.9398
roc_auc  : 0.9820
```

---

### 4.5 `src/outputs/`

노트북 실행 결과와 분석 산출물이 저장됩니다.

주요 결과:

```text
src/outputs/2_model/
```

`2_MODEL.ipynb`의 결과가 저장됩니다.

포함 파일 예시:

- `aggregate_feature_ranking.csv`
- `candidate_feature_sets.csv`
- `dataset_model_search_results.csv`
- `family_ablation_result.csv`
- `final_model_importance.csv`
- `final_model_permutation_importance.csv`
- `final_training_dataset.csv`
- `holdout_comparison.csv`
- `hyperparameter_tuning_results.csv`
- `shap_importance_result.csv`

```text
src/outputs/3_repository_diagnosis/
```

`3_REPOSITORY_DIAGNOSIS.ipynb` 또는 backend diagnosis 실행 결과가 저장됩니다.

포함 파일 예시:

- `{repo_name}_diagnosis.json`
- `{repo_name}_dimension_summary.csv`
- `{repo_name}_dimension_detail.csv`
- `{repo_name}_features.csv`

---

### 4.6 `src/backend_handoff/`

Backend 팀에 전달하기 위한 독립 실행 패키지입니다.

이 폴더는 backend에서 모델 inference를 바로 수행할 수 있도록 필요한 파일만 모아둔 handoff package입니다.

```text
backend_handoff/
├── models/
├── data/
├── features/
├── inference/
├── examples/
├── requirements.txt
├── README.md
├── API_CONTRACT.md
└── .env.example
```

#### `backend_handoff/inference/oss_health_diagnosis.py`

Backend에서 import해서 사용할 핵심 inference module입니다.

주요 함수:

```python
diagnose_repository(repo_url_or_full_name)
```

예시:

```python
from inference.oss_health_diagnosis import diagnose_repository

result = diagnose_repository("https://github.com/fastapi/fastapi")
```

#### `backend_handoff/examples/smoke_test.py`

Backend handoff package가 정상적으로 작동하는지 확인하는 테스트 스크립트입니다.

실행 예시:

```bash
cd src/backend_handoff
python examples/smoke_test.py https://github.com/fastapi/fastapi
```

---

## 5. Notebook Pipeline

본 프로젝트의 핵심 분석 pipeline은 `src/0_DATASET.ipynb`부터 `src/3_REPOSITORY_DIAGNOSIS.ipynb`까지 이어집니다.

---

### 5.1 `0_DATASET.ipynb`

Dataset을 만드는 notebook입니다.

주요 역할:

1. GitHub repository list 구성
2. healthy / unhealthy repository 후보 정의
3. GitHub API를 통한 repository feature 수집
4. 개별 repository feature CSV 저장
5. healthy / unhealthy dataframe 병합
6. 초기 `label` 부여
7. `data/main/dataset.csv` 생성

이 notebook은 모델 학습에 필요한 raw feature dataset을 만드는 출발점입니다.

---

### 5.2 `1_FEATURE.ipynb`

Feature engineering과 feature analysis를 수행하는 notebook입니다.

주요 역할:

1. feature family 정의
2. health dimension별 score 계산
3. `oss_health_score` 계산
4. median threshold 기반 `new_label` 생성
5. 기존 `label`과 `new_label` 비교
6. engineered feature 생성
7. 최종 modeling feature set 구성
8. univariate analysis
9. logistic regression odds ratio 분석
10. nonlinear feature importance 분석
11. SHAP analysis
12. family ablation test

`1_FEATURE.ipynb`는 모델 학습보다 feature 해석과 target 재정의를 담당합니다.

---

### 5.3 `2_MODEL.ipynb`

최종 modeling pipeline입니다.

주요 역할:

1. `new_label`을 target으로 설정
2. raw + engineered feature pool 구성
3. leakage column 제거
4. feature signal analysis
5. family ablation
6. aggregate feature ranking
7. candidate dataset generation
8. model zoo 비교
9. best dataset + best model 선택
10. hyperparameter tuning
11. holdout evaluation
12. final model retraining
13. backend-ready model artifact 저장

비교한 모델 예시:

- Logistic Regression
- SVC RBF
- RandomForest
- ExtraTrees
- GradientBoosting
- HistGradientBoosting
- XGBoost
- LightGBM
- CatBoost

최종 저장 파일:

```text
src/models/oss_health_best_model.joblib
src/models/oss_health_best_features.json
src/models/oss_health_model_metadata.json
```

---

### 5.4 `3_REPOSITORY_DIAGNOSIS.ipynb`

특정 GitHub repository URL을 입력받아 최종 OSS health 진단을 수행하는 notebook입니다.

주요 역할:

1. 저장된 최종 모델 load
2. repository URL parsing
3. GitHub API raw data 수집
4. feature extraction
5. engineered feature 생성
6. 모델 `predict_proba` 기반 총괄 score 계산
7. 5개 평가 차원별 percentile score 계산
8. 차원별 강점 / 위험 요인 해석
9. diagnosis JSON / CSV 저장
10. backend response 형태 예시 제공

입력 예시:

```python
REPO_URL = "https://github.com/pandas-dev/pandas"
```

출력 예시:

```json
{
  "repo_name": "pandas-dev/pandas",
  "overall_score": 92.14,
  "healthy_probability": 0.9214,
  "overall_grade": "Excellent",
  "dimension_scores": []
}
```

---

## 6. Target Label 설명

이 프로젝트에는 두 종류의 label이 존재합니다.

### 6.1 `label`

`0_DATASET.ipynb`에서 repository list 기반으로 부여된 초기 label입니다.

```text
1 = healthy repository
0 = unhealthy repository
```

초기 수집 단계에서 사용된 기준이며, 이후 feature 기반 health scoring과 비교하는 기준으로 활용됩니다.

### 6.2 `new_label`

`1_FEATURE.ipynb`에서 feature family 기반 health score를 사용해 새롭게 만든 label입니다.

계산 과정:

```text
feature family scores
-> oss_health_score
-> median threshold
-> new_label
```

정의:

```text
oss_health_score >= median  -> new_label = 1
oss_health_score < median   -> new_label = 0
```

최종 모델링에서는 `new_label`을 target으로 사용합니다.

이유는 `new_label`이 다음과 같은 다차원적 health 기준을 반영하기 때문입니다.

- community activity
- contributor sustainability
- release engineering
- popularity adoption
- language structure
- maintenance
- governance

---

## 7. 총괄 OSS Health Score

최종 진단에서 사용하는 총괄 점수는 모델의 binary prediction이 아니라, positive class probability를 0~100점으로 변환한 값입니다.

```python
healthy_probability = model.predict_proba(X)[0, 1]
overall_score = healthy_probability * 100
```

의미:

```text
모델이 해당 repository를 healthy하다고 판단하는 확률 기반 점수
```

예를 들어:

```text
healthy_probability = 0.9214
overall_score = 92.14
```

이 점수는 threshold를 적용하기 전의 연속적인 점수이므로, 단순 healthy / unhealthy 분류보다 더 세밀한 진단에 사용할 수 있습니다.

---

## 8. 5개 차원별 점수 계산 방식

총괄 점수는 모델 probability 기반입니다.

반면 5개 차원별 점수는 reference dataset 대비 percentile score로 계산합니다.

계산 방식:

1. 각 차원에 해당하는 feature list를 정의합니다.
2. 특정 repository의 feature 값을 계산합니다.
3. reference dataset의 같은 feature 분포와 비교합니다.
4. 해당 값이 reference dataset에서 어느 percentile인지 계산합니다.
5. 값이 클수록 좋은 feature는 percentile을 그대로 사용합니다.
6. 값이 작을수록 좋은 feature는 `100 - percentile` 방식으로 뒤집습니다.
7. feature score들을 평균 또는 가중 평균하여 dimension score를 만듭니다.

예시:

```text
num_events가 reference dataset의 90 percentile 위치
-> feature score = 90

last_push_recency_days가 reference dataset의 80 percentile 위치
하지만 작을수록 좋은 feature
-> feature score = 20
```

---

## 9. 등급 체계

총괄 점수와 차원별 점수는 다음 등급 체계로 해석합니다.

```text
85점 이상  -> Excellent
70점 이상  -> Good
55점 이상  -> Moderate
40점 이상  -> Weak
40점 미만  -> Risk
```

이 등급은 사용자에게 직관적인 해석을 제공하기 위한 기준입니다.

---

## 10. Backend Handoff 사용법

Backend에서 사용할 수 있는 독립 패키지는 다음 위치에 있습니다.

```text
src/backend_handoff/
```

### 10.1 설치

```bash
cd src/backend_handoff
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 10.2 GitHub Token 설정

GitHub API rate limit을 피하려면 token 설정이 필요합니다.

```bash
export GITHUB_TOKEN=your_github_token
```

또는 `.env` 파일을 사용할 수 있습니다.

```text
GITHUB_TOKEN=your_github_token
```

### 10.3 Smoke Test

```bash
python examples/smoke_test.py https://github.com/fastapi/fastapi
```

### 10.4 Python 사용 예시

```python
from inference.oss_health_diagnosis import diagnose_repository

result = diagnose_repository("https://github.com/fastapi/fastapi")
```

---

## 11. Backend API Contract

추천 request 형식:

```json
{
  "repo_url": "https://github.com/fastapi/fastapi"
}
```

추천 response 형식:

```json
{
  "repo_name": "fastapi/fastapi",
  "overall_score": 99.91,
  "healthy_probability": 0.9991,
  "overall_grade": "Excellent",
  "model_name": "LogisticRegression",
  "target": "new_label",
  "dimension_scores": [
    {
      "dimension": "community_activity",
      "label": "커뮤니티 활성도",
      "score": 77.36,
      "grade": "Good",
      "core_question": "이 프로젝트는 현재 살아 움직이고 있는가?",
      "concepts": "Activity Volume, Responsiveness, Engagement Quality",
      "summary": "커뮤니티 활성도 점수는 77.4점으로 양호하다.",
      "strength_features": [
        {
          "feature": "num_events",
          "label": "최근 활동량",
          "score": 95.0,
          "description": "최근 GitHub 이벤트가 많아 프로젝트 활동량이 높게 관찰됩니다."
        }
      ],
      "risk_features": [
        {
          "feature": "dominant_event_ratio",
          "label": "특정 이벤트 쏠림 정도",
          "score": 15.0,
          "description": "특정 이벤트 유형에 활동이 집중되어 다양성이 낮을 수 있습니다."
        }
      ]
    }
  ]
}
```

`strength_features`와 `risk_features`는 내부 feature name만 노출하지 않고, 사용자 친화적인 한국어 label과 description을 함께 제공합니다.

---

## 12. 실행 환경

Python 버전은 `pyproject.toml` 기준으로 다음 범위를 사용합니다.

```text
>=3.11,<3.14.5
```

주요 dependency:

- pandas
- numpy
- scipy
- scikit-learn
- xgboost
- lightgbm
- catboost
- shap
- joblib
- requests
- python-dotenv
- matplotlib
- seaborn
- plotly
- jupyter

설치 예시:

```bash
pip install -r requirements.txt
```

또는 pyproject 기반 환경을 사용할 수 있습니다.

```bash
pip install -e .
```

---

## 13. 주의사항

### 13.1 GitHub API Rate Limit

이 프로젝트는 GitHub API를 사용합니다.

따라서 대량 수집 또는 반복 진단 시 rate limit이 발생할 수 있습니다.

권장 사항:

```bash
export GITHUB_TOKEN=your_github_token
```

운영 환경에서는 동일 repository에 대한 feature extraction 결과를 cache하는 것이 좋습니다.

---

### 13.2 Token 보안

GitHub token은 절대 notebook output, README, commit log 등에 노출하지 않아야 합니다.

이미 노출된 token은 즉시 revoke하고 새 token을 발급해야 합니다.

---

### 13.3 현재 Feature의 한계

현재 feature set은 GitHub API metadata와 activity data를 중심으로 구성되어 있습니다.

따라서 다음 항목은 직접 반영되지 않습니다.

- license text
- SPDX license validation
- security policy
- vulnerability advisory
- dependency vulnerability
- CI pass rate
- test coverage
- code review quality
- issue close time
- PR merge time
- maintainer response time
- documentation quality
- code of conduct
- contributor license agreement

이러한 feature를 추가하면 특히 다음 차원의 정확도가 개선될 수 있습니다.

- 코드 품질 및 신뢰성
- 법적/운영 거버넌스
- 지속 가능성
- 커뮤니티 활성도

---

## 14. 향후 개선 방향

### 14.1 GitHub Actions / CI Feature 추가

추가 가능한 feature:

- workflow 존재 여부
- 최근 workflow 성공률
- failed workflow 비율
- CI activity recency
- test workflow 존재 여부

---

### 14.2 Issue / PR Lifecycle Feature 추가

추가 가능한 feature:

- 평균 issue close time
- 평균 PR merge time
- stale issue ratio
- maintainer first response time
- PR review count
- issue response latency

---

### 14.3 Security Feature 추가

추가 가능한 feature:

- security policy 존재 여부
- Dependabot 활성화 여부
- vulnerability alert 여부
- dependency file 존재 여부
- signed release 여부

---

### 14.4 Legal / Governance Feature 추가

추가 가능한 feature:

- license 존재 여부
- SPDX-compatible license 여부
- code of conduct 존재 여부
- contributing guide 존재 여부
- governance document 존재 여부
- CLA 여부

---

### 14.5 Documentation Feature 추가

추가 가능한 feature:

- README 품질
- docs directory 존재 여부
- examples directory 존재 여부
- API documentation 존재 여부
- changelog 존재 여부

---

## 15. 전체 Pipeline 요약

전체 pipeline은 다음과 같습니다.

```text
GitHub Repository URL
        |
        v
GitHub API Data Extraction
        |
        v
Raw Repository DataFrame
        |
        v
Feature Extraction
        |
        v
Raw + Engineered Features
        |
        v
OSS Health Model
        |
        v
Overall Health Probability Score
        |
        v
5-Dimension Percentile Diagnosis
        |
        v
Backend JSON Response
```

Notebook 기준 흐름:

```text
data_extraction/data_extraction.ipynb
        |
        |  sample API structure observation
        v
src/0_DATASET.ipynb
        |
        |  dataset construction
        v
src/1_FEATURE.ipynb
        |
        |  feature engineering and new_label creation
        v
src/2_MODEL.ipynb
        |
        |  model selection, tuning, artifact saving
        v
src/3_REPOSITORY_DIAGNOSIS.ipynb
        |
        |  repository-level diagnosis
        v
src/backend_handoff/
        |
        |  backend-ready inference package
        v
API / Service Integration
```

---

## 16. 현재 최종 산출물

현재 프로젝트의 최종 산출물은 다음과 같습니다.

### 모델

```text
src/models/oss_health_best_model.joblib
```

### 모델 feature list

```text
src/models/oss_health_best_features.json
```

### 모델 metadata

```text
src/models/oss_health_model_metadata.json
```

### Backend handoff package

```text
src/backend_handoff/
```

### Repository diagnosis notebook

```text
src/3_REPOSITORY_DIAGNOSIS.ipynb
```

---

## 17. 한 줄 요약

이 repository는 GitHub open source project의 활동성, 지속 가능성, 신뢰성, 거버넌스, 성숙도를 feature 기반으로 분석하고, 학습된 모델을 통해 특정 repository의 OSS health를 점수화하는 end-to-end OSS health diagnosis pipeline입니다.
