# OSS Health Backend Handoff

This folder contains the model and inference logic needed to diagnose a GitHub open source repository.

## Contents

```text
backend_handoff/
  models/
    oss_health_best_model.joblib
    oss_health_best_features.json
    oss_health_model_metadata.json
  data/
    reference_dataset.csv
    extract_github_repo.py
  features/
    build_features.py
  inference/
    oss_health_diagnosis.py
  examples/
    smoke_test.py
  requirements.txt
  API_CONTRACT.md
  .env.example
```

## Setup

```bash
cd backend_handoff
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set a GitHub token before running in production:

```bash
export GITHUB_TOKEN=ghp_your_token_here
```

## Smoke Test

```bash
python examples/smoke_test.py https://github.com/pandas-dev/pandas
```

## Backend Usage

```python
from inference.oss_health_diagnosis import diagnose_repository

result = diagnose_repository("https://github.com/pandas-dev/pandas")
```

## Output Meaning

- `overall_score`: final model healthy probability converted to a 0-100 score.
- `healthy_probability`: raw positive-class probability from the trained classifier.
- `dimension_scores`: five proposal-aligned diagnostic scores based on reference dataset percentiles.

The five diagnostic dimensions are:

1. 커뮤니티 활성도
2. 지속 가능성
3. 코드 품질 및 신뢰성
4. 법적/운영 거버넌스
5. 프로젝트 성숙도

## Production Notes

- The backend must preserve the feature order in `models/oss_health_best_features.json`.
- The model artifact is a sklearn-compatible pipeline, so preprocessing is included.
- GitHub API calls can hit rate limits. Use `GITHUB_TOKEN` and cache feature extraction results.
- If new features are added later, retrain the model and replace all three files in `models/` together.


## User-facing Feature Labels

The API response does not expose only raw feature names to end users. Each strength/risk item includes:

- `feature`: internal feature key for debugging
- `label`: Korean display label
- `score`: reference percentile score
- `description`: short user-facing interpretation
