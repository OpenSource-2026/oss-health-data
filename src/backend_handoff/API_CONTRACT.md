# OSS Health Diagnosis API Contract

## Request

POST /api/oss-health/diagnose

```json
{
  "repo_url": "https://github.com/pandas-dev/pandas"
}
```

`repo_url` may also be an `owner/repo` string, such as:

```json
{
  "repo_url": "pandas-dev/pandas"
}
```

## Response

```json
{
  "repo_name": "pandas-dev/pandas",
  "overall_score": 92.14,
  "healthy_probability": 0.9214,
  "overall_grade": "Excellent",
  "model_name": "LogisticRegression",
  "target": "new_label",
  "dimension_scores": [
    {
      "dimension": "community_activity",
      "label": "커뮤니티 활성도",
      "score": 87.21,
      "grade": "Excellent",
      "core_question": "이 프로젝트는 현재 살아 움직이고 있는가?",
      "concepts": "Activity Volume, Responsiveness, Engagement Quality",
      "summary": "커뮤니티 활성도 점수는 87.2점으로 양호하다.",
      "strength_features": [
        {
          "feature": "event_type_entropy",
          "label": "활동 분산도",
          "score": 92.46,
          "description": "활동이 한 유형에만 치우치지 않고 비교적 다양하게 분포합니다."
        }
      ],
      "risk_features": [
        {
          "feature": "dominant_event_ratio",
          "label": "특정 이벤트 쏠림 정도",
          "score": 28.71,
          "description": "특정 이벤트 유형에 활동이 집중되어 다양성이 낮을 수 있습니다."
        }
      ]
    }
  ]
}
```

## Notes

- `overall_score` is `predict_proba(...)[1] * 100` from the final trained model.
- The five dimension scores are reference-dataset percentile scores, not model probabilities.
- Set `GITHUB_TOKEN` in the backend environment to avoid GitHub API rate limits.
- Cache repeated repository feature extraction results in production.

## Feature Insight Format

`strength_features` and `risk_features` return user-facing labels and explanations, while preserving the original model feature key for debugging.

```json
{
  "feature": "num_events",
  "label": "최근 활동량",
  "score": 95.13,
  "description": "최근 GitHub 이벤트가 많아 프로젝트 활동량이 높게 관찰됩니다."
}
```
