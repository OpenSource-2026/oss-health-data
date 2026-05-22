import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference.oss_health_diagnosis import diagnose_repository

repo = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/pandas-dev/pandas"
result = diagnose_repository(repo)
print(json.dumps(result, ensure_ascii=False, indent=2))
