from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.main import app


def main() -> None:
    client = TestClient(app)

    health = client.get("/api/v1/health")
    status = client.get("/api/v1/status")

    payload = {
      "health_status": health.status_code,
      "health_body": health.json(),
      "status_status": status.status_code,
      "status_body": status.json(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
