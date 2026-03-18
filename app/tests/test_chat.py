from fastapi.testclient import TestClient

from app.core.tracing import TraceStore
from app.main import app


def test_health_route() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["service"] == "nano-rag"
