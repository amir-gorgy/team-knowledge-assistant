from fastapi.testclient import TestClient

from team_knowledge_assistant.config import settings
from team_knowledge_assistant.main import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200


def test_health_returns_503_when_db_unreachable(monkeypatch):
    monkeypatch.setattr(settings, "database_url", "postgresql://app:app@localhost:1/tka")
    r = client.get("/health")
    assert r.status_code == 503
    assert r.json() == {"status": "degraded", "db": "unreachable"}
