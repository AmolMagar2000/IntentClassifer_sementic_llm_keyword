from fastapi.testclient import TestClient

from app.main import app


def test_live_server_preflight_is_allowed():
    with TestClient(app) as client:
        response = client.options(
            "/api/classify",
            headers={
                "Origin": "http://127.0.0.1:5500",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5500"
