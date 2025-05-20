import pytest
from fastapi.testclient import TestClient

from fastapi_app.main import app

client = TestClient(app)


def test_models_endpoint():
    resp = client.get("/models")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_match_endpoint():
    resp = client.post("/match", json={"image1": "tests/image1.jpg", "image2": "tests/image2.jpg"})
    assert resp.status_code == 200
    assert "similarity" in resp.json()


def test_recognize_endpoint():
    resp = client.post("/recognize", json={"image": "tests/image1.jpg"})
    assert resp.status_code == 200
    assert "name" in resp.json()


def test_register_endpoint():
    resp = client.post(
        "/register",
        json={"name": "test", "image": "tests/image1.jpg"},
    )
    assert resp.status_code == 200
    assert "id" in resp.json()


def test_process_endpoint():
    data = {
        "cameras": [
            {"camera_id": "cam1", "image": "tests/image1.jpg", "rtsp": "rtsp://cam1"},
            {"camera_id": "cam2", "image": "tests/image2.jpg", "rtsp": "rtsp://cam2"},
        ]
    }
    resp = client.post("/process", json=data)
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) == 2
    assert body[0]["rtsp"] == "rtsp://cam1"
