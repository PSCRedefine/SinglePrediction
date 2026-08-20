"""HTTP behaviour, covering specification §9.2 in full.

The specification asks for: /health, /predict with a valid request, /predict
with invalid parameters, and boundary values (watch_time = 0, hour_of_day = 0
and 23). Each has a test below, plus the missing-model case from §9.1.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from single_prediction.api import confidence_label, create_app


@pytest.fixture
def client(trained_artifacts) -> TestClient:
    app = create_app(
        model_path=trained_artifacts["model"],
        metadata_path=trained_artifacts["metadata"],
        users_path=trained_artifacts["users"],
        videos_path=trained_artifacts["videos"],
    )
    return TestClient(app)


# --- §9.2 GET /health -------------------------------------------------------
def test_health_reports_ready_state(client):
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] and body["feature_store_loaded"]
    assert body["users_indexed"] == 5
    assert body["videos_indexed"] == 5


def test_health_stays_up_when_the_model_is_missing(tmp_path, trained_artifacts):
    """§9.1: the backend must report a clear error when the model file is gone."""
    app = create_app(
        model_path=tmp_path / "missing.joblib",
        metadata_path=tmp_path / "missing.json",
        users_path=trained_artifacts["users"],
        videos_path=trained_artifacts["videos"],
    )
    client = TestClient(app)
    body = client.get("/health").json()
    assert body["status"] == "not_ready"
    assert body["error"]
    response = client.post(
        "/predict",
        json={"user_id": "user_000001", "video_id": "video_0000001", "watch_time": 10.0},
    )
    assert response.status_code == 503
    assert "not ready" in response.json()["detail"]


# --- §9.2 POST /predict, valid request --------------------------------------
def test_predict_returns_a_structured_result(client):
    response = client.post(
        "/predict",
        json={"user_id": "user_000001", "video_id": "video_0000001",
              "watch_time": 15.0, "hour_of_day": 14},
    )
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["confidence"] in {"low", "medium", "high"}
    assert body["hour_of_day"] == 14
    assert body["response_time_ms"] >= 0
    assert body["timestamp"].endswith("+00:00")
    assert isinstance(body["predicted_engaged"], bool)


def test_predict_is_deterministic_for_the_same_request(client):
    payload = {"user_id": "user_000002", "video_id": "video_0000002",
               "watch_time": 30.0, "hour_of_day": 9}
    first = client.post("/predict", json=payload).json()["probability"]
    second = client.post("/predict", json=payload).json()["probability"]
    assert first == second


def test_watch_ratio_drives_the_prediction(client):
    """Watching more of the same video must not lower the probability."""
    low = client.post("/predict", json={"user_id": "user_000001",
                                        "video_id": "video_0000003",
                                        "watch_time": 5.0, "hour_of_day": 12}).json()
    high = client.post("/predict", json={"user_id": "user_000001",
                                         "video_id": "video_0000003",
                                         "watch_time": 85.0, "hour_of_day": 12}).json()
    assert high["probability"] >= low["probability"]


# --- §9.2 POST /predict, invalid parameters ---------------------------------
@pytest.mark.parametrize(
    "payload",
    [
        {"user_id": "oops", "video_id": "video_0000001", "watch_time": 10.0},
        {"user_id": "user_000001", "video_id": "oops", "watch_time": 10.0},
        {"user_id": "user_000001", "video_id": "video_0000001", "watch_time": -1.0},
        {"user_id": "user_000001", "video_id": "video_0000001", "watch_time": 99999.0},
        {"user_id": "user_000001", "video_id": "video_0000001", "watch_time": 10.0,
         "hour_of_day": 24},
        {"user_id": "user_000001", "video_id": "video_0000001", "watch_time": 10.0,
         "hour_of_day": -1},
        {"user_id": "user_000001", "video_id": "video_0000001"},
        {"user_id": "user_000001", "video_id": "video_0000001", "watch_time": "abc"},
        {"user_id": "user_000001", "video_id": "video_0000001", "watch_time": 10.0,
         "unexpected": 1},
    ],
)
def test_malformed_requests_are_rejected(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_unknown_identifiers_return_404(client):
    """Well formed but absent is a different failure from malformed."""
    unknown_user = client.post(
        "/predict",
        json={"user_id": "user_999999", "video_id": "video_0000001", "watch_time": 10.0},
    )
    assert unknown_user.status_code == 404
    assert "user_999999" in unknown_user.json()["detail"]

    unknown_video = client.post(
        "/predict",
        json={"user_id": "user_000001", "video_id": "video_9999999", "watch_time": 10.0},
    )
    assert unknown_video.status_code == 404


# --- §9.2 boundary values ---------------------------------------------------
@pytest.mark.parametrize("watch_time", [0.0, 3600.0])
@pytest.mark.parametrize("hour_of_day", [0, 23])
def test_boundary_values_are_accepted(client, watch_time, hour_of_day):
    response = client.post(
        "/predict",
        json={"user_id": "user_000001", "video_id": "video_0000001",
              "watch_time": watch_time, "hour_of_day": hour_of_day},
    )
    assert response.status_code == 200
    assert 0.0 <= response.json()["probability"] <= 1.0


def test_zero_duration_video_does_not_divide_by_zero(client):
    """video_0000005 has duration 0 in the fixture."""
    response = client.post(
        "/predict",
        json={"user_id": "user_000001", "video_id": "video_0000005", "watch_time": 30.0},
    )
    assert response.status_code == 200
    assert 0.0 <= response.json()["probability"] <= 1.0


def test_hour_is_optional_and_echoed(client):
    body = client.post(
        "/predict",
        json={"user_id": "user_000001", "video_id": "video_0000001", "watch_time": 10.0},
    ).json()
    assert 0 <= body["hour_of_day"] <= 23


# --- model info -------------------------------------------------------------
def test_model_info_exposes_the_selection_story(client):
    body = client.get("/model/info").json()
    assert body["model_name"] == "test_model"
    assert body["features"] == ["watch_time_seconds", "watch_ratio"]
    assert body["recommended_threshold"] == 0.35


def test_confidence_measures_distance_from_a_coin_flip():
    assert confidence_label(0.05) == "high"
    assert confidence_label(0.30) == "medium"
    assert confidence_label(0.50) == "low"
    assert confidence_label(0.95) == "high"
