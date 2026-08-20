"""FastAPI inference service for single-interaction engagement prediction.

Operational choices:

* Model and identifier store load once at process start, not per request.
* A load failure does not kill the process. ``/health`` stays up and reports the
  cause, so an orchestrator sees "not ready" rather than a crash loop and an
  operator sees the actual error rather than an empty log.
* CORS is restricted to the local Streamlit origins.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import __version__
from .config import (
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    METADATA_PATH,
    MODEL_PATH,
    USERS_PATH,
    VIDEOS_PATH,
)
from .features import FEATURE_COLUMNS, MAX_WATCH_TIME_SECONDS, FeatureStore, validate_ids

logger = logging.getLogger("single_prediction.api")


class PredictionRequest(BaseModel):
    """The request contract from the specification.

    ``hour_of_day`` is accepted, validated and echoed because the specification
    defines it as an input. It is **not** a model feature: measurement puts its
    univariate ROC-AUC at 0.5010 with an interval that includes pure noise. The
    API contract and the feature set are separate decisions, and conflating them
    is how noise ends up in models.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=6, max_length=80, examples=["user_000001"])
    video_id: str = Field(min_length=7, max_length=80, examples=["video_0000001"])
    watch_time: float = Field(ge=0, le=MAX_WATCH_TIME_SECONDS, examples=[45.0])
    hour_of_day: int | None = Field(default=None, ge=0, le=23, examples=[14])

    @field_validator("user_id", "video_id")
    @classmethod
    def validate_identifier(cls, value: str, info) -> str:
        if info.field_name == "user_id":
            validate_ids(value, "video_placeholder")
        else:
            validate_ids("user_placeholder", value)
        return value


class PredictionResponse(BaseModel):
    user_id: str
    video_id: str
    watch_time: float
    hour_of_day: int
    probability: float
    confidence: str
    predicted_engaged: bool
    threshold: float
    model_name: str
    model_version: str
    response_time_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    feature_store_loaded: bool
    model_name: str
    model_version: str
    users_indexed: int
    videos_indexed: int
    uptime_seconds: float
    error: str | None = None


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    trained_at: str | None = None
    split_strategy: str | None = None
    selection_metric: str | None = None
    recommended_threshold: float | None = None
    features: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    test_metrics: dict[str, Any] = Field(default_factory=dict)


def confidence_label(probability: float) -> str:
    """Distance from a coin flip.

    This measures decisiveness, not correctness: a confidently wrong prediction
    still reports "high". With a ceiling near 0.60 AUC on this dataset, most
    predictions will legitimately read "low", and that is the honest answer.
    """
    certainty = max(probability, 1.0 - probability)
    if certainty >= CONFIDENCE_HIGH:
        return "high"
    if certainty >= CONFIDENCE_MEDIUM:
        return "medium"
    return "low"


def create_app(
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
    users_path: Path = USERS_PATH,
    videos_path: Path = VIDEOS_PATH,
) -> FastAPI:
    app = FastAPI(
        title="Cognitive Shorts Prediction API",
        version=__version__,
        description=(
            "Single-interaction engagement prediction.\n\n"
            "* `POST /predict` - score one user-video interaction\n"
            "* `GET /health` - liveness and readiness\n"
            "* `GET /model/info` - active model, selection evidence and feature list"
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.state.started_at = time.time()
    app.state.model = None
    app.state.store = None
    app.state.metadata = {"model_name": "unknown", "model_version": __version__}
    app.state.load_error = None
    try:
        app.state.model = joblib.load(model_path)
        app.state.store = FeatureStore.from_csv(str(users_path), str(videos_path))
        if Path(metadata_path).exists():
            app.state.metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        logger.info("loaded model %s", app.state.metadata.get("model_name"))
    except Exception as exc:  # noqa: BLE001 - keep /health serviceable
        app.state.load_error = str(exc)
        logger.error("failed to load model or feature store: %s", exc)

    def ready() -> None:
        if app.state.model is None or app.state.store is None:
            raise HTTPException(
                status_code=503,
                detail=f"model service is not ready: "
                       f"{app.state.load_error or 'train a model first'}",
            )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        store = app.state.store
        is_ready = app.state.model is not None and store is not None
        return HealthResponse(
            status="healthy" if is_ready else "not_ready",
            model_loaded=app.state.model is not None,
            feature_store_loaded=store is not None,
            model_name=str(app.state.metadata.get("model_name", "unknown")),
            model_version=str(app.state.metadata.get("model_version", __version__)),
            users_indexed=0 if store is None else len(store.user_ids),
            videos_indexed=0 if store is None else len(store.video_ids),
            uptime_seconds=round(time.time() - app.state.started_at, 3),
            error=app.state.load_error,
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        ready()
        started = time.perf_counter()
        hour = request.hour_of_day if request.hour_of_day is not None else datetime.now(UTC).hour
        try:
            frame = app.state.store.build_one(
                request.user_id, request.video_id, request.watch_time
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc).strip("'")) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        probability = float(
            np.clip(app.state.model.predict_proba(frame[FEATURE_COLUMNS])[0, 1], 0.0, 1.0)
        )
        threshold = float(app.state.metadata.get("recommended_threshold", 0.5))
        elapsed = (time.perf_counter() - started) * 1000
        return PredictionResponse(
            user_id=request.user_id,
            video_id=request.video_id,
            watch_time=request.watch_time,
            hour_of_day=hour,
            probability=round(probability, 6),
            confidence=confidence_label(probability),
            predicted_engaged=probability >= threshold,
            threshold=threshold,
            model_name=str(app.state.metadata.get("model_name", "unknown")),
            model_version=str(app.state.metadata.get("model_version", __version__)),
            response_time_ms=round(elapsed, 3),
            timestamp=datetime.now(UTC).isoformat(),
        )

    @app.get("/model/info", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        ready()
        metadata = app.state.metadata
        return ModelInfoResponse(
            model_name=str(metadata.get("model_name", "unknown")),
            model_version=str(metadata.get("model_version", __version__)),
            trained_at=metadata.get("trained_at"),
            split_strategy=metadata.get("split_strategy"),
            selection_metric=metadata.get("selection_metric"),
            recommended_threshold=metadata.get("recommended_threshold"),
            features=list(metadata.get("features", FEATURE_COLUMNS)),
            metrics=metadata.get("metrics", {}),
            test_metrics=metadata.get("test_metrics", {}),
        )

    return app


app = create_app()
