"""FastAPI inference service."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from . import __version__
from .config import METADATA_PATH, MODEL_PATH, USERS_PATH, VIDEOS_PATH
from .features import FeatureStore, validate_ids


class PredictionRequest(BaseModel):
    user_id: str = Field(min_length=6, max_length=80)
    video_id: str = Field(min_length=7, max_length=80)
    watch_time: float = Field(ge=0, le=3600)
    hour_of_day: int = Field(ge=0, le=23)

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
    probability: float
    confidence: str
    predicted_engaged: bool
    model_name: str
    model_version: str
    response_time_ms: float
    timestamp: str


def confidence_label(probability: float) -> str:
    certainty = max(probability, 1 - probability)
    if certainty >= 0.8:
        return "high"
    if certainty >= 0.65:
        return "medium"
    return "low"


def create_app(
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
    users_path: Path = USERS_PATH,
    videos_path: Path = VIDEOS_PATH,
) -> FastAPI:
    app = FastAPI(title="Cognitive Shorts Prediction API", version=__version__)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.state.model = None
    app.state.store = None
    app.state.metadata = {"model_name": "unknown", "model_version": __version__}
    app.state.load_error = None
    try:
        app.state.model = joblib.load(model_path)
        app.state.store = FeatureStore.from_csv(str(users_path), str(videos_path))
        if metadata_path.exists():
            app.state.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:  # Keep health endpoint available with a useful message.
        app.state.load_error = str(exc)

    @app.get("/health")
    def health() -> dict[str, object]:
        ready = app.state.model is not None and app.state.store is not None
        return {
            "status": "healthy" if ready else "not_ready",
            "model_loaded": app.state.model is not None,
            "feature_store_loaded": app.state.store is not None,
            "model_name": app.state.metadata.get("model_name", "unknown"),
            "error": app.state.load_error,
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        started = time.perf_counter()
        if app.state.model is None or app.state.store is None:
            raise HTTPException(
                status_code=503,
                detail=f"模型服务尚未就绪: {app.state.load_error or '请先训练模型'}",
            )
        try:
            frame = app.state.store.build_one(
                request.user_id, request.video_id, request.watch_time, request.hour_of_day
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc).strip("'")) from exc
        probability = float(app.state.model.predict_proba(frame)[0, 1])
        elapsed = (time.perf_counter() - started) * 1000
        return PredictionResponse(
            user_id=request.user_id,
            video_id=request.video_id,
            probability=round(probability, 6),
            confidence=confidence_label(probability),
            predicted_engaged=probability >= 0.5,
            model_name=str(app.state.metadata.get("model_name", "unknown")),
            model_version=str(app.state.metadata.get("model_version", __version__)),
            response_time_ms=round(elapsed, 3),
            timestamp=datetime.now(UTC).isoformat(),
        )

    return app


app = create_app()
