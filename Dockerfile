# One image, three entry points: train, serve, console. They share every
# dependency, and building separate images would let them drift apart — a
# feature-contract change landing in the trainer but not the server is exactly
# the failure this avoids.

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first: this layer is cached until requirements.txt changes, so
# editing source does not reinstall scikit-learn and LightGBM.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps -e .

COPY app.py ./
COPY scripts/ ./scripts/
COPY monitoring/ ./monitoring/
COPY .streamlit/ ./.streamlit/

# The lookup tables are small and tracked, so they are baked in. The raw
# interaction log (139 MB) and the trained model are not: the log is too large
# to ship and the model does not exist in a fresh clone. Both arrive as mounts.
COPY data/users.csv data/videos.csv ./data/

RUN mkdir -p models reports image \
 && useradd --create-home --uid 10001 appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8501

CMD ["uvicorn", "single_prediction.api:app", "--host", "0.0.0.0", "--port", "8000"]
