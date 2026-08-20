"""Streamlit front end for single-interaction engagement prediction.

The page holds no modelling logic. It collects the four request fields, applies
client-side validation so obviously malformed input never reaches the network,
posts to the API and renders what comes back.
"""

from __future__ import annotations

import os
import re
from typing import Any

import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("SINGLE_PREDICTION_API_URL", "http://127.0.0.1:8000").rstrip("/")
CLIENT_ERROR = "__client_error__"

USER_ID_PATTERN = re.compile(r"user_[A-Za-z0-9_-]+")
VIDEO_ID_PATTERN = re.compile(r"video_[A-Za-z0-9_-]+")

st.set_page_config(page_title="Cognitive Shorts", page_icon="📱", layout="wide")


def failed(response: dict[str, Any]) -> str | None:
    """Return the transport failure message, or None if the call succeeded.

    The sentinel key is namespaced deliberately. ``/health`` returns a field
    literally called ``error`` (null when healthy), so a naive
    ``"error" in response`` check reports every healthy service as offline.
    """
    return response.get(CLIENT_ERROR)


def call_api(path: str, payload: dict[str, Any] | None = None, timeout: int = 15) -> dict[str, Any]:
    """One place where every network failure mode is named.

    Returns a dict rather than raising, so a dead backend shows a red bar
    instead of a stack trace.
    """
    url = f"{API_URL}/{path.lstrip('/')}"
    try:
        response = (
            requests.get(url, timeout=timeout)
            if payload is None
            else requests.post(url, json=payload, timeout=timeout)
        )
    except requests.ConnectionError:
        return {CLIENT_ERROR: "API server is offline - start the FastAPI service first"}
    except requests.Timeout:
        return {CLIENT_ERROR: f"Request timed out after {timeout}s"}
    except requests.RequestException as exc:
        return {CLIENT_ERROR: f"Request failed: {exc}"}
    if response.ok:
        return response.json()
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    return {CLIENT_ERROR: f"HTTP {response.status_code}: {detail}"}


def validate_inputs(user_id: str, video_id: str) -> list[str]:
    """Client-side format checks (specification §7.1).

    These duplicate the server's validation on purpose. The server's copy is the
    one that protects the service; this copy exists so a typo produces an
    immediate, readable message instead of a round trip and a 422.
    """
    problems = []
    if not USER_ID_PATTERN.fullmatch(user_id.strip()):
        problems.append("User ID must look like `user_000001`.")
    if not VIDEO_ID_PATTERN.fullmatch(video_id.strip()):
        problems.append("Video ID must look like `video_0000001`.")
    return problems


st.title("📱 Cognitive Shorts Recommendation System")
st.caption("Predict the probability that a user actively engages with a video")

health = call_api("health")
with st.sidebar:
    st.header("🎛️ Controls")
    if failed(health):
        st.error(f"❌ API Offline\n\n{failed(health)}")
    elif health.get("status") == "healthy":
        st.success("✅ API Status: Healthy")
        st.metric("Uptime", f"{health['uptime_seconds']:.0f}s")
        st.caption(
            f"Model `{health['model_name']}` · "
            f"{health['users_indexed']:,} users / {health['videos_indexed']:,} videos indexed"
        )
    else:
        st.warning(f"⚠️ API Not Ready: {health.get('error') or 'train a model first'}")
    st.divider()
    page = st.selectbox("Select Page", ["Single Prediction", "Model Info"])


if page == "Single Prediction":
    st.header("🎯 Single Prediction")
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Input Parameters")
        user_id = st.text_input("User ID", value="user_000001")
        video_id = st.text_input("Video ID", value="video_0000001")
        watch_time = st.slider("Watch Time (seconds)", 0.0, 180.0, 45.0, 1.0)
        hour_of_day = st.selectbox("Hour of Day", list(range(24)), index=14)
        st.caption(
            "The model uses watch time and watch ratio. `hour_of_day` is part of "
            "the request contract but measured to carry no signal, so it is "
            "recorded and echoed rather than used as a feature."
        )

    with right:
        st.subheader("Prediction")
        if st.button("🚀 Predict", type="primary", use_container_width=True):
            problems = validate_inputs(user_id, video_id)
            if problems:
                for problem in problems:
                    st.error(f"❌ {problem}")
            else:
                response = call_api(
                    "predict",
                    {
                        "user_id": user_id.strip(),
                        "video_id": video_id.strip(),
                        "watch_time": float(watch_time),
                        "hour_of_day": int(hour_of_day),
                    },
                )
                if failed(response):
                    st.error(f"❌ {failed(response)}")
                else:
                    probability = response["probability"]
                    threshold = response.get("threshold", 0.5)
                    st.success("Prediction complete")
                    gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=probability,
                            title={"text": "Engagement probability"},
                            gauge={
                                "axis": {"range": [0, 1]},
                                "bar": {"color": "#2a78d6"},
                                "threshold": {
                                    "line": {"color": "#e34948", "width": 3},
                                    "value": threshold,
                                },
                            },
                        )
                    )
                    gauge.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
                    st.plotly_chart(gauge, use_container_width=True)
                    a, b, c = st.columns(3)
                    a.metric("Probability", f"{probability:.3f}")
                    b.metric("Confidence", response["confidence"])
                    c.metric("Response Time", f"{response['response_time_ms']:.1f} ms")
                    st.caption(
                        f"Decision threshold {threshold:.3f} "
                        f"→ {'engaged' if response['predicted_engaged'] else 'not engaged'}. "
                        "The threshold comes from the operating-point analysis, not from 0.5."
                    )
                    with st.expander("Model response JSON"):
                        st.json(response)

else:
    st.header("🧠 Model Info")
    info = call_api("model/info")
    if failed(info):
        st.error(f"❌ {failed(info)}")
    else:
        import pandas as pd

        top = st.columns(4)
        top[0].metric("Model", info["model_name"])
        top[1].metric("Version", info["model_version"])
        top[2].metric("Split", info.get("split_strategy") or "-")
        top[3].metric("Threshold", f"{info.get('recommended_threshold') or 0.5:.3f}")
        st.caption(f"Trained at {info.get('trained_at') or 'unknown'}")

        st.subheader("Features actually used")
        st.write(info.get("features", []))
        st.caption(
            "Two features, selected by measurement from twenty-five candidates. "
            "See docs/FEATURE_SELECTION.md."
        )

        test_metrics = info.get("test_metrics") or {}
        if test_metrics:
            st.subheader("Held-out test performance")
            st.dataframe(
                pd.DataFrame([test_metrics]).T.rename(columns={0: "value"}),
                use_container_width=True,
            )

        metrics = info.get("metrics") or {}
        if metrics:
            st.subheader("Candidate comparison (validation split)")
            st.dataframe(
                pd.DataFrame(metrics).T.reset_index(names="model"),
                use_container_width=True,
                hide_index=True,
            )
