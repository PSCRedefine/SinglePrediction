"""Streamlit single-prediction UI."""

from __future__ import annotations

import os

import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("SINGLE_PREDICTION_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Cognitive Shorts", page_icon="📱", layout="wide")
st.title("📱 Cognitive Shorts Recommendation System")
st.caption("输入一个真实用户和视频，预测发生主动互动的概率")

with st.sidebar:
    st.header("🎛️ Controls")
    try:
        health = requests.get(f"{API_URL}/health", timeout=2).json()
        if health.get("status") == "healthy":
            st.success(f"API Healthy · {health.get('model_name', 'model')}")
        else:
            st.warning(f"API Not Ready: {health.get('error') or '请先训练模型'}")
    except requests.RequestException:
        st.error("API Offline · 请先启动 FastAPI")
    st.selectbox(
        "Select Page",
        ["Single Prediction", "Batch Prediction（后续）", "Analytics（后续）", "Model Info（后续）"],
    )

st.header("🎯 Single Prediction")
left, right = st.columns([2, 1])
with left:
    st.subheader("Input Parameters")
    user_id = st.text_input("User ID", value="user_000001")
    video_id = st.text_input("Video ID", value="video_0000001")
    watch_time = st.slider("Watch Time (seconds)", 0.0, 180.0, 45.0, 1.0)
    hour_of_day = st.selectbox("Hour of Day", list(range(24)), index=14)

with right:
    st.subheader("Prediction")
    if st.button("🚀 Predict", type="primary", use_container_width=True):
        payload = {
            "user_id": user_id.strip(),
            "video_id": video_id.strip(),
            "watch_time": watch_time,
            "hour_of_day": hour_of_day,
        }
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            if response.ok:
                result = response.json()
                probability = result["probability"]
                st.success("Prediction Complete!")
                figure = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=probability,
                        title={"text": "Active-engagement probability"},
                        gauge={"axis": {"range": [0, 1]}},
                    )
                )
                figure.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
                st.plotly_chart(figure, use_container_width=True)
                a, b, c = st.columns(3)
                a.metric("Probability", f"{probability:.3f}")
                b.metric("Confidence", result["confidence"])
                c.metric("Response Time", f"{result['response_time_ms']:.1f} ms")
                with st.expander("Model response JSON"):
                    st.json(result)
            else:
                detail = response.json().get("detail", response.text)
                st.error(f"Prediction failed ({response.status_code}): {detail}")
        except requests.RequestException as exc:
            st.error(f"Cannot reach API: {exc}")
