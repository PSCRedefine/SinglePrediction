"""The evaluation code carries every claim in the documentation, so it is tested."""

from __future__ import annotations

import numpy as np
import pytest

from single_prediction.metrics import (
    classification_metrics,
    decile_lift,
    expected_calibration_error,
    operating_points,
    paired_bootstrap_auc,
    recommend_threshold,
    threshold_sweep,
)


def test_perfect_calibration_scores_near_zero():
    rng = np.random.default_rng(0)
    probability = rng.uniform(0, 1, 200_000)
    outcome = (rng.uniform(0, 1, 200_000) < probability).astype(int)
    assert expected_calibration_error(outcome, probability) < 0.01


def test_systematic_overconfidence_is_detected():
    assert expected_calibration_error(np.zeros(1000, dtype=int), np.full(1000, 0.9)) == pytest.approx(0.9)


def test_metrics_on_a_perfect_separator():
    metrics = classification_metrics(np.array([0, 0, 1, 1]), np.array([0.01, 0.02, 0.98, 0.99]))
    assert metrics["roc_auc"] == 1.0
    assert metrics["f1"] == 1.0


def test_degenerate_model_is_visible_as_zero_f1():
    """A model whose scores never cross the threshold must report F1 = 0."""
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.15, 0.25])   # nothing reaches 0.5
    metrics = classification_metrics(y, p, threshold=0.5)
    assert metrics["f1"] == 0.0
    assert metrics["flagged_rate"] == 0.0
    assert metrics["roc_auc"] > 0.5        # yet it still ranks correctly


def test_decile_lift_orders_best_first():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 10_000)
    y = (rng.uniform(0, 1, 10_000) < p).astype(int)
    lift = decile_lift(y, p)
    assert len(lift) == 10
    assert lift["lift"].iloc[0] > lift["lift"].iloc[-1]
    assert lift["cumulative_capture"].iloc[-1] == pytest.approx(1.0)


def test_operating_points_report_lift_at_each_budget():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 20_000)
    y = (rng.uniform(0, 1, 20_000) < p).astype(int)
    points = operating_points(y, p)
    assert (points["lift"] >= 1.0).all()
    assert points["actual_flag_rate"].is_monotonic_increasing


def test_operating_points_deduplicate_identical_thresholds():
    """A step-function model resolves several budgets to the same cut."""
    y = np.array([0, 1] * 500)
    p = np.where(np.arange(1000) < 500, 0.2, 0.4)   # only two distinct scores
    assert len(operating_points(y, p)) < 5


def test_recommendation_respects_the_volume_floor():
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, 20_000)
    y = (rng.uniform(0, 1, 20_000) < p).astype(int)
    recommendation = recommend_threshold(y, p, minimum_flag_rate=0.20)
    assert recommendation["flag_rate"] >= 0.20
    assert recommendation["lift"] > 1.0


def test_threshold_sweep_is_monotone_in_recall():
    rng = np.random.default_rng(4)
    p = rng.uniform(0, 1, 5_000)
    y = (rng.uniform(0, 1, 5_000) < p).astype(int)
    sweep = threshold_sweep(y, p)
    assert sweep["recall"].is_monotonic_decreasing
    assert sweep["flagged_rate"].is_monotonic_decreasing


def test_identical_models_are_reported_as_tied():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 5_000)
    p = rng.uniform(0, 1, 5_000)
    summary = paired_bootstrap_auc(y, {"a": p, "b": p.copy()}, resamples=50)
    assert summary["a"]["statistically_tied_with_best"]
    assert summary["b"]["statistically_tied_with_best"]


def test_a_clearly_better_model_is_not_tied():
    rng = np.random.default_rng(6)
    y = rng.integers(0, 2, 20_000)
    good = y + rng.normal(0, 0.35, 20_000)
    bad = rng.uniform(0, 1, 20_000)
    summary = paired_bootstrap_auc(y, {"good": good, "bad": bad}, resamples=100)
    assert not summary["bad"]["statistically_tied_with_best"]
