"""Unit tests for multi-label classification metrics and thresholding."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.metrics import (
    build_compute_metrics,
    compute_multilabel_metrics,
    logits_to_preds,
    sigmoid,
)


def test_sigmoid_numerical_stability() -> None:
    x = np.array([-1000.0, 0.0, 1000.0], dtype=np.float64)
    probs = sigmoid(x)
    assert np.isclose(probs[0], 0.0)
    assert np.isclose(probs[1], 0.5)
    assert np.isclose(probs[2], 1.0)


def test_logits_to_preds_thresholding() -> None:
    logits = np.array([[-2.0, 0.0, 2.0]])
    preds_default = logits_to_preds(logits, threshold=0.5)
    assert np.array_equal(preds_default, np.array([[0, 1, 1]]))

    preds_strict = logits_to_preds(logits, threshold=0.8)
    assert np.array_equal(preds_strict, np.array([[0, 0, 1]]))


def test_compute_multilabel_metrics_perfect_and_zero() -> None:
    labels = np.array([[1, 0, 1], [0, 1, 0]], dtype=int)
    perfect_logits = np.array([[5.0, -5.0, 5.0], [-5.0, 5.0, -5.0]])

    m_perfect = compute_multilabel_metrics(perfect_logits, labels, threshold=0.5)
    assert m_perfect["macro_f1"] == pytest.approx(1.0)
    assert m_perfect["micro_f1"] == pytest.approx(1.0)
    assert m_perfect["weighted_f1"] == pytest.approx(1.0)
    assert m_perfect["subset_accuracy"] == pytest.approx(1.0)

    all_negative_logits = np.full_like(perfect_logits, -5.0)
    m_zero = compute_multilabel_metrics(all_negative_logits, labels, threshold=0.5)
    assert m_zero["macro_f1"] == pytest.approx(0.0)
    assert m_zero["micro_f1"] == pytest.approx(0.0)
    assert m_zero["subset_accuracy"] == pytest.approx(0.0)


def test_build_compute_metrics_handles_tuple_predictions() -> None:
    labels = np.array([[1, 0], [0, 1]], dtype=int)
    logits = np.array([[4.0, -4.0], [-4.0, 4.0]])
    eval_pred = SimpleNamespace(predictions=(logits, "extra_hidden_states"), label_ids=labels)

    fn = build_compute_metrics(threshold=0.5)
    res = fn(eval_pred)
    assert res["macro_f1"] == pytest.approx(1.0)
    assert res["subset_accuracy"] == pytest.approx(1.0)
