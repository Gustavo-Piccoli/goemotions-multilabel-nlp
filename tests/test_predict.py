"""Unit tests for multi-label emotion inference and formatting (src/predict.py)."""

from __future__ import annotations

import numpy as np
import pytest

from src.labels import LABEL2ID, NUM_LABELS
from src.predict import decode_logits_to_predictions, format_prediction_report


def test_decode_logits_to_predictions_activates_above_threshold() -> None:
    texts = [
        "Thank you so much, this really helped!",
        "I'm sorry to hear that friend :(",
    ]
    # Construct synthetic logits of shape (2, 28)
    logits = np.full((2, NUM_LABELS), -4.0, dtype=np.float64)

    # Example 0: high logits for gratitude and admiration (both positive)
    logits[0, LABEL2ID["gratitude"]] = 3.0
    logits[0, LABEL2ID["admiration"]] = 1.5

    # Example 1: all logits below 0.0 (probabilities < 0.5), remorse is highest at -0.2 (~0.45)
    logits[1, LABEL2ID["remorse"]] = -0.2

    results = decode_logits_to_predictions(texts, logits, threshold=0.5, top_k=3)
    assert len(results) == 2

    # Check Example 0
    r0 = results[0]
    assert r0["text"] == texts[0]
    active_names_0 = [item["emotion"] for item in r0["active_emotions"]]
    assert active_names_0 == ["gratitude", "admiration"]
    assert r0["active_emotions"][0]["sentiment"] == "positive"
    assert r0["active_emotions"][0]["probability"] > 0.9
    assert r0["sentiment_summary"] == ["positive"]
    assert len(r0["top_k"]) == 3

    # Check Example 1 (no emotion crosses 0.5 threshold)
    r1 = results[1]
    assert r1["active_emotions"] == []
    assert r1["sentiment_summary"] == ["none (below threshold)"]
    assert r1["top_k"][0]["emotion"] == "remorse"
    assert r1["top_k"][0]["probability"] == pytest.approx(1.0 / (1.0 + np.exp(0.2)), rel=1e-4)


def test_format_prediction_report_renders_clean_output() -> None:
    texts = ["Thank you so much!"]
    logits = np.full((1, NUM_LABELS), -4.0, dtype=np.float64)
    logits[0, LABEL2ID["gratitude"]] = 2.5

    results = decode_logits_to_predictions(texts, logits, threshold=0.5, top_k=2)
    report = format_prediction_report(results, threshold=0.5)

    assert "Thank you so much!" in report
    assert "gratitude" in report
    assert "positive" in report
    assert "0.50" in report
