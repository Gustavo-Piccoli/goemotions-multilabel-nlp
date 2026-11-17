"""Unit tests for the GoEmotions 28-class taxonomy and sentiment groupings."""

from __future__ import annotations

from src.labels import (
    GOEMOTIONS_LABELS,
    ID2LABEL,
    LABEL2ID,
    LABEL2SENTIMENT,
    NUM_LABELS,
    SENTIMENT_GROUPS,
)


def test_num_labels_and_uniqueness() -> None:
    assert NUM_LABELS == 28
    assert len(GOEMOTIONS_LABELS) == 28
    assert len(set(GOEMOTIONS_LABELS)) == 28


def test_label_id_bijection() -> None:
    for idx, label in enumerate(GOEMOTIONS_LABELS):
        assert LABEL2ID[label] == idx
        assert ID2LABEL[idx] == label


def test_sentiment_groups_partition_all_labels() -> None:
    expected_groups = {"positive", "negative", "ambiguous", "neutral"}
    assert set(SENTIMENT_GROUPS.keys()) == expected_groups

    all_grouped = [label for group in SENTIMENT_GROUPS.values() for label in group]
    assert len(all_grouped) == 28
    assert set(all_grouped) == set(GOEMOTIONS_LABELS)

    for label in GOEMOTIONS_LABELS:
        assert label in LABEL2SENTIMENT
        assert LABEL2SENTIMENT[label] in expected_groups
