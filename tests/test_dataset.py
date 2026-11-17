"""Unit tests for multi-hot label encoding in dataset preprocessing."""

from __future__ import annotations

import pytest

from src.dataset import encode_multihot
from src.labels import NUM_LABELS


def test_encode_multihot_shape_and_values() -> None:
    batch_labels = [[0, 15], [27], []]
    encoded = encode_multihot(batch_labels)

    assert len(encoded) == 3
    assert all(len(row) == NUM_LABELS for row in encoded)

    # First example: classes 0 (admiration) and 15 (gratitude)
    assert encoded[0][0] == 1.0
    assert encoded[0][15] == 1.0
    assert sum(encoded[0]) == 2.0

    # Second example: class 27 (neutral)
    assert encoded[1][27] == 1.0
    assert sum(encoded[1]) == 1.0

    # Third example: no active labels
    assert sum(encoded[2]) == 0.0
    assert all(isinstance(val, float) for val in encoded[0])


def test_encode_multihot_invalid_index_raises() -> None:
    with pytest.raises(ValueError, match="fora do intervalo"):
        encode_multihot([[28]])
