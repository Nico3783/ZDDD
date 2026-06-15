"""Unit tests for src/preprocessing/encoder.py.

Covers: encode_labels, decode_labels, create_binary_labels, get_label_mapping.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from src.core.exceptions import DataError
from src.preprocessing.encoder import (
    create_binary_labels,
    decode_labels,
    encode_labels,
    get_label_mapping,
)


# ---------------------------------------------------------------------------
# encode_labels
# ---------------------------------------------------------------------------

class TestEncodeLabels:
    """Tests for the encode_labels function."""

    def test_returns_tuple_of_array_and_encoder(self):
        labels = pd.Series(["BENIGN", "DoS Hulk", "BENIGN"])
        encoded, encoder = encode_labels(labels)
        assert isinstance(encoded, np.ndarray)
        assert isinstance(encoder, LabelEncoder)

    def test_encodes_to_integer_array(self):
        labels = pd.Series(["BENIGN", "DoS Hulk", "BENIGN"])
        encoded, _ = encode_labels(labels)
        assert encoded.dtype in (np.int64, np.int32)
        assert len(encoded) == 3

    def test_multiple_label_values_all_encoded(self):
        labels = pd.Series(["BENIGN", "DoS Hulk", "DoS GoldenEye", "BENIGN", "DoS Slowloris"])
        encoded, encoder = encode_labels(labels)
        assert len(encoder.classes_) == 4
        assert set(encoded.tolist()) == {0, 1, 2, 3}

    def test_consistent_encoding_with_pre_fitted_encoder(self):
        labels_train = pd.Series(["BENIGN", "DoS Hulk", "BENIGN", "DoS GoldenEye"])
        _, fitted_encoder = encode_labels(labels_train)

        labels_test = pd.Series(["DoS Hulk", "BENIGN"])
        encoded_test, _ = encode_labels(labels_test, encoder=fitted_encoder)
        # "BENIGN" should get the same integer as in training
        benign_idx = fitted_encoder.transform(["BENIGN"])[0]
        assert encoded_test[1] == benign_idx

    def test_raises_on_null_labels(self):
        labels = pd.Series(["BENIGN", None, "DoS Hulk"])
        with pytest.raises(DataError, match="null"):
            encode_labels(labels)

    def test_single_label_value(self):
        labels = pd.Series(["BENIGN", "BENIGN", "BENIGN"])
        encoded, encoder = encode_labels(labels)
        assert len(encoder.classes_) == 1
        assert all(encoded == 0)


# ---------------------------------------------------------------------------
# decode_labels
# ---------------------------------------------------------------------------

class TestDecodeLabels:
    """Tests for the decode_labels function."""

    def test_decode_reverses_encoding(self):
        labels = pd.Series(["BENIGN", "DoS Hulk", "DoS GoldenEye"])
        encoded, encoder = encode_labels(labels)
        decoded = decode_labels(encoded, encoder)
        pd.testing.assert_series_equal(
            pd.Series(decoded), labels, check_names=False,
        )

    def test_decode_single_value(self):
        labels = pd.Series(["DoS Slowloris"])
        encoded, encoder = encode_labels(labels)
        decoded = decode_labels(encoded, encoder)
        assert decoded[0] == "DoS Slowloris"

    def test_decode_preserves_order(self):
        original = pd.Series(["B", "A", "C", "A", "B"])
        encoded, encoder = encode_labels(original)
        decoded = decode_labels(encoded, encoder)
        assert list(decoded) == ["B", "A", "C", "A", "B"]


# ---------------------------------------------------------------------------
# get_label_mapping
# ---------------------------------------------------------------------------

class TestGetLabelMapping:
    """Tests for the get_label_mapping function."""

    def test_returns_dict(self):
        labels = pd.Series(["BENIGN", "DoS Hulk"])
        mapping = get_label_mapping(labels)
        assert isinstance(mapping, dict)

    def test_mapping_keys_are_label_strings(self):
        labels = pd.Series(["BENIGN", "DoS Hulk", "DoS GoldenEye"])
        mapping = get_label_mapping(labels)
        assert set(mapping.keys()) == {"BENIGN", "DoS Hulk", "DoS GoldenEye"}

    def test_mapping_values_are_sequential_integers(self):
        labels = pd.Series(["BENIGN", "DoS Hulk", "DoS GoldenEye"])
        mapping = get_label_mapping(labels)
        assert set(mapping.values()) == {0, 1, 2}

    def test_mapping_is_sorted_alphabetically(self):
        labels = pd.Series(["DoS Hulk", "BENIGN", "DoS GoldenEye"])
        mapping = get_label_mapping(labels)
        # Sorted: BENIGN=0, DoS GoldenEye=1, DoS Hulk=2
        assert mapping["BENIGN"] == 0
        assert mapping["DoS GoldenEye"] == 1
        assert mapping["DoS Hulk"] == 2

    def test_single_label(self):
        labels = pd.Series(["BENIGN", "BENIGN"])
        mapping = get_label_mapping(labels)
        assert mapping == {"BENIGN": 0}


# ---------------------------------------------------------------------------
# create_binary_labels
# ---------------------------------------------------------------------------

class TestCreateBinaryLabels:
    """Tests for the create_binary_labels function."""

    def test_benign_label_encodes_to_zero(self):
        labels = pd.Series(["BENIGN", "BENIGN", "BENIGN"])
        binary = create_binary_labels(labels)
        assert all(binary == 0)

    def test_attack_labels_encode_to_one(self):
        labels = pd.Series(["DoS Hulk", "DoS GoldenEye"])
        binary = create_binary_labels(labels)
        assert all(binary == 1)

    def test_mixed_labels(self):
        labels = pd.Series(["BENIGN", "DoS Hulk", "BENIGN", "DoS GoldenEye"])
        binary = create_binary_labels(labels)
        assert list(binary) == [0, 1, 0, 1]

    def test_custom_benign_label(self):
        labels = pd.Series(["NORMAL", "ATTACK", "NORMAL"])
        binary = create_binary_labels(labels, benign_label="NORMAL")
        assert list(binary) == [0, 1, 0]

    def test_output_is_integer_series(self):
        labels = pd.Series(["BENIGN", "DoS Hulk"])
        binary = create_binary_labels(labels)
        assert binary.dtype in (np.int64, np.int32, int)

    def test_whitespace_in_labels_handled(self):
        labels = pd.Series([" BENIGN ", "  DoS Hulk  "])
        binary = create_binary_labels(labels)
        assert list(binary) == [0, 1]


# ---------------------------------------------------------------------------
# Integration: encode_labels round-trip
# ---------------------------------------------------------------------------

class TestEncoderRoundTrip:
    """Integration tests combining encode and decode."""

    def test_encode_then_decode_matches_original(self):
        original = pd.Series(["BENIGN", "DoS Hulk", "DoS GoldenEye", "DoS Slowloris", "BENIGN"])
        encoded, encoder = encode_labels(original)
        decoded = decode_labels(encoded, encoder)
        assert list(decoded) == list(original)

    def test_get_label_mapping_matches_encoder_classes(self):
        labels = pd.Series(["BENIGN", "DoS Hulk", "DoS GoldenEye"])
        mapping = get_label_mapping(labels)
        _, encoder = encode_labels(labels)
        mapping_from_encoder = {
            cls: int(idx) for idx, cls in enumerate(encoder.classes_)
        }
        assert mapping == mapping_from_encoder
