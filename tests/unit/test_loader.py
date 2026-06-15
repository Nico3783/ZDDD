"""Unit tests for src.datasets.loader module."""
from __future__ import annotations

from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.core.exceptions import DataError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_COLUMNS = [
    " Flow Duration",
    " Total Fwd Packets",
    " Total Backward Packets",
    " Fwd Packet Length Mean",
    " Bwd Packet Length Mean",
    " Label",
]


def _make_csv(path: Path, rows: int = 10, columns: list[str] | None = None) -> Path:
    """Write a minimal CICIDS2017-style CSV to *path* and return it."""
    columns = columns or SAMPLE_COLUMNS
    df = pd.DataFrame(
        {col: range(rows) for col in columns},
    )
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadRawDataset:
    """Tests for load_raw_dataset()."""

    def test_valid_csv_returns_dataframe(self, tmp_path: Path):
        csv = _make_csv(tmp_path / "valid.csv")
        with patch("src.datasets.loader.get_config") as mock_cfg:
            mock_cfg.return_value.load.return_value = {
                "data": {"raw": "data/raw.csv"}
            }
            from src.datasets.loader import load_raw_dataset

            df = load_raw_dataset(path=csv)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10

    def test_valid_csv_preserves_columns(self, tmp_path: Path):
        csv = _make_csv(tmp_path / "cols.csv")
        with patch("src.datasets.loader.get_config") as mock_cfg:
            mock_cfg.return_value.load.return_value = {
                "data": {"raw": "data/raw.csv"}
            }
            from src.datasets.loader import load_raw_dataset

            df = load_raw_dataset(path=csv)

        for col in SAMPLE_COLUMNS:
            assert col in df.columns

    def test_nonexistent_path_raises_data_error(self, tmp_path: Path):
        bad_path = tmp_path / "does_not_exist.csv"
        with patch("src.datasets.loader.get_config") as mock_cfg:
            mock_cfg.return_value.load.return_value = {
                "data": {"raw": "data/raw.csv"}
            }
            from src.datasets.loader import load_raw_dataset

            with pytest.raises(DataError, match="not found"):
                load_raw_dataset(path=bad_path)

    def test_empty_csv_raises_data_error(self, tmp_path: Path):
        empty = tmp_path / "empty.csv"
        empty.write_text("")
        with patch("src.datasets.loader.get_config") as mock_cfg:
            mock_cfg.return_value.load.return_value = {
                "data": {"raw": "data/raw.csv"}
            }
            from src.datasets.loader import load_raw_dataset

            with pytest.raises(DataError):
                load_raw_dataset(path=empty)

    def test_none_path_uses_config_default(self, tmp_path: Path, write_config_files):
        """When path is None the loader reads the default path from config."""
        raw_csv = tmp_path / "data" / "raw.csv"
        raw_csv.parent.mkdir(parents=True, exist_ok=True)
        _make_csv(raw_csv, rows=5)

        with patch("src.datasets.loader.get_config") as mock_cfg:
            mock_cfg.return_value.load.return_value = {
                "data": {"raw": "data/raw.csv"}
            }
            # Patch DATA_DIR so the resolved path matches our temp file
            with patch("src.datasets.loader.DATA_DIR", tmp_path / "data"):
                from src.datasets.loader import load_raw_dataset

                df = load_raw_dataset(path=None)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_extra_columns_still_load(self, tmp_path: Path):
        """CSV with columns not in the expected set still loads successfully."""
        cols = SAMPLE_COLUMNS + [" Extra Column", " Another Extra"]
        csv = _make_csv(tmp_path / "extra.csv", columns=cols)
        with patch("src.datasets.loader.get_config") as mock_cfg:
            mock_cfg.return_value.load.return_value = {
                "data": {"raw": "data/raw.csv"}
            }
            from src.datasets.loader import load_raw_dataset

            df = load_raw_dataset(path=csv)

        assert " Extra Column" in df.columns
        assert " Another Extra" in df.columns
        assert len(df.columns) == len(cols)

    def test_large_csv_loads_correctly(self, tmp_path: Path):
        """A larger CSV (1000 rows) loads with correct row count."""
        csv = _make_csv(tmp_path / "large.csv", rows=1000)
        with patch("src.datasets.loader.get_config") as mock_cfg:
            mock_cfg.return_value.load.return_value = {
                "data": {"raw": "data/raw.csv"}
            }
            from src.datasets.loader import load_raw_dataset

            df = load_raw_dataset(path=csv)

        assert len(df) == 1000

    def test_path_as_string(self, tmp_path: Path):
        """Accepts a plain string path, not just Path object."""
        csv = _make_csv(tmp_path / "string_path.csv")
        with patch("src.datasets.loader.get_config") as mock_cfg:
            mock_cfg.return_value.load.return_value = {
                "data": {"raw": "data/raw.csv"}
            }
            from src.datasets.loader import load_raw_dataset

            df = load_raw_dataset(path=str(csv))

        assert isinstance(df, pd.DataFrame)

    def test_returns_dataframe_with_float_dtypes(self, tmp_path: Path):
        """Numeric columns in the CSV become numeric dtypes in the DataFrame."""
        csv = _make_csv(tmp_path / "dtypes.csv")
        with patch("src.datasets.loader.get_config") as mock_cfg:
            mock_cfg.return_value.load.return_value = {
                "data": {"raw": "data/raw.csv"}
            }
            from src.datasets.loader import load_raw_dataset

            df = load_raw_dataset(path=csv)

        # All columns in the sample CSV are numeric (Label column is int here)
        for col in df.columns:
            assert pd.api.types.is_numeric_dtype(df[col])


class TestStripColumnWhitespace:
    """Tests for strip_column_whitespace()."""

    def test_strips_whitespace_from_column_names(self):
        from src.datasets.loader import strip_column_whitespace

        df = pd.DataFrame(columns=[" Col A ", " ColB", "ColC "])
        result = strip_column_whitespace(df)
        assert list(result.columns) == ["Col A", "ColB", "ColC"]

    def test_no_whitespace_columns_unchanged(self):
        from src.datasets.loader import strip_column_whitespace

        df = pd.DataFrame(columns=["a", "b", "c"])
        result = strip_column_whitespace(df)
        assert list(result.columns) == ["a", "b", "c"]

    def test_returns_same_dataframe_object(self):
        from src.datasets.loader import strip_column_whitespace

        df = pd.DataFrame({" x ": [1, 2]})
        result = strip_column_whitespace(df)
        assert result is df

    def test_empty_dataframe(self):
        from src.datasets.loader import strip_column_whitespace

        df = pd.DataFrame()
        result = strip_column_whitespace(df)
        assert len(result.columns) == 0


class TestStandardizeLabels:
    """Tests for standardize_labels()."""

    def test_strips_whitespace_from_label_values(self):
        from src.datasets.loader import standardize_labels

        df = pd.DataFrame({
            "Label": [" BENIGN ", "DoS Hulk", "  DoS GoldenEye  "],
        })
        result = standardize_labels(df, label_column="Label")
        assert list(result["Label"]) == ["BENIGN", "DoS Hulk", "DoS GoldenEye"]

    def test_raises_data_error_when_label_column_missing(self):
        from src.datasets.loader import standardize_labels

        df = pd.DataFrame({"NotLabel": [1, 2, 3]})
        with pytest.raises(DataError, match="not found"):
            standardize_labels(df, label_column="Label")

    def test_handles_label_column_with_leading_space(self):
        from src.datasets.loader import standardize_labels

        df = pd.DataFrame({
            "Label": [" BENIGN ", " DoS Hulk "],
        })
        result = standardize_labels(df, label_column="Label")
        assert list(result["Label"]) == ["BENIGN", "DoS Hulk"]


class TestFilterDosTraffic:
    """Tests for filter_dos_traffic()."""

    def test_keeps_benign_and_known_attack_labels(self):
        from src.datasets.loader import filter_dos_traffic

        df = pd.DataFrame({
            "Label": ["BENIGN", "DoS Hulk", "DoS GoldenEye",
                      "PortScan", "FTP-Patator", "DoS Slowloris"],
        })
        result = filter_dos_traffic(df, label_column="Label")
        expected = {"BENIGN", "DoS Hulk", "DoS GoldenEye", "DoS Slowloris"}
        assert set(result["Label"].unique()) == expected

    def test_removes_non_dos_labels(self):
        from src.datasets.loader import filter_dos_traffic

        df = pd.DataFrame({
            "Label": ["BENIGN", "DoS Hulk", "Infiltration", "Bot"],
        })
        result = filter_dos_traffic(df, label_column="Label")
        assert "Infiltration" not in result["Label"].values
        assert "Bot" not in result["Label"].values

    def test_empty_after_filter_returns_empty_dataframe(self):
        from src.datasets.loader import filter_dos_traffic

        df = pd.DataFrame({"Label": ["Unknown1", "Unknown2"]})
        result = filter_dos_traffic(df, label_column="Label")
        assert len(result) == 0


class TestSeparateFeaturesAndLabels:
    """Tests for separate_features_and_labels()."""

    def test_splits_features_and_labels(self):
        from src.datasets.loader import separate_features_and_labels

        df = pd.DataFrame({
            " Fwd Packet Length Mean": [1.0, 2.0, 3.0],
            " Bwd Packet Length Mean": [4.0, 5.0, 6.0],
            "Label": ["BENIGN", "DoS Hulk", "BENIGN"],
        })
        features, labels = separate_features_and_labels(df, label_column="Label")
        assert "Label" not in features.columns
        assert len(features.columns) == 2
        assert list(labels) == ["BENIGN", "DoS Hulk", "BENIGN"]

    def test_raises_when_label_column_missing(self):
        from src.datasets.loader import separate_features_and_labels

        df = pd.DataFrame({"Feature1": [1, 2], "Feature2": [3, 4]})
        with pytest.raises(DataError, match="not found"):
            separate_features_and_labels(df, label_column=" Label")


class TestExtractBenignOnly:
    """Tests for extract_benign_only()."""

    def test_extracts_only_benign_rows(self):
        from src.datasets.loader import extract_benign_only

        df = pd.DataFrame({
            " Feature1": [1.0, 2.0, 3.0, 4.0],
            "Label": ["BENIGN", "DoS Hulk", "BENIGN", "DoS GoldenEye"],
        })
        result = extract_benign_only(df, label_column="Label")
        assert len(result) == 2
        assert "Label" not in result.columns

    def test_no_benign_returns_empty(self):
        from src.datasets.loader import extract_benign_only

        df = pd.DataFrame({
            " Feature1": [1.0, 2.0],
            "Label": ["DoS Hulk", "DoS GoldenEye"],
        })
        result = extract_benign_only(df, label_column="Label")
        assert len(result) == 0
