from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, Generator, List, Optional

import pandas as pd

from src.core.config import get_config
from src.core.constants import DATA_DIR
logger = logging.getLogger(__name__)


class StreamStreamReader:
    """Reads network traffic from files or streams for real-time processing.

    Supports CSV, Parquet, and line-delimited JSON formats.
    Provides file watching, batch reading, and streaming iteration.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the stream reader.

        Args:
            config: Reader configuration dict. Loaded from config if None.
        """
        if config is None:
            try:
                self.config = get_config("streaming")
            except Exception:
                self.config = {}
        else:
            self.config = config

        self._current_source: Optional[str] = None
        self._read_position: int = 0
        self._total_read: int = 0
        self._callback: Optional[Callable[[pd.DataFrame], None]] = None

        logger.info("StreamStreamReader initialized")

    def read_csv(
        self,
        filepath: str,
        batch_size: int = 1000,
        skip_rows: int = 0,
        columns: Optional[List[str]] = None,
    ) -> Generator[pd.DataFrame, None, None]:
        """Read a CSV file in batches for streaming processing.

        Args:
            filepath: Path to CSV file.
            batch_size: Number of rows per batch.
            skip_rows: Number of initial rows to skip.
            columns: Optional list of columns to read.

        Yields:
            DataFrame batches.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")

        self._current_source = filepath
        self._read_position = 0

        try:
            reader = pd.read_csv(
                filepath,
                chunksize=batch_size,
                skiprows=skip_rows,
                usecols=columns,
            )

            for chunk in reader:
                self._read_position += len(chunk)
                self._total_read += len(chunk)
                yield chunk

        except Exception as e:
            logger.error("Error reading CSV file %s: %s", filepath, e)
            raise

        self._current_source = None
        logger.info("Finished reading CSV: %s (%d rows)", filepath, self._read_position)

    def read_parquet(
        self,
        filepath: str,
        batch_size: int = 1000,
        columns: Optional[List[str]] = None,
    ) -> Generator[pd.DataFrame, None, None]:
        """Read a Parquet file in batches.

        Args:
            filepath: Path to Parquet file.
            batch_size: Number of rows per batch.
            columns: Optional list of columns to read.

        Yields:
            DataFrame batches.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Parquet file not found: {filepath}")

        self._current_source = filepath
        self._read_position = 0

        try:
            import pyarrow.parquet as pq

            parquet_file = pq.ParquetFile(filepath)
            batches = parquet_file.iter_batches(batch_size=batch_size, columns=columns)

            for batch in batches:
                df = batch.to_pandas()
                self._read_position += len(df)
                self._total_read += len(df)
                yield df

        except ImportError:
            logger.warning("pyarrow not installed, falling back to pandas")
            reader = pd.read_parquet(filepath, columns=columns)
            for start in range(0, len(reader), batch_size):
                batch = reader.iloc[start : start + batch_size]
                self._read_position += len(batch)
                self._total_read += len(batch)
                yield batch

        except Exception as e:
            logger.error("Error reading Parquet file %s: %s", filepath, e)
            raise

        self._current_source = None
        logger.info("Finished reading Parquet: %s", filepath)

    def read_jsonl(
        self,
        filepath: str,
        batch_size: int = 1000,
    ) -> Generator[pd.DataFrame, None, None]:
        """Read a line-delimited JSON file in batches.

        Args:
            filepath: Path to JSONL file.
            batch_size: Number of rows per batch.

        Yields:
            DataFrame batches.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"JSONL file not found: {filepath}")

        self._current_source = filepath
        self._read_position = 0
        batch_records: List[Dict[str, Any]] = []

        try:
            import json

            with open(filepath) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            batch_records.append(record)
                        except json.JSONDecodeError as e:
                            logger.warning("Skipping malformed JSON line: %s", e)
                            continue

                    if len(batch_records) >= batch_size:
                        df = pd.DataFrame(batch_records)
                        self._read_position += len(df)
                        self._total_read += len(df)
                        yield df
                        batch_records = []

        except Exception as e:
            logger.error("Error reading JSONL file %s: %s", filepath, e)
            raise

        if batch_records:
            df = pd.DataFrame(batch_records)
            self._read_position += len(df)
            self._total_read += len(df)
            yield df

        self._current_source = None
        logger.info("Finished reading JSONL: %s", filepath)

    def watch_directory(
        self,
        directory: str,
        file_pattern: str = "*.csv",
        poll_interval: float = 1.0,
        batch_size: int = 1000,
    ) -> Generator[pd.DataFrame, None, None]:
        """Watch a directory for new files and process them.

        Args:
            directory: Directory to watch.
            file_pattern: Glob pattern for matching files.
            poll_interval: Seconds between directory checks.
            batch_size: Rows per batch.

        Yields:
            DataFrame batches from new files.
        """
        import glob

        processed_files: set = set()

        logger.info("Watching directory: %s (pattern: %s)", directory, file_pattern)

        while True:
            pattern = os.path.join(directory, file_pattern)
            files = sorted(glob.glob(pattern))

            new_files = [f for f in files if f not in processed_files]

            for filepath in new_files:
                logger.info("Processing new file: %s", filepath)

                if filepath.endswith(".csv"):
                    yield from self.read_csv(filepath, batch_size=batch_size)
                elif filepath.endswith(".parquet"):
                    yield from self.read_parquet(filepath, batch_size=batch_size)
                elif filepath.endswith(".jsonl") or filepath.endswith(".json"):
                    yield from self.read_jsonl(filepath, batch_size=batch_size)

                processed_files.add(filepath)

            time.sleep(poll_interval)

    def set_callback(self, callback: Callable[[pd.DataFrame], None]) -> None:
        """Set a callback function for processing each batch.

        Args:
            callback: Function to call with each DataFrame batch.
        """
        self._callback = callback

    def process_with_callback(
        self,
        filepath: str,
        batch_size: int = 1000,
    ) -> int:
        """Read a file and process each batch through the callback.

        Args:
            filepath: Path to data file.
            batch_size: Rows per batch.

        Returns:
            Number of batches processed.
        """
        if self._callback is None:
            raise ValueError("No callback set. Call set_callback() first.")

        batch_count = 0

        if filepath.endswith(".csv"):
            reader = self.read_csv(filepath, batch_size=batch_size)
        elif filepath.endswith(".parquet"):
            reader = self.read_parquet(filepath, batch_size=batch_size)
        elif filepath.endswith(".jsonl") or filepath.endswith(".json"):
            reader = self.read_jsonl(filepath, batch_size=batch_size)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

        for batch in reader:
            try:
                self._callback(batch)
                batch_count += 1
            except Exception as e:
                logger.error("Callback error on batch %d: %s", batch_count, e)

        return batch_count

    def get_stats(self) -> Dict[str, Any]:
        """Get reader statistics.

        Returns:
            Dictionary with stats.
        """
        return {
            "current_source": self._current_source,
            "read_position": self._read_position,
            "total_read": self._total_read,
        }

    def reset_stats(self) -> None:
        """Reset reading statistics."""
        self._read_position = 0
        self._total_read = 0
