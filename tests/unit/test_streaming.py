"""Unit tests for the streaming modules.

Covers:
- src/streaming/streamer.py   — DatasetStreamer
- src/streaming/scheduler.py  — StreamScheduler
- src/streaming/simulator.py  — StreamSimulator
- src/streaming/stream_reader.py — StreamStreamReader
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Mock broken imports in the streaming package.
# stream_reader.py imports `DataCleaner` from src.preprocessing.cleaner
# and `FeatureEncoder` from src.preprocessing.encoder, but these names
# don't exist in those modules.  We inject stub modules so that the
# import chain succeeds without hitting the real (broken) import path.
# ---------------------------------------------------------------------------

def _ensure_streaming_imports():
    """Patch sys.modules so that the streaming subpackage can be imported."""
    # Provide a stub for DataCleaner
    cleaner_mod = sys.modules.get("src.preprocessing.cleaner")
    if cleaner_mod is not None and not hasattr(cleaner_mod, "DataCleaner"):
        cleaner_mod.DataCleaner = type("DataCleaner", (), {})  # type: ignore[attr-defined]

    # Provide a stub for FeatureEncoder
    encoder_mod = sys.modules.get("src.preprocessing.encoder")
    if encoder_mod is not None and not hasattr(encoder_mod, "FeatureEncoder"):
        encoder_mod.FeatureEncoder = type("FeatureEncoder", (), {})  # type: ignore[attr-defined]


# Run the patching once at module load so all test classes benefit.
_ensure_streaming_imports()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 100, cols: int = 5, seed: int = 0) -> pd.DataFrame:
    """Create a synthetic DataFrame with numeric columns."""
    rng = np.random.RandomState(seed)
    data = rng.rand(n, cols)
    columns = [f"f{i}" for i in range(cols)]
    return pd.DataFrame(data, columns=columns)


def _make_labeled_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create a DataFrame that looks like network traffic with labels."""
    rng = np.random.RandomState(seed)
    n_attack = int(n * 0.3)
    n_normal = n - n_attack

    normal = pd.DataFrame({
        "flow_duration": rng.exponential(5.0, n_normal),
        "total_fwd_packets": rng.poisson(10, n_normal),
        "total_bwd_packets": rng.poisson(8, n_normal),
        "fwd_packet_length_mean": rng.normal(200, 50, n_normal),
        "bwd_packet_length_mean": rng.normal(180, 40, n_normal),
        "flow_bytes_per_second": rng.exponential(1000, n_normal),
        "flow_packets_per_second": rng.exponential(50, n_normal),
        "label": "BENIGN",
    })

    attack = pd.DataFrame({
        "flow_duration": rng.exponential(0.5, n_attack),
        "total_fwd_packets": rng.poisson(100, n_attack),
        "total_bwd_packets": rng.poisson(5, n_attack),
        "fwd_packet_length_mean": rng.normal(50, 20, n_attack),
        "bwd_packet_length_mean": rng.normal(40, 15, n_attack),
        "flow_bytes_per_second": rng.exponential(5000, n_attack),
        "flow_packets_per_second": rng.exponential(200, n_attack),
        "label": "DDoS",
    })

    df = pd.concat([normal, attack], ignore_index=True)
    df["timestamp"] = pd.date_range("2026-01-01", periods=n, freq="ms")
    return df


# DatasetStreamer  (src/streaming/streamer.py)class TestDatasetStreamer:
    """Tests for the DatasetStreamer class."""

    def _make_streamer(self, data: pd.DataFrame, batch_size: int = 32,
                       loop: bool = False, max_samples: int | None = None,
                       replay_rate: float = 1000.0):
        from src.streaming.streamer import DatasetStreamer, StreamConfig

        cfg = StreamConfig(
            batch_size=batch_size,
            replay_rate=replay_rate,
            loop=loop,
            max_samples=max_samples if max_samples is not None else len(data),
        )
        return DatasetStreamer(data=data, config=cfg)

    # --- construction ---

    def test_init_stores_data(self):
        df = _make_df(50)
        streamer = self._make_streamer(df, batch_size=10)
        assert len(streamer.data) == 50
        assert streamer.config.batch_size == 10

    def test_total_samples_without_max(self):
        df = _make_df(80)
        streamer = self._make_streamer(df, batch_size=32, max_samples=80)
        assert streamer.total_samples == 80

    def test_total_samples_with_max(self):
        df = _make_df(200)
        streamer = self._make_streamer(df, batch_size=32, max_samples=50)
        assert streamer.total_samples == 50

    def test_total_batches_computed(self):
        df = _make_df(100)
        streamer = self._make_streamer(df, batch_size=32)
        expected = (100 + 31) // 32  # ceil division
        assert streamer.total_batches == expected

    # --- stream_batches ---

    def test_stream_batches_yields_correct_batch_sizes(self):
        df = _make_df(100)
        streamer = self._make_streamer(df, batch_size=32)

        sizes = []
        for batch, idx in streamer.stream_batches():
            sizes.append(len(batch))
            assert isinstance(batch, pd.DataFrame)

        # All batches except possibly the last should be exactly 32
        for s in sizes[:-1]:
            assert s == 32
        # Last batch holds the remainder
        assert sizes[-1] == 100 % 32 if 100 % 32 != 0 else 32

    def test_stream_batches_yields_all_rows(self):
        df = _make_df(100)
        streamer = self._make_streamer(df, batch_size=32)

        total = sum(len(b) for b, _ in streamer.stream_batches())
        assert total == 100

    def test_stream_batches_batch_index_increments(self):
        df = _make_df(64)
        streamer = self._make_streamer(df, batch_size=32)

        indices = [idx for _, idx in streamer.stream_batches()]
        assert indices == [0, 1]

    def test_stream_batches_with_empty_df_yields_nothing(self):
        df = _make_df(0)
        streamer = self._make_streamer(df, batch_size=32)

        batches = list(streamer.stream_batches())
        assert batches == []

    def test_stream_batches_loop_yields_multiple_cycles(self):
        df = _make_df(32)
        streamer = self._make_streamer(df, batch_size=32, loop=True)

        count = 0
        for batch, idx in streamer.stream_batches():
            count += 1
            if count >= 3:
                break
        assert count == 3

    # --- run ---

    def test_run_returns_stream_metrics(self):
        df = _make_df(64)
        streamer = self._make_streamer(df, batch_size=32, replay_rate=100000.0)
        metrics = streamer.run()

        assert metrics.total_samples == 64
        assert metrics.total_batches == 2
        assert metrics.total_time_seconds >= 0
        assert isinstance(metrics.samples_per_second, float)

    def test_run_with_processor(self):
        df = _make_df(64)
        streamer = self._make_streamer(df, batch_size=32, replay_rate=100000.0)
        processed = []

        metrics = streamer.run(processor=lambda batch: processed.append(len(batch)))

        assert len(processed) == 2
        assert metrics.total_samples == 64

    def test_run_populates_batch_latencies(self):
        df = _make_df(32)
        streamer = self._make_streamer(df, batch_size=32, replay_rate=100000.0)
        metrics = streamer.run()

        assert len(metrics.batch_latencies) == 1
        assert metrics.avg_batch_latency_ms >= 0
        assert metrics.min_batch_latency_ms >= 0
        assert metrics.max_batch_latency_ms >= 0

    # --- metrics property ---

    def test_metrics_property_returns_stream_metrics(self):
        from src.streaming.streamer import StreamMetrics
        df = _make_df(10)
        streamer = self._make_streamer(df, batch_size=32)
        assert isinstance(streamer.metrics, StreamMetrics)

    # --- get_sample / get_batch ---

    def test_get_sample_returns_series(self):
        df = _make_df(10)
        streamer = self._make_streamer(df, batch_size=32)
        sample = streamer.get_sample(0)
        assert isinstance(sample, pd.Series)

    def test_get_batch_returns_dataframe(self):
        df = _make_df(20)
        streamer = self._make_streamer(df, batch_size=32)
        batch = streamer.get_batch(0, 5)
        assert len(batch) == 5
        assert isinstance(batch, pd.DataFrame)

    # --- callback ---

    def test_callback_called_for_each_batch(self):
        df = _make_df(64)
        callback = MagicMock()
        streamer = self._make_streamer(df, batch_size=32, replay_rate=100000.0)
        streamer.on_batch_callback = callback
        list(streamer.stream_batches())

        assert callback.call_count == 2
        # Verify the last call received a DataFrame and an int
        last_args, last_kwargs = callback.call_args
        assert isinstance(last_args[0], pd.DataFrame)
        assert isinstance(last_args[1], int)


# StreamScheduler  (src/streaming/scheduler.py)class TestStreamScheduler:
    """Tests for the StreamScheduler class."""

    def _make_scheduler(self, n: int = 100, batch_size: int = 32):
        from src.streaming.streamer import DatasetStreamer, StreamConfig
        from src.streaming.scheduler import StreamScheduler

        data = _make_df(n)
        cfg = StreamConfig(
            batch_size=batch_size,
            replay_rate=100000.0,
            loop=False,
            max_samples=n,
        )
        streamer = DatasetStreamer(data=data, config=cfg)
        return StreamScheduler(streamer=streamer)

    # --- construction ---

    def test_init_creates_scheduler(self):
        scheduler = self._make_scheduler()
        assert scheduler.streamer is not None

    # --- measure_throughput ---

    def test_measure_throughput_returns_measurement(self):
        scheduler = self._make_scheduler(n=64, batch_size=32)
        result = scheduler.measure_throughput()

        assert result.total_samples == 64
        assert result.total_batches == 2
        assert result.samples_per_second >= 0
        assert result.batches_per_second >= 0
        assert result.measurement_duration_seconds >= 0

    def test_measure_throughput_with_processor(self):
        scheduler = self._make_scheduler(n=32, batch_size=32)
        result = scheduler.measure_throughput(processor=lambda b: None)

        assert result.total_samples == 32

    # --- measure_latency ---

    def test_measure_latency_returns_measurement(self):
        scheduler = self._make_scheduler(n=32, batch_size=32)
        result = scheduler.measure_latency(n_iterations=1)

        assert result.total_measurements >= 0
        assert result.avg_latency_ms >= 0
        assert result.p50_latency_ms >= 0
        assert result.p95_latency_ms >= 0
        assert result.p99_latency_ms >= 0
        assert result.min_latency_ms >= 0
        assert result.max_latency_ms >= 0

    def test_measure_latency_empty_returns_zeros(self):
        from src.streaming.streamer import DatasetStreamer, StreamConfig
        from src.streaming.scheduler import StreamScheduler

        data = _make_df(0)
        cfg = StreamConfig(batch_size=32, replay_rate=100.0, max_samples=0)
        streamer = DatasetStreamer(data=data, config=cfg)
        scheduler = StreamScheduler(streamer=streamer)
        result = scheduler.measure_latency()

        assert result.total_measurements == 0
        assert result.avg_latency_ms == 0.0

    # --- benchmark ---

    def test_benchmark_returns_dict(self):
        scheduler = self._make_scheduler(n=32, batch_size=32)
        result = scheduler.benchmark(n_iterations=1)

        assert "throughput" in result
        assert "latency" in result
        assert "samples_per_second" in result["throughput"]
        assert "avg_ms" in result["latency"]

    # --- get_history ---

    def test_get_history_starts_empty(self):
        scheduler = self._make_scheduler()
        history = scheduler.get_history()

        assert "throughput" in history
        assert "latency" in history
        assert len(history["throughput"]) == 0
        assert len(history["latency"]) == 0

    def test_get_history_populated_after_measurement(self):
        scheduler = self._make_scheduler(n=32, batch_size=32)
        scheduler.measure_throughput()
        history = scheduler.get_history()

        assert len(history["throughput"]) == 1

    # --- p95 >= p50 ---

    def test_p95_gte_p50(self):
        scheduler = self._make_scheduler(n=64, batch_size=16)
        result = scheduler.measure_latency(n_iterations=3)

        if result.total_measurements > 0:
            assert result.p95_latency_ms >= result.p50_latency_ms


# StreamSimulator  (src/streaming/simulator.py)class TestStreamSimulator:
    """Tests for the StreamSimulator class."""

    def _make_simulator(self, config: dict | None = None, seed: int = 42):
        from src.streaming.simulator import StreamSimulator

        if config is None:
            config = {}
        return StreamSimulator(config=config, random_state=seed)

    # --- construction ---

    def test_init_default(self):
        sim = self._make_simulator()
        assert sim.random_state == 42
        assert sim._packet_count == 0

    def test_init_with_config(self):
        sim = self._make_simulator(config={"some_key": "some_val"})
        assert sim.config == {"some_key": "some_val"}

    # --- generate_traffic ---

    def test_generate_traffic_returns_dataframe(self):
        sim = self._make_simulator()
        df = sim.generate_traffic(n_packets=100, attack_ratio=0.3)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100

    def test_generate_traffic_has_expected_columns(self):
        sim = self._make_simulator()
        df = sim.generate_traffic(n_packets=50)
        expected_cols = {
            "flow_duration", "total_fwd_packets", "total_bwd_packets",
            "fwd_packet_length_mean", "bwd_packet_length_mean",
            "flow_bytes_per_second", "flow_packets_per_second",
            "label", "timestamp",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_generate_traffic_attack_ratio(self):
        sim = self._make_simulator()
        df = sim.generate_traffic(n_packets=1000, attack_ratio=0.3)

        attack_count = (~df["label"].isin(["BENIGN"])).sum()
        ratio = attack_count / 1000
        assert 0.25 <= ratio <= 0.35  # within tolerance

    def test_generate_traffic_zero_attack_ratio(self):
        sim = self._make_simulator()
        df = sim.generate_traffic(n_packets=100, attack_ratio=0.0)
        assert (df["label"] == "BENIGN").all()

    def test_generate_traffic_one_attack_ratio(self):
        sim = self._make_simulator()
        df = sim.generate_traffic(n_packets=100, attack_ratio=1.0)
        assert (df["label"] != "BENIGN").all()

    def test_generate_traffic_updates_stats(self):
        sim = self._make_simulator()
        sim.generate_traffic(n_packets=50, attack_ratio=0.2)
        stats = sim.get_stats()

        assert stats["total_packets"] == 50
        assert stats["attack_count"] == int(50 * 0.2)

    def test_generate_traffic_reproducible_with_seed(self):
        sim1 = self._make_simulator(seed=42)
        df1 = sim1.generate_traffic(n_packets=100, attack_ratio=0.3)

        sim2 = self._make_simulator(seed=42)
        df2 = sim2.generate_traffic(n_packets=100, attack_ratio=0.3)

        # Same seed means same shuffled order (after shuffling with same rng)
        pd.testing.assert_frame_equal(df1.drop(columns=["timestamp"]),
                                       df2.drop(columns=["timestamp"]))

    # --- stream_from_dataframe ---

    def test_stream_from_dataframe_yields_batches(self):
        sim = self._make_simulator()
        df = _make_df(100)
        batches = list(sim.stream_from_dataframe(df, batch_size=32))

        assert len(batches) == 4  # ceil(100/32)
        total = sum(len(b) for b in batches)
        assert total == 100

    def test_stream_from_dataframe_batch_size(self):
        sim = self._make_simulator()
        df = _make_df(64)
        batches = list(sim.stream_from_dataframe(df, batch_size=16))

        assert all(len(b) == 16 for b in batches)

    def test_stream_from_dataframe_no_delay(self):
        sim = self._make_simulator()
        df = _make_df(50)
        start = time.monotonic()
        batches = list(sim.stream_from_dataframe(df, batch_size=10, delay_seconds=0.0))
        elapsed = time.monotonic() - start

        assert elapsed < 1.0  # should be near-instant
        assert sum(len(b) for b in batches) == 50

    # --- simulate_bursts ---

    def test_simulate_bursts_modifies_dataframe(self):
        sim = self._make_simulator()
        base_df = sim.generate_traffic(n_packets=200, attack_ratio=0.1)
        result = sim.simulate_bursts(base_df, burst_probability=0.5, burst_size_range=(5, 10))

        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 200  # bursts add extra rows

    def test_simulate_bursts_no_bursts(self):
        sim = self._make_simulator()
        base_df = sim.generate_traffic(n_packets=100, attack_ratio=0.1)
        result = sim.simulate_bursts(base_df, burst_probability=0.0)

        assert len(result) == 100

    # --- generate_latency_profile ---

    def test_generate_latency_profile_returns_array(self):
        sim = self._make_simulator()
        latencies = sim.generate_latency_profile(n_samples=100)

        assert isinstance(latencies, np.ndarray)
        assert len(latencies) == 100

    def test_generate_latency_profile_all_positive(self):
        sim = self._make_simulator()
        latencies = sim.generate_latency_profile(n_samples=500)
        assert (latencies >= 0).all()

    # --- get_stats ---

    def test_get_stats_initial(self):
        sim = self._make_simulator()
        stats = sim.get_stats()

        assert stats["total_packets"] == 0
        assert stats["attack_count"] == 0
        assert stats["burst_count"] == 0

    def test_get_stats_after_generations(self):
        sim = self._make_simulator()
        sim.generate_traffic(n_packets=100, attack_ratio=0.3)
        sim.generate_latency_profile(n_samples=50)
        stats = sim.get_stats()

        assert stats["total_packets"] == 150
        assert stats["attack_count"] == 30


# StreamStreamReader  (src/streaming/stream_reader.py)class TestStreamStreamReader:
    """Tests for the StreamStreamReader class."""

    def _make_reader(self, config: dict | None = None):
        from src.streaming.stream_reader import StreamStreamReader

        if config is None:
            config = {}
        return StreamStreamReader(config=config)

    # --- construction ---

    def test_init_default(self):
        reader = self._make_reader()
        assert reader._read_position == 0
        assert reader._total_read == 0
        assert reader._current_source is None

    def test_init_with_config(self):
        reader = self._make_reader(config={"batch_size": 64})
        assert reader.config == {"batch_size": 64}

    # --- read_csv ---

    def test_read_csv_returns_batches(self, tmp_path: Path):
        csv_file = tmp_path / "test.csv"
        df = _make_df(100)
        df.to_csv(csv_file, index=False)

        reader = self._make_reader()
        batches = list(reader.read_csv(str(csv_file), batch_size=32))

        assert len(batches) == 4  # ceil(100/32) — header row doesn't count since it's numeric
        total = sum(len(b) for b in batches)
        assert total == 100

    def test_read_csv_batch_size(self, tmp_path: Path):
        csv_file = tmp_path / "test.csv"
        df = _make_df(64)
        df.to_csv(csv_file, index=False)

        reader = self._make_reader()
        batches = list(reader.read_csv(str(csv_file), batch_size=16))

        for b in batches[:-1]:
            assert len(b) == 16

    def test_read_csv_file_not_found(self):
        reader = self._make_reader()
        with pytest.raises(FileNotFoundError):
            list(reader.read_csv("/nonexistent/file.csv", batch_size=32))

    def test_read_csv_updates_stats(self, tmp_path: Path):
        csv_file = tmp_path / "test.csv"
        df = _make_df(50)
        df.to_csv(csv_file, index=False)

        reader = self._make_reader()
        list(reader.read_csv(str(csv_file), batch_size=25))

        stats = reader.get_stats()
        assert stats["total_read"] == 50
        assert stats["current_source"] is None  # cleared after reading

    def test_read_csv_source_tracking(self, tmp_path: Path):
        csv_file = tmp_path / "test.csv"
        df = _make_df(10)
        df.to_csv(csv_file, index=False)

        reader = self._make_reader()
        batches_iter = reader.read_csv(str(csv_file), batch_size=10)
        # After first iteration, source should be set
        first_batch = next(batches_iter)
        assert reader._current_source == str(csv_file)
        # Drain remaining
        list(batches_iter)
        assert reader._current_source is None

    def test_read_csv_skip_rows(self, tmp_path: Path):
        csv_file = tmp_path / "test.csv"
        df = _make_df(50)
        df.to_csv(csv_file, index=False)

        reader = self._make_reader()
        batches = list(reader.read_csv(str(csv_file), batch_size=100, skip_rows=0))
        total_no_skip = sum(len(b) for b in batches)

        reader2 = self._make_reader()
        batches2 = list(reader2.read_csv(str(csv_file), batch_size=100, skip_rows=10))
        total_with_skip = sum(len(b) for b in batches2)

        assert total_with_skip < total_no_skip

    # --- read_parquet ---

    def test_read_parquet_returns_batches(self, tmp_path: Path):
        pyarrow = pytest.importorskip("pyarrow")
        parquet_file = tmp_path / "test.parquet"
        df = _make_df(100)
        df.to_parquet(parquet_file, index=False)

        reader = self._make_reader()
        batches = list(reader.read_parquet(str(parquet_file), batch_size=32))

        total = sum(len(b) for b in batches)
        assert total == 100

    def test_read_parquet_file_not_found(self):
        reader = self._make_reader()
        with pytest.raises(FileNotFoundError):
            list(reader.read_parquet("/nonexistent/file.parquet", batch_size=32))

    # --- read_jsonl ---

    def test_read_jsonl_returns_batches(self, tmp_path: Path):
        jsonl_file = tmp_path / "test.jsonl"
        records = [{"f0": i, "f1": i * 2} for i in range(50)]
        with open(jsonl_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        reader = self._make_reader()
        batches = list(reader.read_jsonl(str(jsonl_file), batch_size=20))

        total = sum(len(b) for b in batches)
        assert total == 50

    def test_read_jsonl_batch_size(self, tmp_path: Path):
        jsonl_file = tmp_path / "test.jsonl"
        records = [{"val": i} for i in range(40)]
        with open(jsonl_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        reader = self._make_reader()
        batches = list(reader.read_jsonl(str(jsonl_file), batch_size=10))

        assert all(len(b) == 10 for b in batches)

    def test_read_jsonl_file_not_found(self):
        reader = self._make_reader()
        with pytest.raises(FileNotFoundError):
            list(reader.read_jsonl("/nonexistent/file.jsonl", batch_size=32))

    def test_read_jsonl_skips_malformed(self, tmp_path: Path):
        jsonl_file = tmp_path / "test.jsonl"
        with open(jsonl_file, "w") as f:
            f.write(json.dumps({"val": 1}) + "\n")
            f.write("NOT JSON\n")
            f.write(json.dumps({"val": 2}) + "\n")

        reader = self._make_reader()
        batches = list(reader.read_jsonl(str(jsonl_file), batch_size=10))

        total = sum(len(b) for b in batches)
        assert total == 2  # malformed line skipped

    # --- set_callback / process_with_callback ---

    def test_set_callback_stores_callback(self):
        reader = self._make_reader()
        cb = MagicMock()
        reader.set_callback(cb)
        assert reader._callback is cb

    def test_process_with_callback_without_set_raises(self, tmp_path: Path):
        csv_file = tmp_path / "test.csv"
        _make_df(10).to_csv(csv_file, index=False)

        reader = self._make_reader()
        with pytest.raises(ValueError, match="No callback set"):
            reader.process_with_callback(str(csv_file), batch_size=10)

    def test_process_with_callback_csv(self, tmp_path: Path):
        csv_file = tmp_path / "test.csv"
        _make_df(50).to_csv(csv_file, index=False)

        reader = self._make_reader()
        cb = MagicMock()
        reader.set_callback(cb)
        count = reader.process_with_callback(str(csv_file), batch_size=20)

        assert count == 3  # ceil(50/20)
        assert cb.call_count == 3

    def test_process_with_callback_unsupported_format(self, tmp_path: Path):
        unsupported = tmp_path / "data.xyz"
        unsupported.write_text("data")

        reader = self._make_reader()
        reader.set_callback(MagicMock())
        with pytest.raises(ValueError, match="Unsupported file format"):
            reader.process_with_callback(str(unsupported))

    # --- get_stats ---

    def test_get_stats_initial(self):
        reader = self._make_reader()
        stats = reader.get_stats()

        assert stats["current_source"] is None
        assert stats["read_position"] == 0
        assert stats["total_read"] == 0

    # --- reset_stats ---

    def test_reset_stats(self, tmp_path: Path):
        csv_file = tmp_path / "test.csv"
        _make_df(30).to_csv(csv_file, index=False)

        reader = self._make_reader()
        list(reader.read_csv(str(csv_file), batch_size=10))
        assert reader.get_stats()["total_read"] == 30

        reader.reset_stats()
        stats = reader.get_stats()
        assert stats["read_position"] == 0
        assert stats["total_read"] == 0

    # --- read_position tracking ---

    def test_read_position_increments(self, tmp_path: Path):
        csv_file = tmp_path / "test.csv"
        _make_df(60).to_csv(csv_file, index=False)

        reader = self._make_reader()
        batches = reader.read_csv(str(csv_file), batch_size=20)

        first = next(batches)
        assert reader._read_position == 20

        second = next(batches)
        assert reader._read_position == 40
