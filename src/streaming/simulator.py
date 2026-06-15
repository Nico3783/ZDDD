from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Generator, List, Optional

import numpy as np
import pandas as pd

from src.core.config import get_config
from src.core.constants import RANDOM_SEED
from src.streaming.streamer import DatasetStreamer, StreamConfig

logger = logging.getLogger(__name__)


class StreamSimulator:
    """Realistic network traffic stream simulator.

    Generates synthetic traffic patterns with configurable attack injection,
    burst behavior, latency simulation, and packet loss for testing the
    detection pipeline under realistic conditions.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        random_state: int = RANDOM_SEED,
    ) -> None:
        """Initialize the stream simulator.

        Args:
            config: Simulator configuration dict. Loaded from config if None.
            random_state: Random state for reproducibility.
        """
        if config is None:
            try:
                self.config = get_config("streaming")
            except Exception:
                self.config = {}
        else:
            self.config = config

        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)
        self._packet_count = 0
        self._attack_count = 0
        self._burst_count = 0

        logger.info("StreamSimulator initialized")

    def generate_traffic(
        self,
        n_packets: int = 1000,
        attack_ratio: float = 0.1,
        burst_probability: float = 0.05,
        burst_size_range: tuple = (5, 20),
    ) -> pd.DataFrame:
        """Generate synthetic network traffic data.

        Args:
            n_packets: Total number of packets to generate.
            attack_ratio: Fraction of packets that are attack traffic.
            burst_probability: Probability of a burst occurring at each step.
            burst_size_range: Min/max packets in a burst.

        Returns:
            DataFrame with synthetic traffic data.
        """
        n_attack = int(n_packets * attack_ratio)
        n_normal = n_packets - n_attack

        normal_rows = self._generate_normal_traffic(n_normal)
        attack_rows = self._generate_attack_traffic(n_attack)

        all_rows = normal_rows + attack_rows
        self._rng.shuffle(all_rows)

        df = pd.DataFrame(all_rows)
        df["timestamp"] = pd.date_range(
            start=pd.Timestamp.now(),
            periods=n_packets,
            freq="ms",
        )

        self._packet_count += n_packets
        self._attack_count += n_attack
        logger.info(
            "Generated %d packets (%d normal, %d attack)",
            n_packets,
            n_normal,
            n_attack,
        )
        return df

    def stream_from_dataframe(
        self,
        df: pd.DataFrame,
        batch_size: int = 100,
        delay_seconds: float = 0.0,
        jitter: float = 0.0,
    ) -> Generator[pd.DataFrame, None, None]:
        """Stream data from a DataFrame in batches.

        Args:
            df: Source DataFrame.
            batch_size: Number of rows per batch.
            delay_seconds: Base delay between batches.
            jitter: Random jitter added to delay (±seconds).

        Yields:
            DataFrame batches.
        """
        n_rows = len(df)
        for start in range(0, n_rows, batch_size):
            end = min(start + batch_size, n_rows)
            batch = df.iloc[start:end].copy()

            if delay_seconds > 0:
                actual_delay = delay_seconds + self._rng.uniform(-jitter, jitter)
                time.sleep(max(0, actual_delay))

            yield batch

    def simulate_bursts(
        self,
        base_df: pd.DataFrame,
        burst_probability: float = 0.05,
        burst_size_range: tuple = (5, 20),
        burst_intensity: float = 2.0,
    ) -> pd.DataFrame:
        """Simulate burst traffic patterns on an existing DataFrame.

        Args:
            base_df: Base traffic DataFrame.
            burst_probability: Probability of a burst at each position.
            burst_size_range: Min/max packets per burst.
            burst_intensity: Multiplier for burst traffic volumes.

        Returns:
            DataFrame with burst traffic injected.
        """
        bursts = []
        n_rows = len(base_df)
        i = 0

        while i < n_rows:
            if self._rng.random() < burst_probability:
                burst_size = self._rng.randint(
                    burst_size_range[0], burst_size_range[1] + 1
                )
                burst_rows = self._generate_burst_traffic(burst_size, burst_intensity)
                bursts.extend(burst_rows)
                self._burst_count += 1
                i += burst_size
            else:
                bursts.append(base_df.iloc[i].to_dict())
                i += 1

        result = pd.DataFrame(bursts)
        if "timestamp" not in result.columns:
            result["timestamp"] = pd.date_range(
                start=pd.Timestamp.now(),
                periods=len(result),
                freq="ms",
            )

        logger.info("Simulated %d burst events", self._burst_count)
        return result

    def generate_latency_profile(
        self,
        n_samples: int = 1000,
        base_latency_ms: float = 1.0,
        spike_probability: float = 0.02,
        spike_range_ms: tuple = (10.0, 100.0),
    ) -> np.ndarray:
        """Generate a latency profile with occasional spikes.

        Args:
            n_samples: Number of latency samples.
            base_latency_ms: Base latency in milliseconds.
            spike_probability: Probability of a latency spike.
            spike_range_ms: Min/max latency during spikes.

        Returns:
            Array of latency values in milliseconds.
        """
        latencies = self._rng.exponential(base_latency_ms, n_samples)

        spike_mask = self._rng.random(n_samples) < spike_probability
        n_spikes = int(spike_mask.sum())

        if n_spikes > 0:
            spike_values = self._rng.uniform(
                spike_range_ms[0], spike_range_ms[1], n_spikes
            )
            latencies[spike_mask] = spike_values

        self._packet_count += n_samples
        return latencies

    def _generate_normal_traffic(self, n: int) -> List[Dict[str, Any]]:
        """Generate normal traffic rows."""
        rows = []
        for _ in range(n):
            row = {
                "flow_duration": float(self._rng.exponential(5.0)),
                "total_fwd_packets": int(self._rng.poisson(10)),
                "total_bwd_packets": int(self._rng.poisson(8)),
                "fwd_packet_length_mean": float(self._rng.normal(200, 50)),
                "bwd_packet_length_mean": float(self._rng.normal(180, 40)),
                "flow_bytes_per_second": float(self._rng.exponential(1000)),
                "flow_packets_per_second": float(self._rng.exponential(50)),
                "label": "BENIGN",
            }
            rows.append(row)
        return rows

    def _generate_attack_traffic(self, n: int) -> List[Dict[str, Any]]:
        """Generate attack traffic rows."""
        attack_types = ["DDoS", "DoS", "Web Attack", "Infiltration"]
        rows = []
        for _ in range(n):
            attack_type = self._rng.choice(attack_types)
            row = {
                "flow_duration": float(self._rng.exponential(0.5)),
                "total_fwd_packets": int(self._rng.poisson(100)),
                "total_bwd_packets": int(self._rng.poisson(5)),
                "fwd_packet_length_mean": float(self._rng.normal(50, 20)),
                "bwd_packet_length_mean": float(self._rng.normal(40, 15)),
                "flow_bytes_per_second": float(self._rng.exponential(5000)),
                "flow_packets_per_second": float(self._rng.exponential(200)),
                "label": attack_type,
            }
            rows.append(row)
        return rows

    def _generate_burst_traffic(
        self, size: int, intensity: float
    ) -> List[Dict[str, Any]]:
        """Generate burst traffic rows."""
        rows = []
        for _ in range(size):
            row = {
                "flow_duration": float(self._rng.exponential(0.1)),
                "total_fwd_packets": int(self._rng.poisson(50 * intensity)),
                "total_bwd_packets": int(self._rng.poisson(2)),
                "fwd_packet_length_mean": float(self._rng.normal(100, 30)),
                "bwd_packet_length_mean": float(self._rng.normal(60, 20)),
                "flow_bytes_per_second": float(self._rng.exponential(3000 * intensity)),
                "flow_packets_per_second": float(self._rng.exponential(150 * intensity)),
                "label": "Burst",
            }
            rows.append(row)
        return rows

    def get_stats(self) -> Dict[str, Any]:
        """Get simulation statistics.

        Returns:
            Dictionary with stats.
        """
        return {
            "total_packets": self._packet_count,
            "attack_count": self._attack_count,
            "burst_count": self._burst_count,
        }
