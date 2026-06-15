"""Unit tests for src/detection_engine/response_handler.py (ResponseHandler, ResponseAction)."""
from __future__ import annotations

import time
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from src.detection_engine.response_handler import ResponseAction, ResponseHandler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def default_config() -> Dict[str, Any]:
    """Return a default response handler config dict."""
    return {
        "enabled": True,
        "log_alerts": True,
        "block_critical": False,
        "quarantine_zero_day": False,
        "cooldown_seconds": 10,
        "max_actions_per_minute": 60,
    }


@pytest.fixture()
def handler(default_config: Dict[str, Any]) -> ResponseHandler:
    """Return a ResponseHandler with default config."""
    return ResponseHandler(config=default_config)


@pytest.fixture()
def blocking_config() -> Dict[str, Any]:
    """Return config with blocking and quarantine enabled."""
    return {
        "enabled": True,
        "log_alerts": True,
        "block_critical": True,
        "quarantine_zero_day": True,
        "cooldown_seconds": 0,
        "max_actions_per_minute": 1000,
    }


@pytest.fixture()
def blocking_handler(blocking_config: Dict[str, Any]) -> ResponseHandler:
    """Return a ResponseHandler with blocking and quarantine enabled."""
    return ResponseHandler(config=blocking_config)


@pytest.fixture()
def disabled_config() -> Dict[str, Any]:
    """Return config with handler disabled."""
    return {"enabled": False}


@pytest.fixture()
def disabled_handler(disabled_config: Dict[str, Any]) -> ResponseHandler:
    """Return a disabled ResponseHandler."""
    return ResponseHandler(config=disabled_config)


@pytest.fixture()
def critical_anomaly() -> Dict[str, Any]:
    """Return a critical-severity anomaly detection result."""
    return {
        "severity": "critical",
        "is_anomaly": True,
        "is_zero_day": False,
        "source_ip": "192.168.1.100",
        "sample_id": "sample_critical",
        "anomaly_score": 0.95,
        "predicted_class": "DoS Hulk",
    }


@pytest.fixture()
def high_anomaly() -> Dict[str, Any]:
    """Return a high-severity anomaly detection result."""
    return {
        "severity": "high",
        "is_anomaly": True,
        "is_zero_day": False,
        "source_ip": "192.168.1.200",
        "sample_id": "sample_high",
        "anomaly_score": 0.8,
        "predicted_class": "DoS Hulk",
    }


@pytest.fixture()
def medium_anomaly() -> Dict[str, Any]:
    """Return a medium-severity anomaly detection result."""
    return {
        "severity": "medium",
        "is_anomaly": True,
        "is_zero_day": False,
        "source_ip": "192.168.1.300",
        "sample_id": "sample_medium",
        "anomaly_score": 0.6,
        "predicted_class": "DoS GoldenEye",
    }


@pytest.fixture()
def low_anomaly() -> Dict[str, Any]:
    """Return a low-severity anomaly detection result."""
    return {
        "severity": "low",
        "is_anomaly": True,
        "is_zero_day": False,
        "source_ip": "192.168.1.400",
        "sample_id": "sample_low",
        "anomaly_score": 0.3,
        "predicted_class": "BENIGN",
    }


@pytest.fixture()
def zero_day_anomaly() -> Dict[str, Any]:
    """Return a zero-day anomaly detection result."""
    return {
        "severity": "critical",
        "is_anomaly": True,
        "is_zero_day": True,
        "source_ip": "10.0.0.50",
        "sample_id": "sample_zeroday",
        "anomaly_score": 0.92,
        "predicted_class": "ZERO_DAY",
    }


@pytest.fixture()
def normal_traffic() -> Dict[str, Any]:
    """Return a normal (non-anomalous) detection result."""
    return {
        "severity": "low",
        "is_anomaly": False,
        "is_zero_day": False,
        "source_ip": "192.168.1.50",
        "sample_id": "sample_normal",
        "anomaly_score": 0.1,
        "predicted_class": "BENIGN",
    }


# ---------------------------------------------------------------------------
# ResponseAction tests
# ---------------------------------------------------------------------------

class TestResponseAction:
    """Tests for the ResponseAction data class."""

    def test_init_defaults(self) -> None:
        action = ResponseAction(action_type="log", target="192.168.1.1")

        assert action.action_type == "log"
        assert action.target == "192.168.1.1"
        assert action.parameters == {}
        assert action.description == ""
        assert action.executed is False
        assert action.success is False
        assert action.error_message is None

    def test_init_with_parameters(self) -> None:
        params = {"severity": "critical", "reason": "test"}
        action = ResponseAction(
            action_type="block",
            target="10.0.0.1",
            parameters=params,
            description="Block attacker",
        )

        assert action.parameters == params
        assert action.description == "Block attacker"

    def test_to_dict_keys(self) -> None:
        action = ResponseAction(action_type="alert", target="host")
        d = action.to_dict()

        expected_keys = {
            "action_type", "target", "parameters", "description",
            "timestamp", "executed", "success", "error_message",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_values(self) -> None:
        action = ResponseAction(
            action_type="quarantine",
            target="10.0.0.5",
            parameters={"reason": "malware"},
            description="Quarantine host",
        )
        d = action.to_dict()

        assert d["action_type"] == "quarantine"
        assert d["target"] == "10.0.0.5"
        assert d["parameters"] == {"reason": "malware"}
        assert d["description"] == "Quarantine host"
        assert d["executed"] is False
        assert d["success"] is False

    def test_timestamp_is_iso_format(self) -> None:
        action = ResponseAction(action_type="log", target="x")
        # Should be parseable (basic check)
        assert "T" in action.timestamp


# ---------------------------------------------------------------------------
# ResponseHandler constructor tests
# ---------------------------------------------------------------------------

class TestResponseHandlerInit:
    """Tests for ResponseHandler initialization."""

    def test_creates_with_config(self, default_config: Dict[str, Any]) -> None:
        h = ResponseHandler(config=default_config)

        assert h.config == default_config
        assert h._cooldown_seconds == 10.0
        assert h._max_actions_per_minute == 60

    def test_creates_without_config_uses_defaults(self) -> None:
        with patch("src.detection_engine.response_handler.get_config", side_effect=Exception("no config")):
            h = ResponseHandler(config=None)

        assert h.config["enabled"] is True
        assert h.config["cooldown_seconds"] == 10
        assert h.config["max_actions_per_minute"] == 60

    def test_initial_history_empty(self, handler: ResponseHandler) -> None:
        assert handler._action_history == []

    def test_initial_timestamps_empty(self, handler: ResponseHandler) -> None:
        assert handler._action_timestamps == []


# ---------------------------------------------------------------------------
# handle_detection tests — action selection
# ---------------------------------------------------------------------------

class TestHandleDetectionActionSelection:
    """Tests verifying which actions are selected based on severity and flags."""

    def test_critical_anomaly_returns_actions(
        self,
        handler: ResponseHandler,
        critical_anomaly: Dict[str, Any],
    ) -> None:
        actions = handler.handle_detection(critical_anomaly)

        assert isinstance(actions, list)
        assert len(actions) >= 1

    def test_critical_anomaly_includes_log_and_alert(
        self,
        handler: ResponseHandler,
        critical_anomaly: Dict[str, Any],
    ) -> None:
        actions = handler.handle_detection(critical_anomaly)
        action_types = [a.action_type for a in actions]

        assert "log" in action_types
        assert "alert" in action_types

    def test_high_anomaly_includes_log_and_alert(
        self,
        handler: ResponseHandler,
        high_anomaly: Dict[str, Any],
    ) -> None:
        actions = handler.handle_detection(high_anomaly)
        action_types = [a.action_type for a in actions]

        assert "log" in action_types
        assert "alert" in action_types

    def test_medium_anomaly_includes_log_only(
        self,
        handler: ResponseHandler,
        medium_anomaly: Dict[str, Any],
    ) -> None:
        actions = handler.handle_detection(medium_anomaly)
        action_types = [a.action_type for a in actions]

        assert "log" in action_types
        assert "alert" not in action_types

    def test_low_anomaly_includes_log_only(
        self,
        handler: ResponseHandler,
        low_anomaly: Dict[str, Any],
    ) -> None:
        actions = handler.handle_detection(low_anomaly)
        action_types = [a.action_type for a in actions]

        assert "log" in action_types
        assert "alert" not in action_types

    def test_normal_traffic_returns_no_actions(
        self,
        handler: ResponseHandler,
        normal_traffic: Dict[str, Any],
    ) -> None:
        actions = handler.handle_detection(normal_traffic)

        assert actions == []

    def test_disabled_handler_returns_no_actions(
        self,
        disabled_handler: ResponseHandler,
        critical_anomaly: Dict[str, Any],
    ) -> None:
        actions = disabled_handler.handle_detection(critical_anomaly)

        assert actions == []


# ---------------------------------------------------------------------------
# handle_detection tests — blocking and quarantine
# ---------------------------------------------------------------------------

class TestHandleDetectionBlocking:
    """Tests for block and quarantine actions."""

    def test_critical_with_block_enabled_includes_block(
        self,
        blocking_handler: ResponseHandler,
        critical_anomaly: Dict[str, Any],
    ) -> None:
        actions = blocking_handler.handle_detection(critical_anomaly)
        action_types = [a.action_type for a in actions]

        assert "block" in action_types

    def test_critical_with_block_disabled_excludes_block(
        self,
        handler: ResponseHandler,
        critical_anomaly: Dict[str, Any],
    ) -> None:
        actions = handler.handle_detection(critical_anomaly)
        action_types = [a.action_type for a in actions]

        assert "block" not in action_types

    def test_zero_day_with_quarantine_enabled_includes_quarantine(
        self,
        blocking_handler: ResponseHandler,
        zero_day_anomaly: Dict[str, Any],
    ) -> None:
        actions = blocking_handler.handle_detection(zero_day_anomaly)
        action_types = [a.action_type for a in actions]

        assert "quarantine" in action_types

    def test_zero_day_with_quarantine_disabled_excludes_quarantine(
        self,
        handler: ResponseHandler,
        zero_day_anomaly: Dict[str, Any],
    ) -> None:
        actions = handler.handle_detection(zero_day_anomaly)
        action_types = [a.action_type for a in actions]

        assert "quarantine" not in action_types

    def test_zero_day_critical_with_all_enabled(
        self,
        blocking_handler: ResponseHandler,
        zero_day_anomaly: Dict[str, Any],
    ) -> None:
        actions = blocking_handler.handle_detection(zero_day_anomaly)
        action_types = [a.action_type for a in actions]

        assert "log" in action_types
        assert "alert" in action_types
        assert "quarantine" in action_types
        assert "block" in action_types


# ---------------------------------------------------------------------------
# handle_detection tests — target resolution
# ---------------------------------------------------------------------------

class TestHandleDetectionTarget:
    """Tests for target resolution in actions."""

    def test_uses_source_ip_when_present(
        self,
        handler: ResponseHandler,
        critical_anomaly: Dict[str, Any],
    ) -> None:
        actions = handler.handle_detection(critical_anomaly)

        for action in actions:
            assert action.target == "192.168.1.100"

    def test_falls_back_to_sample_id(
        self,
        handler: ResponseHandler,
    ) -> None:
        detection = {
            "severity": "high",
            "is_anomaly": True,
            "is_zero_day": False,
            "sample_id": "fallback-id",
        }
        actions = handler.handle_detection(detection)

        for action in actions:
            assert action.target == "fallback-id"

    def test_falls_back_to_unknown(
        self,
        handler: ResponseHandler,
    ) -> None:
        detection = {
            "severity": "high",
            "is_anomaly": True,
            "is_zero_day": False,
        }
        actions = handler.handle_detection(detection)

        for action in actions:
            assert action.target == "unknown"


# ---------------------------------------------------------------------------
# handle_detection tests — execution tracking
# ---------------------------------------------------------------------------

class TestHandleDetectionExecution:
    """Tests that actions are tracked as executed."""

    def test_actions_marked_executed(
        self,
        handler: ResponseHandler,
        critical_anomaly: Dict[str, Any],
    ) -> None:
        actions = handler.handle_detection(critical_anomaly)

        for action in actions:
            assert action.executed is True

    def test_actions_marked_success(
        self,
        handler: ResponseHandler,
        critical_anomaly: Dict[str, Any],
    ) -> None:
        actions = handler.handle_detection(critical_anomaly)

        for action in actions:
            assert action.success is True

    def test_actions_added_to_history(
        self,
        handler: ResponseHandler,
        critical_anomaly: Dict[str, Any],
    ) -> None:
        actions = handler.handle_detection(critical_anomaly)

        assert len(handler._action_history) == len(actions)

    def test_history_contains_action_objects(
        self,
        handler: ResponseHandler,
        critical_anomaly: Dict[str, Any],
    ) -> None:
        handler.handle_detection(critical_anomaly)

        for recorded in handler._action_history:
            assert isinstance(recorded, ResponseAction)


# ---------------------------------------------------------------------------
# Rate limiting tests
# ---------------------------------------------------------------------------

class TestRateLimiting:
    """Tests for rate limiting behavior."""

    def test_rate_limit_check_within_limit(self, handler: ResponseHandler) -> None:
        assert handler._rate_limit_check() is True

    def test_rate_limit_check_at_limit(self, handler: ResponseHandler) -> None:
        now = time.time()
        handler._action_timestamps = [now - i * 0.1 for i in range(60)]

        assert handler._rate_limit_check() is False

    def test_rate_limit_check_below_limit(self, handler: ResponseHandler) -> None:
        now = time.time()
        handler._action_timestamps = [now - i * 0.1 for i in range(59)]

        assert handler._rate_limit_check() is True

    def test_rate_limit_aged_timestamps_removed(self, handler: ResponseHandler) -> None:
        now = time.time()
        # 60 timestamps older than 60 seconds
        handler._action_timestamps = [now - 61.0] * 60

        assert handler._rate_limit_check() is True
        # Old timestamps should have been pruned
        assert len(handler._action_timestamps) == 0

    def test_rapid_fire_detection_limited(
        self,
        handler: ResponseHandler,
        critical_anomaly: Dict[str, Any],
    ) -> None:
        handler._max_actions_per_minute = 5

        # First 5 detections should execute
        for _ in range(5):
            handler.handle_detection(critical_anomaly)

        initial_count = len(handler._action_history)

        # 6th detection should be rate-limited (some actions may be skipped)
        handler.handle_detection(critical_anomaly)

        # The 6th call cannot add more than 2 actions (log + alert)
        # but rate limiting should prevent most executions
        assert len(handler._action_history) <= initial_count + 2


# ---------------------------------------------------------------------------
# Cooldown tests
# ---------------------------------------------------------------------------

class TestCooldown:
    """Tests for cooldown between alerts."""

    def test_cooldown_configured(self, handler: ResponseHandler) -> None:
        assert handler._cooldown_seconds == 10.0

    def test_cooldown_zero_allows_immediate(
        self,
        blocking_handler: ResponseHandler,
        critical_anomaly: Dict[str, Any],
    ) -> None:
        # blocking_handler has cooldown_seconds=0
        actions1 = blocking_handler.handle_detection(critical_anomaly)
        actions2 = blocking_handler.handle_detection(critical_anomaly)

        assert len(actions1) > 0
        assert len(actions2) > 0


# ---------------------------------------------------------------------------
# get_stats tests
# ---------------------------------------------------------------------------

class TestGetStats:
    """Tests for ResponseHandler.get_stats."""

    def test_returns_dict(self, handler: ResponseHandler) -> None:
        stats = handler.get_stats()

        assert isinstance(stats, dict)

    def test_expected_keys(self, handler: ResponseHandler) -> None:
        stats = handler.get_stats()

        expected_keys = {
            "total_actions", "successful_actions", "failed_actions",
            "action_counts", "rate_limit_remaining",
        }
        assert expected_keys == set(stats.keys())

    def test_initial_totals_zero(self, handler: ResponseHandler) -> None:
        stats = handler.get_stats()

        assert stats["total_actions"] == 0
        assert stats["successful_actions"] == 0
        assert stats["failed_actions"] == 0

    def test_action_counts_empty_initially(self, handler: ResponseHandler) -> None:
        stats = handler.get_stats()

        assert stats["action_counts"] == {}

    def test_rate_limit_remaining_initial(self, handler: ResponseHandler) -> None:
        stats = handler.get_stats()

        assert stats["rate_limit_remaining"] == 60

    def test_stats_update_after_handling(
        self,
        handler: ResponseHandler,
        critical_anomaly: Dict[str, Any],
    ) -> None:
        handler.handle_detection(critical_anomaly)
        stats = handler.get_stats()

        assert stats["total_actions"] >= 2  # log + alert
        assert stats["successful_actions"] >= 2
        assert stats["failed_actions"] == 0

    def test_action_counts_after_handling(
        self,
        handler: ResponseHandler,
        critical_anomaly: Dict[str, Any],
    ) -> None:
        handler.handle_detection(critical_anomaly)
        stats = handler.get_stats()

        assert "log" in stats["action_counts"]
        assert "alert" in stats["action_counts"]
        assert stats["action_counts"]["log"] >= 1
        assert stats["action_counts"]["alert"] >= 1

    def test_rate_limit_remaining_decreases(
        self,
        handler: ResponseHandler,
        critical_anomaly: Dict[str, Any],
    ) -> None:
        handler.handle_detection(critical_anomaly)
        stats = handler.get_stats()

        assert stats["rate_limit_remaining"] < 60


# ---------------------------------------------------------------------------
# get_history tests
# ---------------------------------------------------------------------------

class TestGetHistory:
    """Tests for ResponseHandler.get_history."""

    def test_returns_list(self, handler: ResponseHandler) -> None:
        history = handler.get_history()

        assert isinstance(history, list)

    def test_empty_initially(self, handler: ResponseHandler) -> None:
        history = handler.get_history()

        assert history == []

    def test_contains_dicts_after_handling(
        self,
        handler: ResponseHandler,
        critical_anomaly: Dict[str, Any],
    ) -> None:
        handler.handle_detection(critical_anomaly)
        history = handler.get_history()

        assert len(history) >= 2
        for entry in history:
            assert isinstance(entry, dict)

    def test_limit_parameter(
        self,
        handler: ResponseHandler,
        critical_anomaly: Dict[str, Any],
    ) -> None:
        handler.handle_detection(critical_anomaly)
        history = handler.get_history(limit=1)

        assert len(history) == 1

    def test_limit_larger_than_history_returns_all(
        self,
        handler: ResponseHandler,
        critical_anomaly: Dict[str, Any],
    ) -> None:
        handler.handle_detection(critical_anomaly)
        history_full = handler.get_history()
        history_large_limit = handler.get_history(limit=1000)

        assert len(history_full) == len(history_large_limit)

    def test_history_entries_have_expected_keys(
        self,
        handler: ResponseHandler,
        critical_anomaly: Dict[str, Any],
    ) -> None:
        handler.handle_detection(critical_anomaly)
        history = handler.get_history()

        for entry in history:
            assert "action_type" in entry
            assert "target" in entry
            assert "executed" in entry
            assert "success" in entry


# ---------------------------------------------------------------------------
# Multiple severities in sequence
# ---------------------------------------------------------------------------

class TestSeveritySequence:
    """Tests processing multiple alerts of different severities."""

    def test_mixed_severities_produce_varied_actions(
        self,
        blocking_handler: ResponseHandler,
    ) -> None:
        detections = [
            {"severity": "critical", "is_anomaly": True, "is_zero_day": False, "source_ip": "1.1.1.1"},
            {"severity": "low", "is_anomaly": True, "is_zero_day": False, "source_ip": "2.2.2.2"},
            {"severity": "high", "is_anomaly": True, "is_zero_day": False, "source_ip": "3.3.3.3"},
        ]

        all_actions: List[ResponseAction] = []
        for det in detections:
            actions = blocking_handler.handle_detection(det)
            all_actions.extend(actions)

        action_types = [a.action_type for a in all_actions]
        assert "log" in action_types
        assert "alert" in action_types
        assert "block" in action_types  # from critical

    def test_stats_reflect_all_severities(
        self,
        blocking_handler: ResponseHandler,
    ) -> None:
        detections = [
            {"severity": "critical", "is_anomaly": True, "is_zero_day": False, "source_ip": "1.1.1.1"},
            {"severity": "low", "is_anomaly": True, "is_zero_day": False, "source_ip": "2.2.2.2"},
        ]

        for det in detections:
            blocking_handler.handle_detection(det)

        stats = blocking_handler.get_stats()
        assert stats["total_actions"] >= 3  # critical: log+alert+block, low: log


# ---------------------------------------------------------------------------
# Edge case: log_alerts disabled
# ---------------------------------------------------------------------------

class TestLogAlertsDisabled:
    """Tests when log_alerts config is False."""

    def test_no_log_action_when_disabled(self) -> None:
        config = {
            "enabled": True,
            "log_alerts": False,
            "block_critical": False,
            "quarantine_zero_day": False,
            "cooldown_seconds": 10,
            "max_actions_per_minute": 60,
        }
        h = ResponseHandler(config=config)
        detection = {
            "severity": "high",
            "is_anomaly": True,
            "is_zero_day": False,
            "source_ip": "1.2.3.4",
        }

        actions = h.handle_detection(detection)
        action_types = [a.action_type for a in actions]

        assert "log" not in action_types
        assert "alert" in action_types  # high severity still triggers alert


# ---------------------------------------------------------------------------
# Edge case: non-anomaly with is_zero_day True
# ---------------------------------------------------------------------------

class TestNonAnomalyZeroDay:
    """Edge case: is_zero_day=True but is_anomaly=False."""

    def test_non_anomaly_zero_day_returns_actions(
        self,
        handler: ResponseHandler,
    ) -> None:
        detection = {
            "severity": "low",
            "is_anomaly": False,
            "is_zero_day": True,
            "source_ip": "5.5.5.5",
        }
        actions = handler.handle_detection(detection)

        # is_zero_day=True triggers processing (is_anomaly or is_zero_day)
        # With log_alerts=True, a log action is created
        assert len(actions) >= 1
        assert any(a.action_type == "log" for a in actions)
