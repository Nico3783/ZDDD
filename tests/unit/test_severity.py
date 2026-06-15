"""Unit tests for src/detection_engine/severity.py — SeverityCalculator."""

import pytest

from src.detection_engine.severity import SeverityCalculator


class TestSeverityCalculatorDefaults:
    """Tests using default threshold configuration."""

    def setup_method(self):
        self.calc = SeverityCalculator()

    def test_low_severity_score_0_2(self):
        assert self.calc.compute(0.2) == "low"

    def test_medium_severity_score_0_6(self):
        assert self.calc.compute(0.6) == "medium"

    def test_high_severity_score_0_8(self):
        assert self.calc.compute(0.8) == "high"

    def test_critical_severity_score_0_95(self):
        assert self.calc.compute(0.95) == "critical"

    def test_boundary_exact_0_9_returns_critical(self):
        assert self.calc.compute(0.9) == "critical"

    def test_boundary_exact_0_5_returns_medium(self):
        assert self.calc.compute(0.5) == "medium"

    def test_boundary_exact_0_7_returns_high(self):
        assert self.calc.compute(0.7) == "high"

    def test_score_0_0_returns_low(self):
        assert self.calc.compute(0.0) == "low"

    def test_score_1_0_returns_critical(self):
        assert self.calc.compute(1.0) == "critical"

    def test_score_below_threshold_returns_low(self):
        assert self.calc.compute(0.49) == "low"

    def test_score_0_49_is_low(self):
        assert self.calc.compute(0.49) == "low"

    def test_score_0_51_is_medium(self):
        assert self.calc.compute(0.51) == "medium"

    def test_score_0_69_is_medium(self):
        assert self.calc.compute(0.69) == "medium"

    def test_score_0_71_is_high(self):
        assert self.calc.compute(0.71) == "high"

    def test_score_0_89_is_high(self):
        assert self.calc.compute(0.89) == "high"


class TestSeverityCalculatorZeroDay:
    """Tests for is_zero_day=True boost behaviour."""

    def setup_method(self):
        self.calc = SeverityCalculator()

    def test_zero_day_boosts_score_by_0_2(self):
        # 0.8 + 0.2 = 1.0 -> critical
        assert self.calc.compute(0.8, is_zero_day=True) == "critical"

    def test_zero_day_caps_at_1_0(self):
        # 0.9 + 0.2 = 1.1 -> capped to 1.0 -> critical
        result = self.calc.compute(0.9, is_zero_day=True)
        assert result == "critical"

    def test_zero_day_score_1_0_stays_1_0(self):
        assert self.calc.compute(1.0, is_zero_day=True) == "critical"

    def test_zero_day_medium_becomes_high(self):
        # 0.6 + 0.2 = 0.8 -> high
        assert self.calc.compute(0.6, is_zero_day=True) == "high"

    def test_zero_day_low_becomes_medium(self):
        # 0.4 + 0.2 = 0.6 -> medium
        assert self.calc.compute(0.4, is_zero_day=True) == "medium"

    def test_zero_day_high_becomes_critical(self):
        # 0.7 + 0.2 = 0.9 -> critical
        assert self.calc.compute(0.7, is_zero_day=True) == "critical"

    def test_zero_day_low_score_stays_low(self):
        # 0.1 + 0.2 = 0.3 -> low
        assert self.calc.compute(0.1, is_zero_day=True) == "low"

    def test_zero_day_boundary_medium_to_high(self):
        # 0.5 + 0.2 = 0.7 -> high (boundary)
        assert self.calc.compute(0.5, is_zero_day=True) == "high"

    def test_zero_day_score_0_88_becomes_critical(self):
        # 0.88 + 0.2 = 1.08 -> capped 1.0 -> critical
        assert self.calc.compute(0.88, is_zero_day=True) == "critical"

    def test_zero_day_does_not_mutate_original_score(self):
        score = 0.3
        self.calc.compute(score, is_zero_day=True)
        assert score == 0.3


class TestSeverityCalculatorCustomThresholds:
    """Tests with user-supplied threshold dictionaries."""

    def test_custom_thresholds_raise_above_critical(self):
        calc = SeverityCalculator()
        calc.severity_levels = {"critical": 0.8, "high": 0.6, "medium": 0.3, "low": 0.0}
        assert calc.compute(0.85) == "critical"

    def test_custom_thresholds_high_band(self):
        calc = SeverityCalculator()
        calc.severity_levels = {"critical": 0.8, "high": 0.6, "medium": 0.3, "low": 0.0}
        assert calc.compute(0.7) == "high"

    def test_custom_thresholds_medium_band(self):
        calc = SeverityCalculator()
        calc.severity_levels = {"critical": 0.8, "high": 0.6, "medium": 0.3, "low": 0.0}
        assert calc.compute(0.5) == "medium"

    def test_custom_thresholds_low_band(self):
        calc = SeverityCalculator()
        calc.severity_levels = {"critical": 0.8, "high": 0.6, "medium": 0.3, "low": 0.0}
        assert calc.compute(0.2) == "low"

    def test_custom_thresholds_tight_critical(self):
        calc = SeverityCalculator()
        calc.severity_levels = {"critical": 0.99, "high": 0.9, "medium": 0.8, "low": 0.0}
        assert calc.compute(0.95) == "high"

    def test_custom_thresholds_with_zero_day_boost(self):
        calc = SeverityCalculator()
        calc.severity_levels = {"critical": 0.9, "high": 0.7, "medium": 0.5, "low": 0.0}
        # 0.4 + 0.2 = 0.6 -> medium with custom medium=0.5
        assert calc.compute(0.4, is_zero_day=True) == "medium"

    def test_custom_thresholds_boundary_exact_critical(self):
        calc = SeverityCalculator()
        calc.severity_levels = {"critical": 0.85, "high": 0.6, "medium": 0.3, "low": 0.0}
        assert calc.compute(0.85) == "critical"

    def test_custom_thresholds_boundary_exact_high(self):
        calc = SeverityCalculator()
        calc.severity_levels = {"critical": 0.85, "high": 0.6, "medium": 0.3, "low": 0.0}
        assert calc.compute(0.6) == "high"

    def test_custom_thresholds_boundary_exact_medium(self):
        calc = SeverityCalculator()
        calc.severity_levels = {"critical": 0.85, "high": 0.6, "medium": 0.3, "low": 0.0}
        assert calc.compute(0.3) == "medium"


class TestSeverityCalculatorEdgeCases:
    """Edge-case and robustness tests."""

    def setup_method(self):
        self.calc = SeverityCalculator()

    def test_negative_score_returns_low(self):
        assert self.calc.compute(-0.1) == "low"

    def test_score_above_1_0_treated_as_critical(self):
        # 1.1 >= 0.9 -> critical (no capping on input, only on zero-day boost)
        assert self.calc.compute(1.1) == "critical"

    def test_zero_day_negative_score_still_low(self):
        # -0.1 + 0.2 = 0.1 -> low
        assert self.calc.compute(-0.1, is_zero_day=True) == "low"

    def test_zero_day_default_false(self):
        # Explicit False should behave like no boost
        assert self.calc.compute(0.8, is_zero_day=False) == "high"

    def test_repeated_calls_are_independent(self):
        assert self.calc.compute(0.2) == "low"
        assert self.calc.compute(0.95) == "critical"
        assert self.calc.compute(0.6) == "medium"

    def test_multiple_instances_independent(self):
        c1 = SeverityCalculator()
        c2 = SeverityCalculator()
        c2.severity_levels = {"critical": 0.5, "high": 0.3, "medium": 0.1, "low": 0.0}
        assert c1.compute(0.6) == "medium"
        assert c2.compute(0.6) == "critical"

    def test_empty_thresholds_dict_does_not_crash(self):
        calc = SeverityCalculator()
        calc.severity_levels = {}
        # With empty thresholds, .get() returns defaults, so 0.95 >= 0.9 -> critical
        result = calc.compute(0.95)
        assert result in ("low", "medium", "high", "critical")
