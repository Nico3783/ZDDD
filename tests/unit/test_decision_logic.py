"""Unit tests for src/detection_engine/decision_logic.py — DetectionDecisionMaker."""

import pytest

from src.detection_engine.decision_logic import DetectionDecisionMaker


class TestDetectionDecisionMakerDefaults:
    """Tests using default threshold configuration."""

    def setup_method(self):
        self.dm = DetectionDecisionMaker()

    def test_not_anomaly_returns_benign(self):
        result = self.dm.decide(
            is_anomaly=False,
            rf_class="DoS Hulk",
            rf_confidence=0.9,
        )
        assert result.is_zero_day is False
        assert result.predicted_class == "BENIGN"
        assert result.decision_reason == "not_anomalous"

    def test_anomaly_known_attack_not_zero_day(self):
        result = self.dm.decide(
            is_anomaly=True,
            rf_class="DoS Hulk",
            rf_confidence=0.9,
        )
        assert result.is_zero_day is False
        assert result.predicted_class == "DoS Hulk"
        assert result.decision_reason == "known_attack"

    def test_anomaly_benign_class_is_zero_day(self):
        result = self.dm.decide(
            is_anomaly=True,
            rf_class="BENIGN",
            rf_confidence=0.9,
        )
        assert result.is_zero_day is True
        assert result.predicted_class == "ZERO_DAY"
        assert result.decision_reason == "anomaly_classified_as_benign"

    def test_anomaly_low_confidence_is_zero_day(self):
        result = self.dm.decide(
            is_anomaly=True,
            rf_class="DoS Hulk",
            rf_confidence=0.3,
        )
        assert result.is_zero_day is True
        assert result.predicted_class == "ZERO_DAY"
        assert result.decision_reason == "low_classification_confidence"

    def test_zero_day_decision_reason_is_classified_as_benign(self):
        result = self.dm.decide(
            is_anomaly=True,
            rf_class="BENIGN",
            rf_confidence=0.9,
        )
        assert result.decision_reason == "anomaly_classified_as_benign"

    def test_known_attack_decision_reason(self):
        result = self.dm.decide(
            is_anomaly=True,
            rf_class="DoS Hulk",
            rf_confidence=0.9,
        )
        assert result.decision_reason == "known_attack"

    def test_returns_required_fields(self):
        result = self.dm.decide(
            is_anomaly=True,
            rf_class="DoS Hulk",
            rf_confidence=0.9,
        )
        required_fields = {
            "predicted_class",
            "classification_confidence",
            "class_probabilities",
            "is_zero_day",
            "decision_reason",
        }
        assert required_fields.issubset(
            {f.name for f in result.__dataclass_fields__.values()}
        )

    def test_confidence_field_is_float(self):
        result = self.dm.decide(
            is_anomaly=True,
            rf_class="DoS Hulk",
            rf_confidence=0.9,
        )
        assert isinstance(result.classification_confidence, float)

    def test_reasoning_is_non_empty_string(self):
        result = self.dm.decide(
            is_anomaly=True,
            rf_class="DoS Hulk",
            rf_confidence=0.9,
        )
        assert isinstance(result.decision_reason, str)
        assert len(result.decision_reason) > 0

    def test_valid_decision_reason(self):
        result = self.dm.decide(
            is_anomaly=True,
            rf_class="DoS Hulk",
            rf_confidence=0.9,
        )
        assert result.decision_reason in (
            "not_anomalous",
            "anomaly_no_classifier",
            "anomaly_classified_as_benign",
            "low_classification_confidence",
            "known_attack",
        )


class TestDetectionDecisionMakerZeroDayLogic:
    """Tests focused on zero-day detection logic paths."""

    def setup_method(self):
        self.dm = DetectionDecisionMaker()

    def test_zero_day_when_class_is_benign(self):
        result = self.dm.decide(
            is_anomaly=True,
            rf_class="BENIGN",
            rf_confidence=0.95,
        )
        assert result.is_zero_day is True

    def test_zero_day_when_confidence_below_minimum(self):
        result = self.dm.decide(
            is_anomaly=True,
            rf_class="DoS Hulk",
            rf_confidence=0.4,
        )
        assert result.is_zero_day is True

    def test_not_zero_day_when_known_attack_high_confidence(self):
        result = self.dm.decide(
            is_anomaly=True,
            rf_class="DoS Hulk",
            rf_confidence=0.95,
        )
        assert result.is_zero_day is False

    def test_not_anomaly_prevents_zero_day(self):
        result = self.dm.decide(
            is_anomaly=False,
            rf_class="BENIGN",
            rf_confidence=0.1,
        )
        assert result.is_zero_day is False
        assert result.predicted_class == "BENIGN"

    def test_zero_day_confidence_exactly_at_minimum(self):
        # At exactly minimum_confidence (0.5), not below -> not zero_day from confidence
        # But class=DoS Hulk, so not zero_day
        result = self.dm.decide(
            is_anomaly=True,
            rf_class="DoS Hulk",
            rf_confidence=0.5,
        )
        assert result.is_zero_day is False

    def test_zero_day_confidence_just_below_minimum(self):
        result = self.dm.decide(
            is_anomaly=True,
            rf_class="DoS Hulk",
            rf_confidence=0.49,
        )
        assert result.is_zero_day is True


class TestDetectionDecisionMakerNoneProbabilities:
    """Tests for rf_probabilities=None and edge cases."""

    def setup_method(self):
        self.dm = DetectionDecisionMaker()

    def test_none_probabilities_does_not_crash(self):
        result = self.dm.decide(
            is_anomaly=True,
            rf_class="DoS Hulk",
            rf_confidence=0.9,
            rf_probabilities=None,
        )
        assert result.is_zero_day is False

    def test_empty_probabilities_dict(self):
        result = self.dm.decide(
            is_anomaly=True,
            rf_class="DoS Hulk",
            rf_confidence=0.9,
            rf_probabilities={},
        )
        assert result.is_zero_day is False

    def test_probabilities_with_benign_highest(self):
        result = self.dm.decide(
            is_anomaly=True,
            rf_class="Unknown",
            rf_confidence=0.6,
            rf_probabilities={"BENIGN": 0.7, "DoS Hulk": 0.3},
        )
        # "Unknown" != "BENIGN" and confidence 0.6 >= 0.5 -> known_attack path
        assert result.is_zero_day is False
        assert result.predicted_class == "Unknown"

    def test_probabilities_with_known_attack_highest(self):
        result = self.dm.decide(
            is_anomaly=True,
            rf_class="DoS Hulk",
            rf_confidence=0.9,
            rf_probabilities={"DoS Hulk": 0.85, "BENIGN": 0.15},
        )
        assert result.is_zero_day is False


class TestDetectionDecisionMakerAnomalyNoClassifier:
    """Tests for anomaly detected but no classifier available."""

    def setup_method(self):
        self.dm = DetectionDecisionMaker()

    def test_anomaly_no_classifier_is_zero_day(self):
        result = self.dm.decide(
            is_anomaly=True,
            rf_class=None,
            rf_confidence=None,
        )
        assert result.is_zero_day is True
        assert result.predicted_class == "UNKNOWN"
        assert result.decision_reason == "anomaly_no_classifier"


class TestDetectionDecisionMakerEdgeCases:
    """Edge-case and robustness tests."""

    def setup_method(self):
        self.dm = DetectionDecisionMaker()

    def test_not_anomaly_ignores_rf_params(self):
        result = self.dm.decide(
            is_anomaly=False,
            rf_class="DoS Hulk",
            rf_confidence=0.9,
        )
        assert result.predicted_class == "BENIGN"
        assert result.decision_reason == "not_anomalous"

    def test_anomaly_with_high_confidence_known_attack(self):
        result = self.dm.decide(
            is_anomaly=True,
            rf_class="DoS Hulk",
            rf_confidence=1.0,
        )
        assert result.is_zero_day is False
        assert result.predicted_class == "DoS Hulk"

    def test_confidence_zero_below_minimum(self):
        result = self.dm.decide(
            is_anomaly=True,
            rf_class="DoS Hulk",
            rf_confidence=0.0,
        )
        assert result.is_zero_day is True

    def test_confidence_one(self):
        result = self.dm.decide(
            is_anomaly=True,
            rf_class="DoS Hulk",
            rf_confidence=1.0,
        )
        assert result.is_zero_day is False

    def test_unknown_predicted_class_low_confidence(self):
        result = self.dm.decide(
            is_anomaly=True,
            rf_class="SomeNewAttack",
            rf_confidence=0.3,
        )
        assert result.is_zero_day is True
        assert result.predicted_class == "ZERO_DAY"

    def test_repeated_calls_are_independent(self):
        r1 = self.dm.decide(
            is_anomaly=False,
            rf_class="DoS Hulk",
            rf_confidence=0.9,
        )
        r2 = self.dm.decide(
            is_anomaly=True,
            rf_class="BENIGN",
            rf_confidence=0.9,
        )
        assert r1.is_zero_day is False
        assert r2.is_zero_day is True

    def test_multiple_instances_independent(self):
        dm1 = DetectionDecisionMaker()
        dm2 = DetectionDecisionMaker()
        r1 = dm1.decide(
            is_anomaly=False,
            rf_class="DoS Hulk",
            rf_confidence=0.9,
        )
        r2 = dm2.decide(
            is_anomaly=True,
            rf_class="DoS Hulk",
            rf_confidence=0.9,
        )
        assert r1.is_zero_day is False
        assert r2.is_zero_day is False
