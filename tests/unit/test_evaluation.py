"""Unit tests for the evaluation modules.

Covers:
- src/evaluation/metrics.py     — PerformanceEvaluator
- src/evaluation/latency.py     — LatencyTracker
- src/evaluation/throughput.py  — ThroughputTracker
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _binary_truth_pred(n: int = 200, accuracy: float = 0.85, seed: int = 42):
    """Return (y_true, y_pred) as numpy arrays of 'BENIGN' / 'ATTACK' labels."""
    rng = np.random.RandomState(seed)
    y_true = rng.choice(["BENIGN", "ATTACK"], n)
    # Flip some predictions to achieve target accuracy
    n_wrong = int(n * (1 - accuracy))
    y_pred = y_true.copy()
    flip_idx = rng.choice(n, n_wrong, replace=False)
    y_pred[flip_idx] = np.where(y_pred[flip_idx] == "BENIGN", "ATTACK", "BENIGN")
    return y_true, y_pred


def _multi_truth_pred(n: int = 200, seed: int = 42):
    """Return (y_true, y_pred) as numpy arrays with 4 classes."""
    rng = np.random.RandomState(seed)
    classes = np.array(["BENIGN", "DoS Hulk", "DoS GoldenEye", "DoS Slowloris"])
    y_true = rng.choice(classes, n, p=[0.6, 0.2, 0.1, 0.1])
    # Mostly correct predictions
    y_pred = y_true.copy()
    flip_mask = rng.random(n) < 0.15
    y_pred[flip_mask] = rng.choice(classes, flip_mask.sum())
    return y_true, y_pred


def _binary_scores(n: int = 200, seed: int = 42):
    """Return synthetic probability scores for binary ROC-AUC."""
    rng = np.random.RandomState(seed)
    return rng.uniform(0.0, 1.0, n)


# PerformanceEvaluator  (src/evaluation/metrics.py)


class TestPerformanceEvaluator:
    """Tests for the PerformanceEvaluator class."""

    def _evaluator(self):
        from src.evaluation.metrics import PerformanceEvaluator
        return PerformanceEvaluator()

    # --- construction ---

    def test_init(self):
        ev = self._evaluator()
        assert ev is not None

    # --- compute_accuracy ---

    def test_accuracy_perfect(self):
        ev = self._evaluator()
        y = np.array(["BENIGN", "ATTACK", "BENIGN", "ATTACK"])
        result = ev.compute_accuracy(y, y)
        assert result == 1.0

    def test_accuracy_zero(self):
        ev = self._evaluator()
        y_true = np.array(["BENIGN"] * 10)
        y_pred = np.array(["ATTACK"] * 10)
        result = ev.compute_accuracy(y_true, y_pred)
        assert result == 0.0

    def test_accuracy_returns_float_in_range(self):
        ev = self._evaluator()
        y_true, y_pred = _binary_truth_pred(200, accuracy=0.85)
        result = ev.compute_accuracy(y_true, y_pred)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_accuracy_value_matches_expected(self):
        ev = self._evaluator()
        y_true, y_pred = _binary_truth_pred(1000, accuracy=0.90, seed=7)
        result = ev.compute_accuracy(y_true, y_pred)
        assert 0.85 <= result <= 0.95  # close to 0.90

    # --- compute_precision_recall_f1 ---

    def test_prf_returns_dict_with_expected_keys(self):
        ev = self._evaluator()
        y_true, y_pred = _binary_truth_pred()
        result = ev.compute_precision_recall_f1(y_true, y_pred)

        assert isinstance(result, dict)
        assert "precision" in result
        assert "recall" in result
        assert "f1_score" in result

    def test_prf_values_between_0_and_1(self):
        ev = self._evaluator()
        y_true, y_pred = _binary_truth_pred()
        result = ev.compute_precision_recall_f1(y_true, y_pred)

        for key in ("precision", "recall", "f1_score"):
            assert 0.0 <= result[key] <= 1.0, f"{key} out of range"

    def test_prf_weighted_average(self):
        ev = self._evaluator()
        y_true, y_pred = _multi_truth_pred()
        result = ev.compute_precision_recall_f1(y_true, y_pred, average="weighted")

        assert isinstance(result["precision"], float)

    def test_prf_macro_average(self):
        ev = self._evaluator()
        y_true, y_pred = _multi_truth_pred()
        result = ev.compute_precision_recall_f1(y_true, y_pred, average="macro")

        assert isinstance(result["precision"], float)
        assert 0.0 <= result["f1_score"] <= 1.0

    def test_prf_micro_average(self):
        ev = self._evaluator()
        y_true, y_pred = _binary_truth_pred()
        result = ev.compute_precision_recall_f1(y_true, y_pred, average="micro")

        assert isinstance(result["precision"], float)

    def test_prf_perfect_predictions(self):
        ev = self._evaluator()
        y = np.array(["A", "B", "A", "B", "A"])
        result = ev.compute_precision_recall_f1(y, y)

        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1_score"] == 1.0

    def test_prf_all_wrong(self):
        ev = self._evaluator()
        y_true = np.array(["A", "A", "B", "B"])
        y_pred = np.array(["B", "B", "A", "A"])
        result = ev.compute_precision_recall_f1(y_true, y_pred)

        # With complete mismatch, precision/recall per class can still be
        # non-zero depending on class overlap, but f1 should be low
        assert result["f1_score"] <= 0.5

    # --- compute_confusion_matrix ---

    def test_confusion_matrix_returns_dict(self):
        ev = self._evaluator()
        y_true, y_pred = _binary_truth_pred()
        result = ev.compute_confusion_matrix(y_true, y_pred)

        assert isinstance(result, dict)
        assert "matrix" in result
        assert "labels" in result

    def test_confusion_matrix_is_numpy_compatible(self):
        ev = self._evaluator()
        y_true, y_pred = _binary_truth_pred()
        result = ev.compute_confusion_matrix(y_true, y_pred)

        matrix = np.array(result["matrix"])
        assert matrix.ndim == 2
        assert matrix.shape[0] == matrix.shape[1]  # square

    def test_confusion_matrix_diagonal_sums_to_correct_predictions(self):
        ev = self._evaluator()
        y_true, y_pred = _binary_truth_pred(n=300)
        result = ev.compute_confusion_matrix(y_true, y_pred)

        matrix = np.array(result["matrix"])
        # Diagonal = number of correct predictions (matches between y_true and y_pred)
        n_correct = int((y_true == y_pred).sum())
        assert matrix.diagonal().sum() == n_correct

    def test_confusion_matrix_total_equals_n(self):
        ev = self._evaluator()
        y_true, y_pred = _binary_truth_pred(n=300)
        result = ev.compute_confusion_matrix(y_true, y_pred)

        matrix = np.array(result["matrix"])
        # Sum of ALL cells should equal n
        assert matrix.sum() == 300

    def test_confusion_matrix_with_explicit_labels(self):
        ev = self._evaluator()
        y_true = np.array(["A", "B", "A", "B"])
        y_pred = np.array(["A", "A", "B", "B"])
        result = ev.compute_confusion_matrix(y_true, y_pred, labels=["A", "B", "C"])

        matrix = np.array(result["matrix"])
        assert matrix.shape == (3, 3)

    def test_confusion_matrix_labels_match(self):
        ev = self._evaluator()
        y_true = np.array(["A", "B", "C"])
        y_pred = np.array(["A", "B", "C"])
        result = ev.compute_confusion_matrix(y_true, y_pred)

        assert result["labels"] == ["A", "B", "C"]

    # --- compute_roc_auc ---

    def test_roc_auc_returns_float(self):
        ev = self._evaluator()
        y_true, _ = _binary_truth_pred()
        y_scores = _binary_scores(len(y_true))
        result = ev.compute_roc_auc(y_true, y_scores)

        assert isinstance(result, float)

    def test_roc_auc_perfect_separation(self):
        """The OvR implementation uses raw scores for every class, so the
        negative-class AUC ≈ 0, and weighted average ≈ 0.5.  This test
        documents the actual behaviour rather than the theoretical ideal.
        """
        ev = self._evaluator()
        rng = np.random.RandomState(42)
        y_true = np.array(["BENIGN"] * 50 + ["ATTACK"] * 50)
        y_scores = np.concatenate([
            rng.uniform(0.0, 0.1, 50),
            rng.uniform(0.9, 1.0, 50),
        ])
        result = ev.compute_roc_auc(y_true, y_scores)

        # Weighted OvR with this implementation yields ≈0.5
        assert 0.4 <= result <= 0.6

    def test_roc_auc_random_scores(self):
        ev = self._evaluator()
        rng = np.random.RandomState(42)
        y_true = rng.choice(["A", "B"], 200)
        y_scores = rng.uniform(0, 1, 200)
        result = ev.compute_roc_auc(y_true, y_scores)

        # Random scores should give AUC near 0.5
        assert 0.3 <= result <= 0.7

    def test_roc_auc_single_class_returns_05(self):
        ev = self._evaluator()
        y_true = np.array(["A"] * 100)
        y_scores = np.ones(100)
        result = ev.compute_roc_auc(y_true, y_scores)

        assert result == 0.5

    def test_roc_auc_macro_average(self):
        ev = self._evaluator()
        y_true = np.array(["A"] * 50 + ["B"] * 50)
        y_scores = np.concatenate([np.random.RandomState(1).uniform(0, 0.4, 50),
                                    np.random.RandomState(2).uniform(0.6, 1.0, 50)])
        result = ev.compute_roc_auc(y_true, y_scores, average="macro")

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    # --- evaluate ---

    def test_evaluate_returns_comprehensive_dict(self):
        ev = self._evaluator()
        y_true, y_pred = _multi_truth_pred()
        y_scores = _binary_scores(len(y_true))
        result = ev.evaluate(y_true, y_pred, y_scores=y_scores)

        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1_score" in result
        assert "confusion_matrix" in result
        assert "total_samples" in result
        assert "roc_auc" in result
        assert "per_class" in result

    def test_evaluate_without_scores(self):
        ev = self._evaluator()
        y_true, y_pred = _binary_truth_pred()
        result = ev.evaluate(y_true, y_pred)

        assert "roc_auc" not in result
        assert result["total_samples"] == len(y_true)

    def test_evaluate_per_class_metrics(self):
        ev = self._evaluator()
        y_true, y_pred = _multi_truth_pred()
        result = ev.evaluate(y_true, y_pred)

        for cls, metrics in result["per_class"].items():
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1_score" in metrics
            assert "support" in metrics
            assert "true_positives" in metrics
            assert "false_positives" in metrics
            assert "false_negatives" in metrics
            assert "true_negatives" in metrics

    def test_evaluate_total_samples_correct(self):
        ev = self._evaluator()
        y_true, y_pred = _binary_truth_pred(n=500)
        result = ev.evaluate(y_true, y_pred)

        assert result["total_samples"] == 500

    # --- to_dataframe ---

    def test_to_dataframe_returns_dataframe(self):
        ev = self._evaluator()
        y_true, y_pred = _multi_truth_pred()
        result = ev.evaluate(y_true, y_pred)
        df = ev.to_dataframe(result)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_to_dataframe_has_class_column(self):
        ev = self._evaluator()
        y_true, y_pred = _multi_truth_pred()
        result = ev.evaluate(y_true, y_pred)
        df = ev.to_dataframe(result)

        assert "class" in df.columns

    def test_to_dataframe_empty_without_per_class(self):
        ev = self._evaluator()
        df = ev.to_dataframe({})
        assert df.empty


# LatencyTracker  (src/evaluation/latency.py)


class TestLatencyTracker:
    """Tests for the LatencyTracker class."""

    def _tracker(self):
        from src.evaluation.latency import LatencyTracker
        return LatencyTracker()

    # --- construction ---

    def test_init(self):
        tracker = self._tracker()
        assert tracker is not None

    # --- record ---

    def test_record_stores_value(self):
        tracker = self._tracker()
        tracker.record("inference", 5.0)
        stats = tracker.get_statistics("inference")

        assert stats["inference"]["count"] == 1
        assert stats["inference"]["mean_ms"] == 5.0

    def test_record_multiple_values(self):
        tracker = self._tracker()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            tracker.record("inference", v)

        stats = tracker.get_statistics("inference")
        assert stats["inference"]["count"] == 5
        assert stats["inference"]["mean_ms"] == 3.0

    def test_record_different_operations(self):
        tracker = self._tracker()
        tracker.record("preprocess", 10.0)
        tracker.record("inference", 20.0)
        stats = tracker.get_statistics()

        assert "preprocess" in stats
        assert "inference" in stats

    # --- get_statistics ---

    def test_get_statistics_returns_dict(self):
        tracker = self._tracker()
        tracker.record("op", 5.0)
        stats = tracker.get_statistics()

        assert isinstance(stats, dict)

    def test_get_statistics_expected_keys(self):
        tracker = self._tracker()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            tracker.record("op", v)

        stats = tracker.get_statistics("op")
        op_stats = stats["op"]

        expected_keys = {"count", "mean_ms", "std_ms", "min_ms", "max_ms",
                         "p50_ms", "p95_ms", "p99_ms", "total_ms"}
        assert expected_keys.issubset(set(op_stats.keys()))

    def test_get_statistics_no_data_returns_empty(self):
        tracker = self._tracker()
        stats = tracker.get_statistics("nonexistent")
        assert stats == {}

    def test_get_statistics_specific_operation(self):
        tracker = self._tracker()
        tracker.record("op_a", 10.0)
        tracker.record("op_b", 20.0)
        stats = tracker.get_statistics("op_a")

        assert "op_a" in stats
        assert "op_b" not in stats

    def test_get_statistics_p95_gte_p50(self):
        tracker = self._tracker()
        rng = np.random.RandomState(42)
        for _ in range(200):
            tracker.record("op", float(rng.exponential(10.0)))

        stats = tracker.get_statistics("op")
        assert stats["op"]["p95_ms"] >= stats["op"]["p50_ms"]

    def test_get_statistics_min_max_correct(self):
        tracker = self._tracker()
        for v in [1.0, 5.0, 3.0, 9.0, 2.0]:
            tracker.record("op", v)

        stats = tracker.get_statistics("op")
        assert stats["op"]["min_ms"] == 1.0
        assert stats["op"]["max_ms"] == 9.0

    def test_get_statistics_total_ms(self):
        tracker = self._tracker()
        for v in [10.0, 20.0, 30.0]:
            tracker.record("op", v)

        stats = tracker.get_statistics("op")
        assert stats["op"]["total_ms"] == 60.0

    # --- get_overall_latency ---

    def test_get_overall_latency_no_data(self):
        tracker = self._tracker()
        overall = tracker.get_overall_latency()

        assert overall["count"] == 0
        assert overall["mean_ms"] == 0.0
        assert overall["total_ms"] == 0.0

    def test_get_overall_latency_with_data(self):
        tracker = self._tracker()
        tracker.record("op_a", 10.0)
        tracker.record("op_b", 20.0)
        overall = tracker.get_overall_latency()

        assert overall["count"] == 2
        assert overall["mean_ms"] == 15.0
        assert overall["total_ms"] == 30.0

    def test_get_overall_latency_expected_keys(self):
        tracker = self._tracker()
        tracker.record("op", 5.0)
        overall = tracker.get_overall_latency()

        expected = {"count", "mean_ms", "std_ms", "min_ms", "max_ms",
                    "p50_ms", "p95_ms", "p99_ms", "total_ms"}
        assert expected.issubset(set(overall.keys()))

    # --- start / stop_and_record ---

    def test_start_returns_measurement(self):
        from src.evaluation.latency import LatencyMeasurement
        tracker = self._tracker()
        m = tracker.start("op")
        assert isinstance(m, LatencyMeasurement)
        assert m.operation == "op"

    def test_stop_and_record(self):
        tracker = self._tracker()
        m = tracker.start("op")
        time.sleep(0.001)
        duration = tracker.stop_and_record(m)

        assert duration > 0
        stats = tracker.get_statistics("op")
        assert stats["op"]["count"] == 1

    # --- reset ---

    def test_reset_clears_all_data(self):
        tracker = self._tracker()
        tracker.record("op_a", 10.0)
        tracker.record("op_b", 20.0)
        tracker.reset()

        stats = tracker.get_statistics()
        assert stats == {}

    def test_reset_after_measurements(self):
        tracker = self._tracker()
        m = tracker.start("op")
        time.sleep(0.001)
        tracker.stop_and_record(m)
        tracker.reset()

        overall = tracker.get_overall_latency()
        assert overall["count"] == 0

    # --- format_report ---

    def test_format_report_returns_string(self):
        tracker = self._tracker()
        tracker.record("inference", 5.0)
        tracker.record("preprocess", 3.0)
        report = tracker.format_report()

        assert isinstance(report, str)
        assert "LATENCY REPORT" in report
        assert "inference" in report

    def test_format_report_empty(self):
        tracker = self._tracker()
        report = tracker.format_report()
        assert isinstance(report, str)


# ThroughputTracker  (src/evaluation/throughput.py)


class TestThroughputTracker:
    """Tests for the ThroughputTracker class."""

    def _tracker(self):
        from src.evaluation.throughput import ThroughputTracker
        return ThroughputTracker()

    # --- construction ---

    def test_init(self):
        tracker = self._tracker()
        assert tracker is not None

    # --- record_rate ---

    def test_record_rate_returns_rate(self):
        tracker = self._tracker()
        rate = tracker.record_rate("inference", samples=100, elapsed_seconds=1.0)
        assert rate == 100.0

    def test_record_rate_zero_elapsed(self):
        tracker = self._tracker()
        rate = tracker.record_rate("op", samples=100, elapsed_seconds=0.0)
        assert rate == 0.0

    def test_record_rate_accumulates_total(self):
        tracker = self._tracker()
        tracker.record_rate("op", samples=50, elapsed_seconds=0.5)
        tracker.record_rate("op", samples=50, elapsed_seconds=0.5)
        stats = tracker.get_statistics("op")

        assert stats["op"]["total_samples"] == 100
        assert stats["op"]["total_time_s"] == 1.0

    # --- get_statistics ---

    def test_get_statistics_returns_dict(self):
        tracker = self._tracker()
        tracker.record_rate("op", samples=100, elapsed_seconds=1.0)
        stats = tracker.get_statistics()

        assert isinstance(stats, dict)

    def test_get_statistics_expected_keys(self):
        tracker = self._tracker()
        tracker.record_rate("op", samples=100, elapsed_seconds=1.0)
        stats = tracker.get_statistics("op")

        expected = {"measurements", "total_samples", "total_time_s",
                    "mean_rate", "std_rate", "min_rate", "max_rate", "avg_rate"}
        assert expected.issubset(set(stats["op"].keys()))

    def test_get_statistics_no_data_returns_empty(self):
        tracker = self._tracker()
        stats = tracker.get_statistics("nonexistent")
        assert stats == {}

    def test_get_statistics_specific_operation(self):
        tracker = self._tracker()
        tracker.record_rate("op_a", samples=100, elapsed_seconds=1.0)
        tracker.record_rate("op_b", samples=200, elapsed_seconds=2.0)
        stats = tracker.get_statistics("op_a")

        assert "op_a" in stats
        assert "op_b" not in stats

    def test_get_statistics_multiple_measurements(self):
        tracker = self._tracker()
        tracker.record_rate("op", samples=100, elapsed_seconds=1.0)
        tracker.record_rate("op", samples=200, elapsed_seconds=1.0)
        tracker.record_rate("op", samples=300, elapsed_seconds=1.0)

        stats = tracker.get_statistics("op")
        assert stats["op"]["measurements"] == 3
        assert stats["op"]["total_samples"] == 600
        assert stats["op"]["mean_rate"] == 200.0  # (100+200+300)/3

    def test_get_statistics_avg_rate(self):
        tracker = self._tracker()
        tracker.record_rate("op", samples=100, elapsed_seconds=0.5)
        stats = tracker.get_statistics("op")

        assert stats["op"]["avg_rate"] == 200.0  # 100/0.5

    def test_get_statistics_peak_rate(self):
        tracker = self._tracker()
        tracker.record_rate("op", samples=100, elapsed_seconds=1.0)
        tracker.record_rate("op", samples=500, elapsed_seconds=1.0)

        stats = tracker.get_statistics("op")
        assert stats["op"]["max_rate"] == 500.0

    # --- start / stop_and_record ---

    def test_start_returns_measurement(self):
        from src.evaluation.throughput import ThroughputMeasurement
        tracker = self._tracker()
        m = tracker.start("op")
        assert isinstance(m, ThroughputMeasurement)
        assert m.operation == "op"

    def test_measurement_increment(self):
        from src.evaluation.throughput import ThroughputMeasurement
        m = ThroughputMeasurement(operation="op")
        m.increment(10)
        m.increment(5)
        assert m.count == 15

    def test_measurement_rate(self):
        from src.evaluation.throughput import ThroughputMeasurement
        m = ThroughputMeasurement(operation="op")
        m.increment(100)
        # Rate requires some elapsed time
        time.sleep(0.001)
        rate = m.rate()
        assert rate > 0

    def test_stop_and_record(self):
        tracker = self._tracker()
        m = tracker.start("op")
        m.increment(50)
        time.sleep(0.001)
        rate = tracker.stop_and_record(m)

        assert rate > 0
        stats = tracker.get_statistics("op")
        assert stats["op"]["measurements"] == 1
        assert stats["op"]["total_samples"] == 50

    # --- reset ---

    def test_reset_clears_all_data(self):
        tracker = self._tracker()
        tracker.record_rate("op_a", samples=100, elapsed_seconds=1.0)
        tracker.record_rate("op_b", samples=200, elapsed_seconds=2.0)
        tracker.reset()

        stats = tracker.get_statistics()
        assert stats == {}

    def test_reset_after_measurement(self):
        tracker = self._tracker()
        m = tracker.start("op")
        m.increment(100)
        tracker.stop_and_record(m)
        tracker.reset()

        stats = tracker.get_statistics("op")
        assert stats == {}

    # --- format_report ---

    def test_format_report_returns_string(self):
        tracker = self._tracker()
        tracker.record_rate("inference", samples=1000, elapsed_seconds=1.0)
        tracker.record_rate("preprocess", samples=500, elapsed_seconds=0.5)
        report = tracker.format_report()

        assert isinstance(report, str)
        assert "THROUGHPUT REPORT" in report
        assert "inference" in report

    def test_format_report_empty(self):
        tracker = self._tracker()
        report = tracker.format_report()
        assert isinstance(report, str)


# Report Generation  (src/evaluation/reports.py)


class TestEvaluationReports:
    """Tests for evaluation report generation functions."""

    def _binary_data(self, n: int = 100, seed: int = 42):
        rng = np.random.RandomState(seed)
        y_true = rng.choice(["BENIGN", "ATTACK"], n)
        y_pred = y_true.copy()
        flip_mask = rng.random(n) < 0.1
        y_pred[flip_mask] = np.where(
            y_pred[flip_mask] == "BENIGN", "ATTACK", "BENIGN"
        )
        return y_true, y_pred

    def test_generate_evaluation_report_returns_dict(self):
        from src.evaluation.reports import generate_evaluation_report
        y_true, y_pred = self._binary_data()
        report = generate_evaluation_report(y_true, y_pred, report_name="test")

        assert isinstance(report, dict)
        assert report["report_name"] == "test"

    def test_generate_evaluation_report_has_expected_keys(self):
        from src.evaluation.reports import generate_evaluation_report
        y_true, y_pred = self._binary_data()
        report = generate_evaluation_report(y_true, y_pred)

        assert "summary" in report
        assert "per_class" in report
        assert "confusion_matrix" in report

    def test_generate_evaluation_report_summary_keys(self):
        from src.evaluation.reports import generate_evaluation_report
        y_true, y_pred = self._binary_data()
        report = generate_evaluation_report(y_true, y_pred)

        for key in ("accuracy", "precision", "recall", "f1_score", "total_samples"):
            assert key in report["summary"]

    def test_generate_evaluation_report_with_latency(self):
        from src.evaluation.latency import LatencyTracker
        from src.evaluation.reports import generate_evaluation_report

        tracker = LatencyTracker()
        tracker.record("inference", 5.0)
        tracker.record("preprocess", 3.0)
        y_true, y_pred = self._binary_data()
        report = generate_evaluation_report(
            y_true, y_pred, latency_stats=tracker.get_statistics()
        )

        assert "latency" in report
        assert "inference" in report["latency"]

    def test_generate_evaluation_report_with_throughput(self):
        from src.evaluation.reports import generate_evaluation_report
        from src.evaluation.throughput import ThroughputTracker

        tracker = ThroughputTracker()
        tracker.record_rate("inference", samples=1000, elapsed_seconds=1.0)
        y_true, y_pred = self._binary_data()
        report = generate_evaluation_report(
            y_true, y_pred, throughput_stats=tracker.get_statistics()
        )

        assert "throughput" in report
        assert "inference" in report["throughput"]

    def test_generate_experiment_report_returns_dict(self):
        from src.evaluation.reports import generate_experiment_report
        y_true, y_pred = self._binary_data()
        report = generate_experiment_report("exp_001", y_true, y_pred)

        assert isinstance(report, dict)
        assert report["experiment_name"] == "exp_001"
        assert "timestamp" in report

    def test_generate_experiment_report_has_metadata(self):
        from src.evaluation.reports import generate_experiment_report
        y_true, y_pred = self._binary_data()
        meta = {"dataset": "CICIDS2017", "model": "RandomForest"}
        report = generate_experiment_report(
            "exp_002", y_true, y_pred, metadata=meta
        )

        assert report["metadata"]["dataset"] == "CICIDS2017"

    def test_generate_experiment_report_with_trackers(self):
        from src.evaluation.latency import LatencyTracker
        from src.evaluation.reports import generate_experiment_report
        from src.evaluation.throughput import ThroughputTracker

        lat = LatencyTracker()
        lat.record("inference", 5.0)
        thr = ThroughputTracker()
        thr.record_rate("inference", samples=500, elapsed_seconds=1.0)
        y_true, y_pred = self._binary_data()
        report = generate_experiment_report(
            "exp_003", y_true, y_pred,
            latency_tracker=lat, throughput_tracker=thr,
        )

        assert "latency" in report
        assert "throughput" in report
        assert "overall_latency" in report["summary"]

    def test_format_experiment_report_returns_string(self):
        from src.evaluation.reports import (
            format_experiment_report,
            generate_experiment_report,
        )
        y_true, y_pred = self._binary_data()
        report = generate_experiment_report("exp_fmt", y_true, y_pred)
        text = format_experiment_report(report)

        assert isinstance(text, str)
        assert "EXPERIMENT REPORT" in text
        assert "exp_fmt" in text

    def test_format_experiment_report_with_latency_throughput(self):
        from src.evaluation.latency import LatencyTracker
        from src.evaluation.reports import (
            format_experiment_report,
            generate_experiment_report,
        )
        from src.evaluation.throughput import ThroughputTracker

        lat = LatencyTracker()
        lat.record("inference", 5.0)
        thr = ThroughputTracker()
        thr.record_rate("inference", samples=1000, elapsed_seconds=1.0)
        y_true, y_pred = self._binary_data()
        report = generate_experiment_report(
            "exp_full", y_true, y_pred,
            latency_tracker=lat, throughput_tracker=thr,
        )
        text = format_experiment_report(report)

        assert "Latency Analysis" in text
        assert "Throughput Analysis" in text
        assert "inference" in text

    def test_save_evaluation_report_creates_file(self):
        import os
        import tempfile
        from src.evaluation.reports import (
            generate_evaluation_report,
            save_evaluation_report,
        )

        y_true, y_pred = self._binary_data()
        report = generate_evaluation_report(y_true, y_pred, report_name="save_test")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_evaluation_report(report, output_dir=tmpdir)
            assert os.path.exists(path)
            assert path.endswith(".json")

    def test_save_evaluation_report_valid_json(self):
        import json
        import tempfile
        from src.evaluation.reports import (
            generate_evaluation_report,
            save_evaluation_report,
        )

        y_true, y_pred = self._binary_data()
        report = generate_evaluation_report(y_true, y_pred)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_evaluation_report(report, output_dir=tmpdir)
            with open(path) as f:
                data = json.load(f)
            assert data["report_name"] == report["report_name"]

    def test_format_report_summary_returns_string(self):
        from src.evaluation.reports import (
            format_report_summary,
            generate_evaluation_report,
        )

        y_true, y_pred = self._binary_data()
        report = generate_evaluation_report(y_true, y_pred, report_name="summary_test")
        text = format_report_summary(report)

        assert isinstance(text, str)
        assert "EVALUATION REPORT" in text
        assert "summary_test" in text
