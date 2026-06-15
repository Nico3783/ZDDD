"""Real-time detection engine.

Orchestrates anomaly detection (Isolation Forest) and classification
(Random Forest) into a unified pipeline with decision logic, zero-day
detection, severity calculation, and alert management.
"""

from src.detection_engine.alert_manager import Alert, AlertManager
from src.detection_engine.decision_logic import DetectionDecision, DetectionDecisionMaker
from src.detection_engine.engine import DetectionEngine, DetectionResult, DetectionStats
from src.detection_engine.orchestrator import DetectionOrchestrator
from src.detection_engine.response_handler import ResponseAction, ResponseHandler
from src.detection_engine.severity import SeverityCalculator, compute_severity

__all__ = [
    # Alert Manager
    "Alert",
    "AlertManager",
    # Decision Logic
    "DetectionDecision",
    "DetectionDecisionMaker",
    # Engine
    "DetectionEngine",
    "DetectionResult",
    "DetectionStats",
    # Orchestrator
    "DetectionOrchestrator",
    # Response Handler
    "ResponseAction",
    "ResponseHandler",
    # Severity
    "SeverityCalculator",
    "compute_severity",
]
