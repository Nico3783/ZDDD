class ConfigError(Exception):
    """Raised when configuration loading or validation fails."""


class DataError(Exception):
    """Raised when data loading or processing fails."""


class ModelError(Exception):
    """Raised when model training, loading, or prediction fails."""


class StreamingError(Exception):
    """Raised when streaming pipeline encounters a failure."""


class AlertError(Exception):
    """Raised when alert generation or delivery fails."""


class DetectionError(Exception):
    """Raised when the real-time detection pipeline encounters an error."""


class EvaluationError(Exception):
    """Raised when evaluation computation fails."""


class DashboardError(Exception):
    """Raised when dashboard rendering or data fetch fails."""
