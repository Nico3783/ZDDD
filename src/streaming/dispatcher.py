from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DispatchResult:
    """Result of dispatching a batch to handlers."""

    batch_id: int
    handler: str
    success: bool
    result: Any = None
    error: Optional[str] = None


class BatchDispatcher:
    """Dispatches batches to multiple processing handlers.

    Routes streaming batches to different handlers based on
    configurable routing rules or round-robin distribution.
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, Callable[[pd.DataFrame], Any]] = {}
        self._routing: List[str] = []
        self._dispatch_history: List[DispatchResult] = []
        self._round_robin_idx: int = 0
        logger.info("BatchDispatcher initialized")

    def register_handler(
        self,
        name: str,
        handler: Callable[[pd.DataFrame], Any],
    ) -> None:
        """Register a processing handler.

        Args:
            name: Handler name.
            handler: Callable that processes a DataFrame batch.
        """
        self._handlers[name] = handler
        self._routing.append(name)
        logger.info("Handler registered: %s", name)

    def dispatch(self, batch: pd.DataFrame, batch_id: int) -> DispatchResult:
        """Dispatch a batch to the next handler (round-robin).

        Args:
            batch: Batch DataFrame.
            batch_id: Batch identifier.

        Returns:
            DispatchResult with outcome.
        """
        if not self._handlers:
            return DispatchResult(
                batch_id=batch_id,
                handler="none",
                success=False,
                error="No handlers registered",
            )

        handler_name = self._routing[self._round_robin_idx % len(self._routing)]
        self._round_robin_idx += 1

        return self.dispatch_to(batch, batch_id, handler_name)

    def dispatch_to(
        self,
        batch: pd.DataFrame,
        batch_id: int,
        handler_name: str,
    ) -> DispatchResult:
        """Dispatch a batch to a specific handler.

        Args:
            batch: Batch DataFrame.
            batch_id: Batch identifier.
            handler_name: Name of the handler to use.

        Returns:
            DispatchResult with outcome.
        """
        if handler_name not in self._handlers:
            result = DispatchResult(
                batch_id=batch_id,
                handler=handler_name,
                success=False,
                error=f"Handler '{handler_name}' not found",
            )
            self._dispatch_history.append(result)
            return result

        try:
            handler_output = self._handlers[handler_name](batch)
            result = DispatchResult(
                batch_id=batch_id,
                handler=handler_name,
                success=True,
                result=handler_output,
            )
        except Exception as e:
            result = DispatchResult(
                batch_id=batch_id,
                handler=handler_name,
                success=False,
                error=str(e),
            )
            logger.error("Dispatch failed for batch %d to %s: %s", batch_id, handler_name, e)

        self._dispatch_history.append(result)
        return result

    def dispatch_all(self, batch: pd.DataFrame, batch_id: int) -> List[DispatchResult]:
        """Dispatch a batch to ALL registered handlers.

        Args:
            batch: Batch DataFrame.
            batch_id: Batch identifier.

        Returns:
            List of DispatchResult, one per handler.
        """
        results: List[DispatchResult] = []
        for name in self._routing:
            results.append(self.dispatch_to(batch, batch_id, name))
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get dispatch statistics.

        Returns:
            Dictionary with dispatch counts and success rates per handler.
        """
        stats: Dict[str, Dict[str, Any]] = {}
        for result in self._dispatch_history:
            h = result.handler
            if h not in stats:
                stats[h] = {"total": 0, "success": 0, "failed": 0}
            stats[h]["total"] += 1
            if result.success:
                stats[h]["success"] += 1
            else:
                stats[h]["failed"] += 1

        # Compute success rates
        for h, s in stats.items():
            s["success_rate"] = s["success"] / s["total"] if s["total"] > 0 else 0.0

        return stats

    def clear_history(self) -> None:
        """Clear dispatch history."""
        self._dispatch_history.clear()
        logger.info("Dispatch history cleared")
