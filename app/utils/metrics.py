# Copyright 2025 John Brosnihan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Thread-safe in-memory metrics collector."""

import logging
import threading

from app.utils.logging_helper import log_info

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Thread-safe in-memory metrics collector for operational counters.

    Tracks job lifecycle metrics and LLM error counts using thread-safe
    atomic operations protected by a lock. Emits structured logs when
    counter values change.

    Attributes:
        _lock: Threading lock for atomic counter updates
        _counters: Dictionary of counter names to current values
    """

    def __init__(self) -> None:
        """Initialize metrics collector with zero counters."""
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {
            "jobs_queued": 0,
            "jobs_pending": 0,
            "jobs_running": 0,
            "jobs_success": 0,
            "jobs_failed": 0,
            "llm_errors": 0,
        }

    def increment(self, counter_name: str, delta: int = 1) -> None:
        """Increment a counter by the specified amount.

        Thread-safe operation that updates the counter and emits a log
        entry with the new value.

        Args:
            counter_name: Name of counter to increment
            delta: Amount to increment by (default: 1)

        Raises:
            ValueError: If counter_name is unknown
        """
        with self._lock:
            if counter_name not in self._counters:
                raise ValueError(f"Unknown counter: {counter_name}")

            self._counters[counter_name] += delta
            new_value = self._counters[counter_name]

        # Log outside lock to avoid holding it during I/O
        log_info(logger, "metric_updated", counter=counter_name, value=new_value, delta=delta)

    def decrement(self, counter_name: str, delta: int = 1) -> None:
        """Decrement a counter by the specified amount.

        Thread-safe operation that updates the counter and emits a log
        entry with the new value. Counters will not go below zero.

        Args:
            counter_name: Name of counter to decrement
            delta: Amount to decrement by (default: 1)

        Raises:
            ValueError: If counter_name is unknown
        """
        with self._lock:
            if counter_name not in self._counters:
                raise ValueError(f"Unknown counter: {counter_name}")

            self._counters[counter_name] = max(0, self._counters[counter_name] - delta)
            new_value = self._counters[counter_name]

        # Log outside lock to avoid holding it during I/O
        log_info(logger, "metric_updated", counter=counter_name, value=new_value, delta=-delta)

    def get_all(self) -> dict[str, int]:
        """Get a snapshot of all counter values.

        Thread-safe operation that returns a copy of the current state.

        Returns:
            Dictionary mapping counter names to current values
        """
        with self._lock:
            return self._counters.copy()

    def reset(self) -> None:
        """Reset all counters to zero.

        Thread-safe operation. Primarily useful for testing.
        """
        with self._lock:
            for key in self._counters:
                self._counters[key] = 0

        log_info(logger, "metrics_reset")


# Global singleton instance
_metrics = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance.

    Returns:
        Global MetricsCollector singleton
    """
    return _metrics
