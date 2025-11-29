"""
Resilience patterns for external dependencies.

Circuit breakers prevent cascade failures when dependencies fail.
"""

from .circuit_breakers import (
    KyroDBCircuitBreakerError,
    get_kyrodb_breaker,
    reset_all_breakers,
    with_kyrodb_circuit_breaker,
    with_retry,
)

__all__ = [
    "KyroDBCircuitBreakerError",
    "get_kyrodb_breaker",
    "reset_all_breakers",
    "with_kyrodb_circuit_breaker",
    "with_retry",
]
