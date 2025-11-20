"""
Pre-Action Gating System.

Provides the GatingService to analyze proposed actions against past failures
and skills to recommend whether to proceed, block, or rewrite.
"""

from .service import GatingService

__all__ = ["GatingService"]
