"""
Data models for Vritti episodic memory system.

This package contains Pydantic models for:
    - Episodes: Failure episodes with multi-perspective reflections
    - Search: Search requests and responses with ranking
    - Skills: Promoted code patterns from successful fixes
    - Gating: Action recommendation and reflection requests
    - Customer: Multi-tenant customer and API key management
    - Clustering: Episode clustering for pattern detection

All models include strict validation for security and data integrity.

Example:
    >>> from src.models import Episode, EpisodeCreate, SearchRequest
    >>> request = SearchRequest(goal="Fix deployment issue")
"""

# Clustering models
from src.models.clustering import (
    ClusterInfo,
    ClusteringStats,
    ClusterTemplate,
)

# Customer models
from src.models.customer import (
    APIKey,
    APIKeyCreate,
    Customer,
    CustomerCreate,
    CustomerStatus,
    CustomerUpdate,
    SubscriptionTier,
)

# Episode models
from src.models.episode import (
    Episode,
    EpisodeCreate,
    EpisodeType,
    ErrorClass,
    LLMPerspective,
    Reflection,
    ReflectionConsensus,
    ReflectionTier,
    UsageStats,
)

# Gating models
from src.models.gating import (
    ActionRecommendation,
    ReflectRequest,
    ReflectResponse,
)

# Search models
from src.models.search import (
    PreconditionCheckResult,
    RankingWeights,
    SearchRequest,
    SearchResponse,
    SearchResult,
)

# Skill models
from src.models.skill import Skill

__all__ = [
    # Episode models
    "Episode",
    "EpisodeCreate",
    "EpisodeType",
    "ErrorClass",
    "LLMPerspective",
    "Reflection",
    "ReflectionConsensus",
    "ReflectionTier",
    "UsageStats",
    # Search models
    "PreconditionCheckResult",
    "RankingWeights",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    # Skill models
    "Skill",
    # Gating models
    "ActionRecommendation",
    "ReflectRequest",
    "ReflectResponse",
    # Customer models
    "APIKey",
    "APIKeyCreate",
    "Customer",
    "CustomerCreate",
    "CustomerStatus",
    "CustomerUpdate",
    "SubscriptionTier",
    # Clustering models
    "ClusterInfo",
    "ClusterTemplate",
    "ClusteringStats",
]
