import pytest
from unittest.mock import AsyncMock, MagicMock
from src.gating.service import GatingService
from src.models.gating import ReflectRequest, ActionRecommendation
from src.models.search import SearchResult, SearchResponse
from src.models.episode import Episode, EpisodeCreate, Reflection
from src.models.skill import Skill

@pytest.fixture
def mock_search_pipeline():
    pipeline = MagicMock()
    pipeline.embedding_service = MagicMock()
    pipeline.embedding_service.embed_text.return_value = [0.1] * 384
    return pipeline

@pytest.fixture
def mock_kyrodb_router():
    router = AsyncMock()
    router.search_skills.return_value = []
    return router

@pytest.fixture
def gating_service(mock_search_pipeline, mock_kyrodb_router):
    return GatingService(mock_search_pipeline, mock_kyrodb_router)

@pytest.mark.asyncio
async def test_reflect_block_high_confidence(gating_service, mock_search_pipeline):
    # Setup mock search result
    episode = Episode(
        create_data=EpisodeCreate(
            goal="Deploy to prod",
            tool_chain=["kubectl"],
            actions_taken=["kubectl apply"],
            error_trace="Error: Resource quota exceeded",
            error_class="resource_error"
        ),
        episode_id=1,
        reflection=Reflection(
            root_cause="Bad config",
            resolution_strategy="Fix config",
            preconditions=[],
            environment_factors=[],
            affected_components=[],
            generalization_score=0.8,
            confidence_score=0.9
        )
    )
    
    search_result = SearchResult(
        episode=episode,
        scores={"similarity": 0.95, "precondition": 0.8},
        rank=1
    )
    
    mock_search_pipeline.search = AsyncMock(return_value=SearchResponse(
        results=[search_result],
        total_candidates=1,
        total_filtered=1,
        total_returned=1,
        search_latency_ms=10,
        collection="failures",
        query_embedding_dimension=384
    ))

    request = ReflectRequest(
        goal="Deploy to prod",
        proposed_action="kubectl apply",
        tool="kubectl"
    )

    response = await gating_service.reflect_before_action(request, "cust_1")

    assert response.recommendation == ActionRecommendation.BLOCK
    assert response.confidence >= 0.9
    assert response.suggested_action == "Fix config"

@pytest.mark.asyncio
async def test_reflect_hint_medium_confidence(gating_service, mock_search_pipeline):
    # Setup mock search result
    episode = Episode(
        create_data=EpisodeCreate(
            goal="Deploy to prod",
            tool_chain=["kubectl"],
            actions_taken=["kubectl apply"],
            error_trace="Error: Resource quota exceeded",
            error_class="resource_error"
        ),
        episode_id=1,
        reflection=Reflection(
            root_cause="Bad config",
            resolution_strategy="Fix config",
            preconditions=[],
            environment_factors=[],
            affected_components=[],
            generalization_score=0.8,
            confidence_score=0.9
        )
    )
    
    search_result = SearchResult(
        episode=episode,
        scores={"similarity": 0.75, "precondition": 0.5},
        rank=1
    )
    
    mock_search_pipeline.search = AsyncMock(return_value=SearchResponse(
        results=[search_result],
        total_candidates=1,
        total_filtered=1,
        total_returned=1,
        search_latency_ms=10,
        collection="failures",
        query_embedding_dimension=384
    ))

    request = ReflectRequest(
        goal="Deploy to prod",
        proposed_action="kubectl apply",
        tool="kubectl"
    )

    response = await gating_service.reflect_before_action(request, "cust_1")

    assert response.recommendation == ActionRecommendation.HINT
    assert response.confidence == 0.7
    assert len(response.hints) > 0

@pytest.mark.asyncio
async def test_reflect_proceed_low_confidence(gating_service, mock_search_pipeline):
    mock_search_pipeline.search = AsyncMock(return_value=SearchResponse(
        results=[],
        total_candidates=0,
        total_filtered=0,
        total_returned=0,
        search_latency_ms=10,
        collection="failures",
        query_embedding_dimension=384
    ))

    request = ReflectRequest(
        goal="Deploy to prod",
        proposed_action="kubectl apply",
        tool="kubectl"
    )

    response = await gating_service.reflect_before_action(request, "cust_1")

    assert response.recommendation == ActionRecommendation.PROCEED
    assert response.confidence == 1.0  # Proceed with confidence if no failures found


@pytest.mark.asyncio
async def test_reflect_rewrite_with_skill(gating_service, mock_search_pipeline, mock_kyrodb_router):
    """Test REWRITE recommendation when high-confidence skill is found."""
    # Setup mock skill
    skill = Skill(
        skill_id=1,
        customer_id="cust_1",
        name="safe_kubectl_deploy",
        docstring="Safely deploy to Kubernetes with pre-checks",
        code="def safe_deploy(): check_image(); kubectl_apply()",
        usage_count=15,
        success_count=14,
        failure_count=1,
        source_episodes=[101, 102, 103],
        error_class="deployment_error",
        tags=["kubernetes", "deployment"]
    )
    
    mock_kyrodb_router.search_skills.return_value = [(skill, 0.90)]
    
    # No failures, just the skill
    mock_search_pipeline.search = AsyncMock(return_value=SearchResponse(
        results=[],
        total_candidates=0,
        total_filtered=0,
        total_returned=0,
        search_latency_ms=10,
        collection="failures",
        query_embedding_dimension=384
    ))

    request = ReflectRequest(
        goal="Deploy to prod",
        proposed_action="kubectl apply -f deployment.yaml",
        tool="kubectl"
    )

    response = await gating_service.reflect_before_action(request, "cust_1")

    assert response.recommendation == ActionRecommendation.REWRITE
    assert response.confidence == 0.9
    assert "safe_kubectl_deploy" in response.rationale
    assert response.suggested_action == skill.code
    assert len(response.hints) > 0
    assert "93%" in response.hints[0]  # Success rate (14/15)


@pytest.mark.asyncio
async def test_reflect_rewrite_with_failure(gating_service, mock_search_pipeline):
    """Test REWRITE recommendation for medium-high confidence failure match."""
    episode = Episode(
        create_data=EpisodeCreate(
            goal="Deploy to prod",
            tool_chain=["kubectl"],
            actions_taken=["kubectl apply"],
            error_trace="Error: Resource quota exceeded",
            error_class="resource_error"
        ),
        episode_id=1,
        reflection=Reflection(
            root_cause="Insufficient resource quota",
            resolution_strategy="Increase resource limits in deployment.yaml before applying",
            preconditions=["Check resource quota"],
            environment_factors=[],
            affected_components=["kubernetes"],
            generalization_score=0.8,
            confidence_score=0.9
        )
    )
    
    search_result = SearchResult(
        episode=episode,
        scores={"similarity": 0.85, "precondition": 0.6},  # Medium-high
        rank=1
    )
    
    mock_search_pipeline.search = AsyncMock(return_value=SearchResponse(
        results=[search_result],
        total_candidates=1,
        total_filtered=1,
        total_returned=1,
        search_latency_ms=10,
        collection="failures",
        query_embedding_dimension=384
    ))

    request = ReflectRequest(
        goal="Deploy to prod",
        proposed_action="kubectl apply",
        tool="kubectl"
    )

    response = await gating_service.reflect_before_action(request, "cust_1")

    assert response.recommendation == ActionRecommendation.REWRITE
    assert response.confidence == 0.85
    assert "likely to fail" in response.rationale.lower()
    assert response.suggested_action == "Increase resource limits in deployment.yaml before applying"


@pytest.mark.asyncio
async def test_reflect_proceed_with_low_confidence_skill_hint(gating_service, mock_search_pipeline, mock_kyrodb_router):
    """Test that low-confidence skills appear as hints instead of being ignored."""
    skill = Skill(
        skill_id=2,
        customer_id="cust_1",
        name="docker_build_helper",
        docstring="Build Docker images with caching strategies",
        code="def build_cached(): ...",
        usage_count=10,
        success_count=9,
        failure_count=1,
        source_episodes=[201],
        error_class="build_error",
        tags=["docker"]
    )
    
    # Low confidence skill (below 0.85 threshold)
    mock_kyrodb_router.search_skills.return_value = [(skill, 0.70)]
    
    # No failures
    mock_search_pipeline.search = AsyncMock(return_value=SearchResponse(
        results=[],
        total_candidates=0,
        total_filtered=0,
        total_returned=0,
        search_latency_ms=10,
        collection="failures",
        query_embedding_dimension=384
    ))

    request = ReflectRequest(
        goal="Build Docker image",
        proposed_action="docker build -t myapp:latest .",
        tool="docker"
    )

    response = await gating_service.reflect_before_action(request, "cust_1")

    assert response.recommendation == ActionRecommendation.PROCEED
    assert response.confidence == 1.0
    assert len(response.hints) == 1
    assert "docker_build_helper" in response.hints[0]
    assert "0.70" in response.hints[0]


@pytest.mark.asyncio
async def test_reflect_error_handling_fail_open(gating_service, mock_search_pipeline):
    """Test that gating service fails open (PROCEED) when errors occur."""
    # Simulate search error
    mock_search_pipeline.search = AsyncMock(side_effect=Exception("KyroDB connection failed"))

    request = ReflectRequest(
        goal="Deploy to prod",
        proposed_action="kubectl apply",
        tool="kubectl"
    )

    response = await gating_service.reflect_before_action(request, "cust_1")

    # Should fail open with PROCEED
    assert response.recommendation == ActionRecommendation.PROCEED
    assert response.confidence == 0.0  # Low confidence due to error
    assert "error" in response.rationale.lower()
    assert len(response.matched_failures) == 0


@pytest.mark.asyncio
async def test_reflect_block_priority_over_hint(gating_service, mock_search_pipeline):
    """Test that BLOCK takes priority when both high similarity and preconditions match."""
    episode = Episode(
        create_data=EpisodeCreate(
            goal="Delete production database",
            tool_chain=["psql"],
            actions_taken=["DROP DATABASE prod"],
            error_trace="Error: Permission denied",
            error_class="permission_error"
        ),
        episode_id=1,
        reflection=Reflection(
            root_cause="Attempted destructive action without proper authorization",
            resolution_strategy="Use backup script with confirmation prompts",
            preconditions=["Admin access", "Production environment"],
            environment_factors=["production"],
            affected_components=["database"],
            generalization_score=0.9,
            confidence_score=0.95
        )
    )
    
    search_result = SearchResult(
        episode=episode,
        scores={"similarity": 0.92, "precondition": 0.85},  # High on both
        rank=1
    )
    
    mock_search_pipeline.search = AsyncMock(return_value=SearchResponse(
        results=[search_result],
        total_candidates=1,
        total_filtered=1,
        total_returned=1,
        search_latency_ms=10,
        collection="failures",
        query_embedding_dimension=384
    ))

    request = ReflectRequest(
        goal="Delete database",
        proposed_action="DROP DATABASE prod",
        tool="psql",
        current_state={"environment": "production"}
    )

    response = await gating_service.reflect_before_action(request, "cust_1")

    assert response.recommendation == ActionRecommendation.BLOCK
    assert response.confidence == 0.95
    assert "high risk" in response.rationale.lower()
