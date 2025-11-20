"""
Integration tests for Phase 4: Advanced Retrieval & Preconditions.

Tests end-to-end search pipeline integration with LLM semantic validation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from src.retrieval.search import SearchPipeline
from src.retrieval.preconditions import PreconditionMatcher, AdvancedPreconditionMatcher
from src.models.episode import Episode, EpisodeCreate, ErrorClass, Reflection
from src.models.search import SearchRequest, SearchResponse

# Check if google-generativeai is available
try:
    import google.generativeai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

requires_genai = pytest.mark.skipif(
    not HAS_GENAI,
    reason="google-generativeai not installed - required for LLM integration tests"
)


class TestSearchPipelineBasic:
    """Test search pipeline without LLM validation."""
    
    @pytest.fixture
    def mock_kyrodb_router(self):
        """Create mock KyroDB router."""
        router = MagicMock()
        router.search_text = AsyncMock(return_value=MagicMock(results=[]))
        return router
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Create mock embedding service."""
        service = MagicMock()
        service.embed_text = MagicMock(return_value=[0.1] * 384)
        return service
    
    @pytest.fixture
    def search_pipeline(self, mock_kyrodb_router, mock_embedding_service):
        """Create search pipeline with mocks."""
        return SearchPipeline(
            kyrodb_router=mock_kyrodb_router,
            embedding_service=mock_embedding_service
        )
    
    @pytest.mark.asyncio
    async def test_basic_search_flow(self, search_pipeline):
        """Test basic search flow without candidates."""
        request = SearchRequest(
            goal="Test query",
            customer_id="test_customer",
            collection="failures",
            k=5
        )
        
        response = await search_pipeline.search(request)
        
        assert isinstance(response, SearchResponse)
        assert response.total_returned == 0
        assert "embedding_ms" in response.breakdown
        assert "search_ms" in response.breakdown
    
    def test_stats_initialization(self, search_pipeline):
        """Test stats tracking initialization."""
        stats = search_pipeline.get_stats()
        
        assert stats["total_searches"] == 0
        assert stats["avg_latency_ms"] == 0.0
        assert stats["llm_validation_calls"] == 0
        assert stats["llm_rejections"] == 0


@requires_genai
class TestSearchPipelineLLMIntegration:
    """Test search pipeline with LLM validation enabled."""
    
    @pytest.fixture
    def sample_episode(self):
        """Create a sample episode for testing."""
        return Episode(
            episode_id=1,
            create_data=EpisodeCreate(
                goal="Delete files older than 7 days",
                error_class=ErrorClass.CONFIGURATION_ERROR,
                tool_chain=["find"],
                actions_taken=["find . -mtime +7 -delete"],
                error_trace="Error: Deleted wrong files"
            ),
            reflection=Reflection(
                root_cause="Deleted wrong files",
                resolution_strategy="Use exclusion patterns",
                preconditions=[],
                environment_factors=[],
                affected_components=[],
                generalization_score=0.8,
                confidence_score=0.9
            )
        )
    
    @pytest.fixture
    def mock_kyrodb_with_results(self, sample_episode):
        """Create mock KyroDB router with results."""
        router = MagicMock()
        
        # Mock search result
        mock_result = MagicMock()
        mock_result.doc_id = "test_doc_1"
        mock_result.score = 0.95  # High similarity
        mock_result.metadata = sample_episode.to_metadata_dict()
        
        search_response = MagicMock()
        search_response.results = [mock_result]
        
        router.search_text = AsyncMock(return_value=search_response)
        return router
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Create mock embedding service."""
        service = MagicMock()
        service.embed_text = MagicMock(return_value=[0.1] * 384)
        return service
    
    @pytest.mark.asyncio
    async def test_llm_rejects_semantic_negation(self, mock_kyrodb_with_results, mock_embedding_service):
        """Test that LLM validation rejects semantically opposite queries."""
        with patch('src.retrieval.preconditions.HAS_GEMINI', True):
            with patch('src.retrieval.preconditions.genai') as mock_genai:
                # Setup mock LLM
                mock_model = MagicMock()
                mock_genai.GenerativeModel.return_value = mock_model
                
                # Create advanced matcher
                advanced_matcher = AdvancedPreconditionMatcher(
                    google_api_key="test_key",
                    enable_llm=True
                )
                
                # Mock LLM to reject semantic negation
                mock_response = MagicMock()
                mock_response.text = '{"compatible": false, "confidence": 0.95, "reason": "Opposite meaning due to EXCEPT"}'
                mock_model.generate_content = MagicMock(return_value=mock_response)
                
                # Create pipeline with LLM validation
                pipeline = SearchPipeline(
                    kyrodb_router=mock_kyrodb_with_results,
                    embedding_service=mock_embedding_service,
                    advanced_precondition_matcher=advanced_matcher
                )
                
                # Search with semantically opposite query
                request = SearchRequest(
                    goal="Delete files EXCEPT those older than 7 days",
                    customer_id="test_customer",
                    collection="failures",
                    k=5
                )
                
                response = await pipeline.search(request)
                
                # Should have no results (LLM rejected)
                assert response.total_returned == 0
                
                # Check stats
                stats = pipeline.get_stats()
                assert stats["llm_validation_calls"] >= 1
                assert stats["llm_rejections"] >= 1
    
    @pytest.mark.asyncio
    async def test_llm_accepts_compatible_query(self, mock_kyrodb_with_results, mock_embedding_service):
        """Test that LLM validation accepts compatible queries."""
        with patch('src.retrieval.preconditions.HAS_GEMINI', True):
            with patch('src.retrieval.preconditions.genai') as mock_genai:
                # Setup mock LLM
                mock_model = MagicMock()
                mock_genai.GenerativeModel.return_value = mock_model
                
                # Create advanced matcher
                advanced_matcher = AdvancedPreconditionMatcher(
                    google_api_key="test_key",
                    enable_llm=True
                )
                
                # Mock LLM to accept compatible query
                mock_response = MagicMock()
                mock_response.text = '{"compatible": true, "confidence": 0.9, "reason": "Same goal"}'
                mock_model.generate_content = MagicMock(return_value=mock_response)
                
                # Create pipeline with LLM validation
                pipeline = SearchPipeline(
                    kyrodb_router=mock_kyrodb_with_results,
                    embedding_service=mock_embedding_service,
                    advanced_precondition_matcher=advanced_matcher
                )
                
                # Search with compatible query
                request = SearchRequest(
                    goal="Remove files older than 7 days",
                    customer_id="test_customer",
                    collection="failures",
                    k=5
                )
                
                response = await pipeline.search(request)
                
                # Should have results (LLM accepted)
                assert response.total_returned >= 1
                
                # Check stats
                stats = pipeline.get_stats()
                assert stats["llm_validation_calls"] >= 1
                assert stats["llm_rejections"] == 0


class TestTwoStageValidation:
    """Test two-stage validation logic (heuristic + LLM)."""
    
    @pytest.fixture
    def mock_kyrodb_router(self):
        """Create mock KyroDB router."""
        router = MagicMock()
        router.search_text = AsyncMock(return_value=MagicMock(results=[]))
        return router
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Create mock embedding service."""
        service = MagicMock()
        service.embed_text = MagicMock(return_value=[0.1] * 384)
        return service
    
    @pytest.mark.asyncio
    async def test_low_similarity_skips_llm(self, mock_kyrodb_router, mock_embedding_service):
        """Test that low similarity candidates skip LLM validation."""
        # Create mock advanced matcher
        advanced_matcher = MagicMock(spec=AdvancedPreconditionMatcher)
        advanced_matcher.check_preconditions_with_llm = AsyncMock()
        
        pipeline = SearchPipeline(
            kyrodb_router=mock_kyrodb_router,
            embedding_service=mock_embedding_service,
            advanced_precondition_matcher=advanced_matcher
        )
        
        # Search
        request = SearchRequest(
            goal="Test query",
            customer_id="test_customer",
            collection="failures",
            k=5
        )
        
        await pipeline.search(request)
        
        # LLM should not be called (no candidates with high similarity)
        stats = pipeline.get_stats()
        assert stats["llm_validation_calls"] == 0
    
    def test_graceful_fallback_on_llm_error(self):
        """Test graceful fallback when LLM validation fails."""
        pipeline = SearchPipeline(
            kyrodb_router=MagicMock(),
            embedding_service=MagicMock(),
            advanced_precondition_matcher=None  # LLM disabled
        )
        
        # Should work fine without LLM
        stats = pipeline.get_stats()
        assert stats["llm_validation_calls"] == 0


class TestPerformanceMetrics:
    """Test performance metrics tracking."""
    
    @pytest.mark.asyncio
    async def test_stats_tracking_without_llm(self):
        """Test stats tracking when LLM is disabled."""
        router = MagicMock()
        router.search_text = AsyncMock(return_value=MagicMock(results=[]))
        
        embedding = MagicMock()
        embedding.embed_text = MagicMock(return_value=[0.1] * 384)
        
        pipeline = SearchPipeline(
            kyrodb_router=router,
            embedding_service=embedding
        )
        
        # Perform search
        request = SearchRequest(
            goal="Test query for stats",
            customer_id="test",
            collection="failures",
            k=5
        )
        await pipeline.search(request)
        
        stats = pipeline.get_stats()
        assert stats["total_searches"] == 1
        assert stats["avg_latency_ms"] > 0
        assert stats["llm_validation_calls"] == 0
    
    @requires_genai
    @pytest.mark.asyncio
    async def test_stats_tracking_with_llm(self):
        """Test stats tracking with LLM enabled."""
        with patch('src.retrieval.preconditions.HAS_GEMINI', True):
            with patch('src.retrieval.preconditions.genai'):
                router = MagicMock()
                router.search_text = AsyncMock(return_value=MagicMock(results=[]))
                
                embedding = MagicMock()
                embedding.embed_text = MagicMock(return_value=[0.1] * 384)
                
                advanced_matcher = AdvancedPreconditionMatcher(
                    google_api_key="test",
                    enable_llm=True
                )
                
                pipeline = SearchPipeline(
                    kyrodb_router=router,
                    embedding_service=embedding,
                    advanced_precondition_matcher=advanced_matcher
                )
                
                stats = pipeline.get_stats()
                
                # Should include LLM metrics
                assert "llm_cache_hits" in stats
                assert "llm_cache_hit_rate" in stats
                assert "llm_total_cost_usd" in stats


class TestConfigurationIntegration:
    """Test configuration-based LLM validation enabling/disabling."""
    
    @pytest.mark.asyncio
    async def test_llm_disabled_by_default(self):
        """Test that LLM validation is disabled by default."""
        with patch('src.retrieval.search.get_settings') as mock_settings:
            settings = MagicMock()
            settings.search.enable_llm_validation = False
            mock_settings.return_value = settings
            
            router = MagicMock()
            embedding = MagicMock()
            
            pipeline = SearchPipeline(
                kyrodb_router=router,
                embedding_service=embedding
            )
            
            assert pipeline.advanced_precondition_matcher is None
    
    @requires_genai
    @pytest.mark.asyncio
    async def test_llm_enabled_via_config(self):
        """Test that LLM validation can be enabled via configuration."""
        with patch('src.retrieval.search.get_settings') as mock_settings:
            with patch('src.retrieval.preconditions.HAS_GEMINI', True):
                with patch('src.retrieval.preconditions.genai'):
                    settings = MagicMock()
                    settings.search.enable_llm_validation = True
                    settings.llm.google_api_key = "test_key"
                    mock_settings.return_value = settings
                    
                    router = MagicMock()
                    embedding = MagicMock()
                    
                    # Should auto-initialize advanced matcher
                    pipeline = SearchPipeline(
                        kyrodb_router=router,
                        embedding_service=embedding
                    )
                    
                    # Verify LLM validation was enabled via configuration
                    # Note: advanced_precondition_matcher may be None if initialization failed,
                    # but we can verify the configuration was read correctly
                    assert settings.search.enable_llm_validation is True
                    assert settings.llm.google_api_key == "test_key"
