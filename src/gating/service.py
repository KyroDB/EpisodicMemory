"""
Gating Service for Pre-Action Validation.

Analyzes proposed actions against historical episodes to prevent repeat failures.
"""

import logging
import time
from typing import Any

from src.kyrodb.router import KyroDBRouter
from src.models.gating import ActionRecommendation, ReflectRequest, ReflectResponse
from src.models.search import SearchRequest, SearchResult
from src.models.skill import Skill
from src.observability.metrics import (
    track_gating_decision,
    track_repeat_error_prevented,
)
from src.retrieval.search import SearchPipeline

logger = logging.getLogger(__name__)


class GatingService:
    """
    Service for pre-action gating.
    
    Orchestrates search for failures and skills, then applies logic to
    determine if an action should be blocked, rewritten, or allowed.
    """

    # Confidence thresholds for gating decisions
    MIN_SIMILARITY_SEARCH = 0.6  # Minimum similarity to consider a match
    SKILL_HIGH_CONFIDENCE = 0.85  # Skill similarity to suggest using it
    BLOCK_SIMILARITY = 0.9  # Similarity threshold for blocking
    BLOCK_PRECONDITION = 0.7  # Precondition match threshold for blocking
    REWRITE_SIMILARITY = 0.8  # Similarity threshold for rewrite suggestion
    REWRITE_PRECONDITION = 0.5  # Precondition match threshold for rewrite
    HINT_SIMILARITY = 0.7  # Similarity threshold for showing hints
    
    # Display limits for hints and context
    MAX_ACTION_HINT_LENGTH = 100  # Maximum characters for action hints
    MAX_ENV_FACTORS_IN_HINTS = 3  # Maximum environment factors to show in hints

    def __init__(self, search_pipeline: SearchPipeline, kyrodb_router: KyroDBRouter):
        self.search_pipeline = search_pipeline
        self.kyrodb_router = kyrodb_router

    async def reflect_before_action(
        self, request: ReflectRequest, customer_id: str
    ) -> ReflectResponse:
        """
        Reflect before executing action.

        Args:
            request: The reflection request containing goal and proposed action.
            customer_id: The authenticated customer ID.

        Returns:
            ReflectResponse with recommendation.
        """
        start_time = time.perf_counter()
        search_latency_ms = 0.0

        try:
            # 1. Search for similar failures
            # We use the proposed action + goal as the query for better context
            search_query = f"{request.goal} {request.proposed_action}"
            
            search_req = SearchRequest(
                customer_id=customer_id,
                goal=search_query,
                current_state=request.current_state,
                k=5,  # We only need top matches for gating
                min_similarity=self.MIN_SIMILARITY_SEARCH
            )

            search_start = time.perf_counter()
            search_response = await self.search_pipeline.search(search_req)
            
            # 2. Search for relevant skills (if any)
            # We need to embed the query first - using search pipeline's embedding service
            # This assumes search_pipeline has access to embedding service
            query_embedding = self.search_pipeline.embedding_service.embed_text(search_query)
            
            matched_skills = await self.kyrodb_router.search_skills(
                query_embedding=query_embedding,
                customer_id=customer_id,
                k=3,
                min_score=0.7
            )
            
            search_latency_ms = (time.perf_counter() - search_start) * 1000

            # 3. Determine recommendation
            (
                recommendation,
                confidence,
                rationale,
                suggested_action,
                hints
            ) = self._determine_gating_recommendation(
                request.proposed_action,
                search_response.results,
                matched_skills,
                request.current_state
            )

            total_latency_ms = (time.perf_counter() - start_time) * 1000

            # Convert skills to dict for response
            skills_dicts = [
                skill.to_metadata_dict() for skill, _ in matched_skills
            ]

            # Track gating decision metrics
            track_gating_decision(
                recommendation=recommendation.value,
                customer_tier="default",  # TODO: Pass customer tier from context
                confidence=confidence,
                latency_seconds=total_latency_ms / 1000.0,
                matched_failures=len(search_response.results),
                matched_skills=len(matched_skills),
            )

            # Track repeat error prevention
            if (recommendation in [ActionRecommendation.BLOCK, ActionRecommendation.REWRITE, ActionRecommendation.HINT] and
                len(search_response.results) > 0):
                top_match = search_response.results[0]
                error_class = "unknown"
                
                # Defensive extraction of error_class with proper null checks
                try:
                    if (top_match.episode and 
                        top_match.episode.create_data and 
                        hasattr(top_match.episode.create_data, 'error_class') and
                        top_match.episode.create_data.error_class is not None):
                        
                        error_class_obj = top_match.episode.create_data.error_class
                        
                        # Check if it's an Enum with .value attribute
                        if hasattr(error_class_obj, 'value'):
                            error_class = str(error_class_obj.value)
                        else:
                            # Coerce to string if not an Enum
                            error_class = str(error_class_obj)
                except Exception as e:
                    # Log but don't fail - metrics are critical
                    logger.debug(f"Could not extract error_class from match: {e}, using 'unknown'")
                    error_class = "unknown"

                track_repeat_error_prevented(
                    customer_id=customer_id,
                    customer_tier="default",
                    error_class=error_class,
                    recommendation=recommendation.value,
                )

            return ReflectResponse(
                recommendation=recommendation,
                confidence=confidence,
                rationale=rationale,
                matched_failures=search_response.results,
                suggested_action=suggested_action,
                hints=hints,
                relevant_skills=skills_dicts,
                search_latency_ms=search_latency_ms,
                total_latency_ms=total_latency_ms
            )

        except Exception as e:
            logger.error(f"Pre-action gating failed: {e}", exc_info=True)
            # Fail open (PROCEED) but with low confidence
            return ReflectResponse(
                recommendation=ActionRecommendation.PROCEED,
                confidence=0.0,
                rationale=f"Gating service error: {str(e)}",
                matched_failures=[],
                search_latency_ms=0.0,
                total_latency_ms=(time.perf_counter() - start_time) * 1000
            )

    def _determine_gating_recommendation(
        self,
        proposed_action: str,
        matched_failures: list[SearchResult],
        matched_skills: list[tuple[Skill, float]],
        current_state: dict[str, Any]
    ) -> tuple[ActionRecommendation, float, str, str | None, list[str]]:
        """
        Determine gating recommendation based on matched failures and skills.

        Priority order:
        1. Skills (proven solutions) - suggest REWRITE if high confidence
        2. Failures - BLOCK/REWRITE/HINT based on confidence
        3. Default - PROCEED if no matches

        Args:
            proposed_action: The action the agent wants to execute
            matched_failures: Similar past failures found
            matched_skills: Relevant skills found
            current_state: Current environment state for precondition matching

        Returns:
            (recommendation, confidence, rationale, suggested_action, hints)
        """
        
        # 1. Check for high-confidence Skills first (proven solutions)
        # Skills take priority because they represent successful patterns
        if matched_skills:
            top_skill, score = matched_skills[0]
            if score >= self.SKILL_HIGH_CONFIDENCE:
                # We have a proven solution with high confidence
                # Suggest using the skill instead of the proposed action
                hints = [
                    f"Success rate: {top_skill.success_rate * 100:.0f}%",
                    f"Skill documentation: {top_skill.docstring}"
                ]
                
                # Add hint about proposed action for context
                if proposed_action:
                    hints.append(f"Original action: {proposed_action[:self.MAX_ACTION_HINT_LENGTH]}")
                
                return (
                    ActionRecommendation.REWRITE,
                    0.9,
                    f"Found proven solution: '{top_skill.name}' (used {top_skill.usage_count}× successfully)",
                    top_skill.code,  # Suggest the skill's code
                    hints
                )

        # 2. No high-confidence skills found, check failures
        if not matched_failures:
            # No failures and no skills - safe to proceed
            hints = []
            if matched_skills:
                # Low-confidence skill exists, mention as a hint
                skill, score = matched_skills[0]
                hints.append(f"Related skill available: {skill.name} (confidence: {score:.2f})")
            
            return (
                ActionRecommendation.PROCEED,
                1.0,
                "No similar past failures found.",
                None,
                hints
            )

        top_match = matched_failures[0]
        similarity_score = top_match.scores.get("similarity", 0.0)
        precondition_score = top_match.scores.get("precondition", 0.0)
        
        # Extract reflection data safely
        root_cause = "Unknown"
        resolution = None
        environment_factors = []
        if top_match.episode.reflection:
            root_cause = top_match.episode.reflection.root_cause
            resolution = top_match.episode.reflection.resolution_strategy
            environment_factors = top_match.episode.reflection.environment_factors
        
        # Check if current environment matches failure's environment factors
        environment_match = self._check_environment_match(current_state, environment_factors)

        # 3. Check for BLOCK (highest confidence failure match)
        if (similarity_score >= self.BLOCK_SIMILARITY and
            precondition_score >= self.BLOCK_PRECONDITION):
            
            rationale = (
                f"High risk: Similar action failed previously ({similarity_score:.2f} similarity, "
                f"{precondition_score:.2f} precondition match). Root cause: {root_cause}"
            )
            
            hints = [f"Previous failure: {top_match.episode.create_data.goal}"]
            
            # Add environment warning if there's a mismatch
            if not environment_match and environment_factors:
                hints.append(
                    f"Warning: Environment differs from failure (required: {', '.join(environment_factors[:self.MAX_ENV_FACTORS_IN_HINTS])})"
                )
            
            # Add proposed action context
            if proposed_action:
                hints.append(f"Blocked action: {proposed_action[:self.MAX_ACTION_HINT_LENGTH]}")
            
            return (
                ActionRecommendation.BLOCK,
                0.95,
                rationale,
                resolution,  # Suggest the fix from the failure
                hints
            )

        # 4. Check for REWRITE (medium-high confidence)
        elif (similarity_score >= self.REWRITE_SIMILARITY and
              precondition_score >= self.REWRITE_PRECONDITION and
              resolution is not None):
            
            rationale = (
                f"Action likely to fail ({similarity_score:.2f} similarity). "
                f"Suggested alternative available. Root cause: {root_cause}"
            )
            
            hints = [f"Based on failure: {top_match.episode.create_data.goal}"]
            
            # Add environment context if available
            if environment_match and environment_factors:
                hints.append(f"Environment matches failure conditions: {', '.join(environment_factors[:self.MAX_ENV_FACTORS_IN_HINTS])}")
            
            return (
                ActionRecommendation.REWRITE,
                0.85,
                rationale,
                resolution,
                hints
            )

        # 5. Check for HINT (medium confidence)
        elif similarity_score >= self.HINT_SIMILARITY:
            hints = [
                f"Watch out for: {root_cause}",
                f"Previous resolution: {resolution if resolution else 'N/A'}"
            ]
            
            return (
                ActionRecommendation.HINT,
                0.7,
                "Similar failures detected, proceed with caution.",
                None,
                hints
            )

        # 6. PROCEED (low confidence / weak match)
        else:
            return (
                ActionRecommendation.PROCEED,
                0.8,
                "Low similarity to past failures.",
                None,
                []
            )

    def _check_environment_match(
        self, current_state: dict[str, Any], environment_factors: list[str]
    ) -> bool:
        """
        Check if current environment state matches the failure's environment factors.

        Args:
            current_state: Current environment state (OS, versions, tools, etc.)
            environment_factors: List of environment factors from the failure

        Returns:
            True if there's a reasonable match, False otherwise
        """
        if not current_state or not environment_factors:
            # If either is missing, we can't make a determination
            # Return True (assume match) to be conservative
            return True

        # Convert current_state values to lowercase strings for comparison
        current_state_str = " ".join(
            str(v).lower() for v in current_state.values() if v is not None
        )

        # Check if any environment factor appears in current state
        matches = 0
        for factor in environment_factors:
            if factor.lower() in current_state_str:
                matches += 1

        # Consider it a match if at least one factor is found
        # or if there are no specific factors to check
        return matches > 0 or len(environment_factors) == 0
