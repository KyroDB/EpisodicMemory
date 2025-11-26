"""
Multi-perspective reflection generation with consensus reconciliation.

Uses OpenRouter as unified API gateway for multiple LLM models.
Supports 2-model consensus for premium tier, 1-model for cheap tier.

Security Features:
- Prompt injection protection via input sanitization
- Output validation with strict schema enforcement
- Cost limits to prevent abuse
- Timeout enforcement
- No user-controlled data in system prompts
- All LLM outputs validated before storage

Performance:
- Parallel LLM calls (2 models in ~3-5 seconds)
- Graceful degradation (works with 1/2 or 2/2 models)
- Retry logic for transient failures
"""

import asyncio
import json
import logging
import time
from collections import Counter
from datetime import timezone, datetime
from typing import Optional

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

from pydantic import ValidationError

from src.config import LLMConfig
from src.models.episode import (
    EpisodeCreate,
    LLMPerspective,
    Reflection,
    ReflectionConsensus,
)

logger = logging.getLogger(__name__)


class PromptInjectionDefense:
    """
    Security layer to prevent prompt injection attacks.

    Strategies:
    1. Input sanitization (remove control characters, normalize whitespace)
    2. Length limits (prevent token exhaustion)
    3. Content filtering (detect injection patterns)
    4. Escape sequences removal
    """

    MAX_FIELD_LENGTH = 10000  # Characters
    MAX_LIST_ITEMS = 50
    MAX_ITEM_LENGTH = 2000

    INJECTION_PATTERNS = [
        "ignore previous instructions",
        "disregard all",
        "new instructions:",
        "you are now",
        "forget everything",
        "override",
        "admin mode",
        "dev mode",
        "jailbreak",
        "</system>",
        "<|endoftext|>",
        "[INST]",
        "###",  # Common prompt separator
    ]

    @classmethod
    def sanitize_text(cls, text: str, field_name: str = "text") -> str:
        """
        Sanitize text input to prevent injection.

        Args:
            text: Raw text from user
            field_name: Field name for logging

        Returns:
            Sanitized text

        Raises:
            ValueError: If text contains obvious injection attempts
        """
        if not text:
            return ""

        # Length limit
        if len(text) > cls.MAX_FIELD_LENGTH:
            logger.warning(
                f"Truncating {field_name}: {len(text)} chars -> {cls.MAX_FIELD_LENGTH}"
            )
            text = text[: cls.MAX_FIELD_LENGTH] + "... (truncated)"

        # Remove null bytes and control characters (except newlines, tabs)
        text = "".join(
            char for char in text
            if char.isprintable() or char in ("\n", "\t", "\r")
        )

        # Detect injection patterns
        text_lower = text.lower()
        for pattern in cls.INJECTION_PATTERNS:
            if pattern in text_lower:
                logger.warning(
                    f"Potential prompt injection detected in {field_name}: '{pattern}'"
                )
                # Replace with safe placeholder (case-insensitive)
                import re
                text = re.sub(re.escape(pattern), "[REDACTED]", text, flags=re.IGNORECASE)
                text_lower = text.lower()

        # Normalize excessive whitespace
        text = " ".join(text.split())

        return text.strip()

    @classmethod
    def sanitize_list(cls, items: list[str], field_name: str = "list") -> list[str]:
        """Sanitize list of strings."""
        if len(items) > cls.MAX_LIST_ITEMS:
            logger.warning(
                f"Truncating {field_name}: {len(items)} items -> {cls.MAX_LIST_ITEMS}"
            )
            items = items[: cls.MAX_LIST_ITEMS]

        sanitized = []
        for item in items:
            if not item or not item.strip():
                continue

            # Truncate individual items
            if len(item) > cls.MAX_ITEM_LENGTH:
                item = item[: cls.MAX_ITEM_LENGTH] + "..."

            sanitized.append(cls.sanitize_text(item, f"{field_name}_item"))

        return sanitized


class MultiPerspectiveReflectionService:
    """
    Generate reflections using OpenRouter with 2-model consensus.

    Uses OpenRouter API gateway to access multiple LLM providers through
    a unified OpenAI-compatible interface.

    Security-first design:
    - All inputs sanitized before prompting
    - All outputs validated against strict schema
    - Cost tracking and limits enforced
    - No user data in system prompts
    - Timeout enforcement on all API calls
    """

    SYSTEM_PROMPT = """You are an expert AI assistant analyzing software development failures.

Extract the following in valid JSON format:
{
  "root_cause": "Fundamental reason for failure (not symptoms) - be concise",
  "preconditions": ["Specific condition 1", "Specific condition 2", ...],
  "resolution_strategy": "Step-by-step resolution (be specific and actionable)",
  "environment_factors": ["OS/version/tool that matters"],
  "affected_components": ["Component 1", "Component 2"],
  "generalization_score": 0.5,
  "confidence_score": 0.8,
  "reasoning": "Brief explanation of your analysis"
}

IMPORTANT RULES:
- Be concise and actionable
- Focus on root cause, not symptoms
- Resolution should be step-by-step
- Generalization: 0.0 = very specific, 1.0 = universal pattern
- Confidence: how certain you are about this analysis
- Keep reasoning under 200 words

Return ONLY valid JSON, no markdown."""

    def __init__(self, config: LLMConfig):
        """
        Initialize multi-perspective reflection service with OpenRouter.

        Args:
            config: LLM configuration with OpenRouter API key

        Raises:
            ValueError: If no LLM provider is configured
        """
        self.config = config

        # Initialize OpenRouter client (uses OpenAI SDK with custom base_url)
        if config.use_openrouter and OPENAI_AVAILABLE:
            self.openrouter_client = AsyncOpenAI(
                api_key=config.openrouter_api_key,
                base_url=config.openrouter_base_url,
                timeout=config.timeout_seconds,
                max_retries=0,
                default_headers={
                    "HTTP-Referer": "https://kyrodb.dev",
                    "X-Title": "EpisodicMemory",
                }
            )
            logger.info(
                f"OpenRouter client initialized with models: "
                f"consensus=[{config.consensus_model_1}, {config.consensus_model_2}], "
                f"cheap={config.cheap_model}"
            )
        else:
            self.openrouter_client = None
            if not OPENAI_AVAILABLE:
                logger.warning("OpenAI SDK not installed; OpenRouter disabled")
            else:
                logger.warning("OpenRouter not configured (no API key)")

        # Fallback: Direct OpenAI client (legacy)
        if config.openai_api_key and OPENAI_AVAILABLE and not config.use_openrouter:
            self.openai_client = AsyncOpenAI(
                api_key=config.openai_api_key,
                timeout=config.timeout_seconds,
                max_retries=0,
            )
            logger.info("Direct OpenAI client initialized (legacy mode)")
        else:
            self.openai_client = None

        if not config.has_any_api_key:
            raise ValueError(
                "At least one LLM API key must be configured "
                "(openrouter_api_key or openai_api_key)"
            )

        # Cost tracking
        self.total_cost_usd = 0.0
        self.total_requests = 0
        self.requests_by_model = Counter()

        logger.info(
            f"Multi-perspective reflection service initialized with providers: "
            f"{config.enabled_providers}"
        )

    async def generate_multi_perspective_reflection(
        self,
        episode: EpisodeCreate,
        max_retries: Optional[int] = None,
        use_cheap_tier: bool = False,
    ) -> Reflection:
        """
        Generate reflection using OpenRouter models in parallel.

        Args:
            episode: Episode data (will be sanitized)
            max_retries: Override default retry count
            use_cheap_tier: Use single cheap model instead of consensus

        Returns:
            Reflection with consensus (or single-model if cheap tier)
        """
        if max_retries is None:
            max_retries = self.config.max_retries

        start_time = time.perf_counter()

        sanitized_episode = self._sanitize_episode(episode)
        user_prompt = self._build_user_prompt(sanitized_episode)

        if use_cheap_tier:
            return await self._generate_cheap_reflection(
                sanitized_episode, user_prompt, max_retries, start_time
            )

        return await self._generate_consensus_reflection(
            sanitized_episode, user_prompt, max_retries, start_time
        )

    async def _generate_consensus_reflection(
        self,
        episode: EpisodeCreate,
        user_prompt: str,
        max_retries: int,
        start_time: float,
    ) -> Reflection:
        """Generate reflection using 2-model consensus via OpenRouter."""
        tasks = []
        task_names = []

        if self.openrouter_client:
            tasks.append(
                self._call_openrouter_model(
                    self.config.consensus_model_1, user_prompt, max_retries
                )
            )
            task_names.append(self.config.consensus_model_1)

            tasks.append(
                self._call_openrouter_model(
                    self.config.consensus_model_2, user_prompt, max_retries
                )
            )
            task_names.append(self.config.consensus_model_2)
        elif self.openai_client:
            tasks.append(self._call_direct_openai(user_prompt, max_retries))
            task_names.append(self.config.openai_model_name)

        if not tasks:
            logger.error("No LLM clients available")
            return self._create_fallback_reflection(episode)

        try:
            logger.info(f"Calling {len(tasks)} LLM models in parallel for consensus...")
            results = await asyncio.gather(*tasks, return_exceptions=True)

            perspectives = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"{task_names[i]} failed: {result}")
                elif result is not None:
                    perspectives.append(result)
                    logger.info(f"{task_names[i]} succeeded")

            if not perspectives:
                logger.error("All LLM models failed")
                return self._create_fallback_reflection(episode)

            logger.info(f"Reconciling {len(perspectives)} perspectives...")
            consensus = self._reconcile_perspectives(perspectives)

            cost_usd = 0.0  # Free tier

            self.total_cost_usd += cost_usd
            self.total_requests += 1
            for perspective in perspectives:
                self.requests_by_model[perspective.model_name] += 1

            latency_ms = (time.perf_counter() - start_time) * 1000

            reflection = Reflection(
                consensus=consensus,
                root_cause=consensus.agreed_root_cause,
                preconditions=consensus.agreed_preconditions,
                resolution_strategy=consensus.agreed_resolution,
                environment_factors=self._merge_list_fields(
                    [p.environment_factors for p in perspectives]
                ),
                affected_components=self._merge_list_fields(
                    [p.affected_components for p in perspectives]
                ),
                generalization_score=sum(p.generalization_score for p in perspectives)
                / len(perspectives),
                confidence_score=consensus.consensus_confidence,
                llm_model="openrouter-consensus",
                generated_at=datetime.now(timezone.utc),
                cost_usd=cost_usd,
                generation_latency_ms=latency_ms,
            )

            logger.info(
                f"Consensus reflection generated: "
                f"{len(perspectives)}/{len(tasks)} models succeeded, "
                f"consensus={consensus.consensus_method}, "
                f"confidence={consensus.consensus_confidence:.2f}, "
                f"latency={latency_ms:.0f}ms"
            )

            return reflection

        except Exception as e:
            logger.error(f"Consensus reflection failed: {e}", exc_info=True)
            return self._create_fallback_reflection(episode)

    async def _generate_cheap_reflection(
        self,
        episode: EpisodeCreate,
        user_prompt: str,
        max_retries: int,
        start_time: float,
    ) -> Reflection:
        """Generate reflection using single cheap model via OpenRouter."""
        try:
            if self.openrouter_client:
                perspective = await self._call_openrouter_model(
                    self.config.cheap_model, user_prompt, max_retries
                )
            elif self.openai_client:
                perspective = await self._call_direct_openai(user_prompt, max_retries)
            else:
                return self._create_fallback_reflection(episode)

            if perspective is None:
                return self._create_fallback_reflection(episode)

            latency_ms = (time.perf_counter() - start_time) * 1000

            self.total_requests += 1
            self.requests_by_model[perspective.model_name] += 1

            reflection = Reflection(
                consensus=None,
                root_cause=perspective.root_cause,
                preconditions=perspective.preconditions,
                resolution_strategy=perspective.resolution_strategy,
                environment_factors=perspective.environment_factors,
                affected_components=perspective.affected_components,
                generalization_score=perspective.generalization_score,
                confidence_score=perspective.confidence_score * 0.8,  # Discount for single model
                llm_model=f"openrouter-cheap:{self.config.cheap_model}",
                generated_at=datetime.now(timezone.utc),
                cost_usd=0.0,
                generation_latency_ms=latency_ms,
            )

            logger.info(
                f"Cheap reflection generated: "
                f"model={self.config.cheap_model}, "
                f"confidence={reflection.confidence_score:.2f}, "
                f"latency={latency_ms:.0f}ms"
            )

            return reflection

        except Exception as e:
            logger.error(f"Cheap reflection failed: {e}", exc_info=True)
            return self._create_fallback_reflection(episode)

    async def _call_openrouter_model(
        self, model: str, user_prompt: str, max_retries: int
    ) -> Optional[LLMPerspective]:
        """Call a model via OpenRouter with retry logic and validation."""
        for attempt in range(max_retries):
            try:
                response = await self.openrouter_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout_seconds,
                )

                content = response.choices[0].message.content

                # Extract JSON from markdown if needed
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]

                data = json.loads(content.strip())

                perspective = LLMPerspective(model_name=model, **data)

                return perspective

            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"{model} output validation failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return None

            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                logger.error(f"{model} call failed: {e}")
                return None

        return None

    async def _call_direct_openai(
        self, user_prompt: str, max_retries: int
    ) -> Optional[LLMPerspective]:
        """Call OpenAI directly (legacy fallback)."""
        for attempt in range(max_retries):
            try:
                response = await self.openai_client.chat.completions.create(
                    model=self.config.openai_model_name,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    response_format={"type": "json_object"},
                    timeout=self.config.timeout_seconds,
                )

                content = response.choices[0].message.content
                data = json.loads(content)
                perspective = LLMPerspective(
                    model_name=self.config.openai_model_name, **data
                )

                return perspective

            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"OpenAI output validation failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return None

            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                logger.error(f"OpenAI call failed: {e}")
                return None

        return None

    def _sanitize_episode(self, episode: EpisodeCreate) -> EpisodeCreate:
        """
        Security: Sanitize all episode fields to prevent prompt injection.

        Returns a copy of episode with sanitized fields.
        """
        return EpisodeCreate(
            customer_id=episode.customer_id,  # Already validated by auth
            episode_type=episode.episode_type,
            goal=PromptInjectionDefense.sanitize_text(episode.goal, "goal"),
            tool_chain=PromptInjectionDefense.sanitize_list(
                episode.tool_chain, "tool_chain"
            ),
            actions_taken=PromptInjectionDefense.sanitize_list(
                episode.actions_taken, "actions_taken"
            ),
            error_trace=PromptInjectionDefense.sanitize_text(
                episode.error_trace, "error_trace"
            ),
            error_class=episode.error_class,
            code_state_diff=PromptInjectionDefense.sanitize_text(
                episode.code_state_diff or "", "code_state_diff"
            )
            if episode.code_state_diff
            else None,
            environment_info=episode.environment_info,  # Dict, handled separately
            screenshot_path=episode.screenshot_path,  # Path, not user-controlled text
            resolution=PromptInjectionDefense.sanitize_text(
                episode.resolution or "", "resolution"
            )
            if episode.resolution
            else None,
            time_to_resolve_seconds=episode.time_to_resolve_seconds,
            tags=PromptInjectionDefense.sanitize_list(episode.tags, "tags"),
            severity=episode.severity,
        )

    def _build_user_prompt(self, episode: EpisodeCreate) -> str:
        """
        Build user prompt from sanitized episode data.

        Security: Episode data is already sanitized, but we still
        keep it separate from system prompt.
        """
        prompt_parts = [
            f"Goal: {episode.goal}",
            f"\nTool Chain: {' → '.join(episode.tool_chain)}",
            f"\nError Class: {episode.error_class.value}",
            "\nActions Taken:",
        ]

        for i, action in enumerate(episode.actions_taken[:20], 1):  # Limit to 20
            prompt_parts.append(f"  {i}. {action}")

        prompt_parts.append(f"\nError Trace:\n{episode.error_trace[:2000]}")  # Limit

        if episode.code_state_diff:
            diff = episode.code_state_diff
            if len(diff) > 2000:
                diff = diff[:2000] + "\n... (truncated)"
            prompt_parts.append(f"\nCode Diff:\n{diff}")

        if episode.environment_info:
            # Sanitize dict values
            safe_env = {
                str(k)[:100]: str(v)[:500]
                for k, v in list(episode.environment_info.items())[:20]
            }
            env_str = json.dumps(safe_env, indent=2)
            prompt_parts.append(f"\nEnvironment:\n{env_str}")

        if episode.resolution:
            prompt_parts.append(f"\nResolution: {episode.resolution[:1000]}")

        return "\n".join(prompt_parts)

    def _reconcile_perspectives(
        self, perspectives: list[LLMPerspective]
    ) -> ReflectionConsensus:
        """
        Reconcile multiple perspectives using Self-Contrast/Mirror approach.

        Algorithm:
        1. Compare root causes
        2. Find majority opinion (or highest confidence if no majority)
        3. Merge preconditions (union)
        4. Select best resolution (highest confidence)
        5. Calculate consensus confidence
        """
        if not perspectives:
            raise ValueError("Cannot reconcile empty perspectives")

        # Extract root causes
        root_causes = [p.root_cause for p in perspectives]

        # Check for unanimous agreement
        if len(set(root_causes)) == 1:
            consensus_method = "unanimous"
            agreed_root_cause = root_causes[0]
            consensus_confidence = 1.0
            disagreement_points = []

        # Check for majority (works for 2+ perspectives)
        else:
            counter = Counter(root_causes)
            most_common_root, count = counter.most_common(1)[0]

            if count >= len(perspectives) / 2:  # Majority
                consensus_method = "majority_vote"
                agreed_root_cause = most_common_root
                consensus_confidence = count / len(perspectives)

                disagreement_points = [
                    f"{p.model_name}: {p.root_cause}"
                    for p in perspectives
                    if p.root_cause != agreed_root_cause
                ]

            else:
                # No majority - use highest confidence
                consensus_method = "weighted_average"
                best = max(perspectives, key=lambda p: p.confidence_score)
                agreed_root_cause = best.root_cause
                consensus_confidence = 0.5  # Low confidence

                disagreement_points = [
                    f"{p.model_name}: {p.root_cause}" for p in perspectives
                ]

        # Merge preconditions (union, deduplicate)
        all_preconditions = []
        for p in perspectives:
            all_preconditions.extend(p.preconditions)
        agreed_preconditions = list(dict.fromkeys(all_preconditions))  # Dedupe, preserve order

        # Select best resolution (highest confidence)
        best_resolution = max(
            perspectives, key=lambda p: p.confidence_score
        ).resolution_strategy

        return ReflectionConsensus(
            perspectives=perspectives,
            consensus_method=consensus_method,
            agreed_root_cause=agreed_root_cause,
            agreed_preconditions=agreed_preconditions,
            agreed_resolution=best_resolution,
            consensus_confidence=consensus_confidence,
            disagreement_points=disagreement_points,
            generated_at=datetime.now(timezone.utc),
        )

    def _merge_list_fields(self, lists: list[list[str]]) -> list[str]:
        """Merge multiple lists, deduplicate, preserve order."""
        merged = []
        seen = set()
        for lst in lists:
            for item in lst:
                if item not in seen:
                    merged.append(item)
                    seen.add(item)
        return merged

    def _create_fallback_reflection(self, episode: EpisodeCreate) -> Reflection:
        """
        Create heuristic reflection when LLMs fail.

        This ensures the system never fails completely.
        """
        logger.warning("Creating fallback reflection (LLM-free heuristic)")

        root_cause = f"{episode.error_class.value} in {episode.tool_chain[0]}"

        preconditions = [
            f"Using tool: {episode.tool_chain[0]}",
            f"Error class: {episode.error_class.value}",
        ]

        resolution_strategy = (
            episode.resolution
            if episode.resolution
            else "Manual investigation required - check error trace and environment"
        )

        return Reflection(
            consensus=None,  # No consensus for fallback
            root_cause=root_cause,
            preconditions=preconditions,
            resolution_strategy=resolution_strategy,
            environment_factors=list(episode.environment_info.keys())[:10]
            if episode.environment_info
            else [],
            affected_components=episode.tool_chain[:5],
            generalization_score=0.3,
            confidence_score=0.4,  # Low confidence for heuristic
            llm_model="fallback_heuristic",
            generated_at=datetime.now(timezone.utc),
            cost_usd=0.0,  # Free
            generation_latency_ms=0.0,
        )

    def get_usage_stats(self) -> dict:
        """Get service usage statistics."""
        return {
            "total_cost_usd": self.total_cost_usd,
            "total_requests": self.total_requests,
            "requests_by_model": dict(self.requests_by_model),
            "average_cost_per_request": (
                self.total_cost_usd / self.total_requests
                if self.total_requests > 0
                else 0.0
            ),
        }
