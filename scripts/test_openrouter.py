#!/usr/bin/env python3
"""
Quick test script to verify OpenRouter integration works with free models.

Run: python scripts/test_openrouter.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load .env file
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from config import get_settings
from models.episode import EpisodeCreate, ErrorClass, EpisodeType


async def test_openrouter_connection():
    """Test basic OpenRouter connectivity."""
    settings = get_settings()
    llm_config = settings.llm
    
    print("=== OpenRouter Configuration ===")
    print(f"API Key configured: {bool(llm_config.openrouter_api_key)}")
    print(f"Base URL: {llm_config.openrouter_base_url}")
    print(f"Consensus Model 1: {llm_config.consensus_model_1}")
    print(f"Consensus Model 2: {llm_config.consensus_model_2}")
    print(f"Cheap Model: {llm_config.cheap_model}")
    print(f"Using OpenRouter: {llm_config.use_openrouter}")
    print(f"Enabled providers: {llm_config.enabled_providers}")
    print()
    
    if not llm_config.use_openrouter:
        print("OpenRouter is not configured. Set LLM_OPENROUTER_API_KEY in .env")
        return False
    
    # Test with real API call
    from ingestion.multi_perspective_reflection import MultiPerspectiveReflectionService
    
    print("=== Testing Multi-Perspective Reflection Service ===")
    try:
        service = MultiPerspectiveReflectionService(llm_config)
        print(f"Service initialized successfully")
    except Exception as e:
        print(f"Failed to initialize service: {e}")
        return False
    
    # Create test episode
    test_episode = EpisodeCreate(
        customer_id="test-customer",
        episode_type=EpisodeType.FAILURE,
        goal="Implement user authentication endpoint",
        tool_chain=["cursor", "python", "fastapi"],
        actions_taken=[
            "Created auth.py with JWT token generation",
            "Added /login endpoint",
            "Tested with curl - got 500 error",
        ],
        error_trace="""
        Traceback (most recent call last):
          File "auth.py", line 42, in login
            token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
        NameError: name 'SECRET_KEY' is not defined
        """,
        error_class=ErrorClass.UNKNOWN,
        environment_info={"python_version": "3.11", "os": "macOS"},
        tags=["auth", "jwt"],
    )
    
    print("\n=== Testing Cheap Tier (Single Model) ===")
    try:
        reflection = await service.generate_multi_perspective_reflection(
            episode=test_episode,
            use_cheap_tier=True,
        )
        print(f"Cheap tier reflection generated!")
        print(f"  Model: {reflection.llm_model}")
        print(f"  Root cause: {reflection.root_cause[:100]}...")
        print(f"  Confidence: {reflection.confidence_score:.2f}")
        print(f"  Latency: {reflection.generation_latency_ms:.0f}ms")
        print(f"  Preconditions: {len(reflection.preconditions)}")
    except Exception as e:
        print(f"Cheap tier test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n=== Testing Consensus Tier (2 Models) ===")
    try:
        reflection = await service.generate_multi_perspective_reflection(
            episode=test_episode,
            use_cheap_tier=False,
        )
        print(f"Consensus reflection generated!")
        print(f"  Model: {reflection.llm_model}")
        print(f"  Root cause: {reflection.root_cause[:100]}...")
        print(f"  Confidence: {reflection.confidence_score:.2f}")
        print(f"  Latency: {reflection.generation_latency_ms:.0f}ms")
        if reflection.consensus:
            print(f"  Consensus method: {reflection.consensus.consensus_method}")
            print(f"  Perspectives: {len(reflection.consensus.perspectives)}")
            for p in reflection.consensus.perspectives:
                print(f"    - {p.model_name}: {p.confidence_score:.2f}")
    except Exception as e:
        print(f"Consensus tier test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n=== Usage Stats ===")
    stats = service.get_usage_stats()
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Total cost: ${stats['total_cost_usd']:.4f}")
    print(f"  Requests by model: {stats['requests_by_model']}")
    
    print("\n=== All Tests Passed ===")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_openrouter_connection())
    sys.exit(0 if success else 1)
