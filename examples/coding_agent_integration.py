"""
Vritti Integration Example for Coding Agents
==============================================

Shows how to integrate Vritti episodic memory with any coding agent.
This wrapper can be used with Aider, claude-code, or custom agents.

Usage:
    from vritti_agent import VrittiAgent
    
    agent = VrittiAgent(api_key="your-key")
    
    # Before risky action
    should_proceed = await agent.check_action(
        action="rm -rf /tmp/*",
        goal="Clean temporary files",
        context={"disk_usage": "95%"}
    )
    
    # After failure
    await agent.learn_from_failure(
        goal="Deploy to production",
        actions=["kubectl apply -f deployment.yaml"],
        error="ImagePullBackOff: image not found"
    )
"""

import httpx
from typing import Optional, Dict, List, Any
from enum import Enum


class ActionRecommendation(str, Enum):
    """Vritti's recommendations."""
    ALLOW = "allow"
    BLOCK = "block"
    REWRITE = "rewrite"
    HINT = "hint"


class VrittiAgent:
    """
    Wrapper for integrating Vritti with coding agents.
    
    Provides memory and gating capabilities to prevent repeated failures.
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "http://localhost:8000",
        timeout: int = 10
    ):
        """
        Initialize Vritti agent wrapper.
        
        Args:
            api_key: Your Vritti API key
            base_url: Vritti API endpoint (default: localhost:8000)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
    
    async def check_action(
        self,
        action: str,
        goal: str,
        tool: str = "shell",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Check if action is safe before executing (gating).
        
        Args:
            action: The proposed action/command
            goal: What you're trying to accomplish
            tool: Tool being used (shell, python, git, etc.)
            context: Additional context (env vars, state, etc.)
        
        Returns:
            {
                'safe': bool,  # True if ALLOW, False otherwise
                'recommendation': str,  # ALLOW/BLOCK/REWRITE/HINT
                'reason': str,  # Why this recommendation
                'alternative': str,  # Suggested alternative (if REWRITE)
                'similar_failure': dict,  # Past failure details (if found)
            }
        
        Example:
            result = await agent.check_action(
                action="rm -rf /",
                goal="Clean system",
                context={"user": "root"}
            )
            
            if result['safe']:
                os.system(action)
            else:
                print(f"BLOCKED: {result['reason']}")
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/v1/gate/reflect",
                headers=self.headers,
                json={
                    "proposed_action": action,
                    "goal": goal,
                    "tool": tool,
                    "context": str(context or {}),
                    "current_state": context or {}
                }
            )
            response.raise_for_status()
            data = response.json()
        
        # Parse response
        recommendation = data.get('recommendation', 'ALLOW')
        
        return {
            'safe': recommendation == ActionRecommendation.ALLOW,
            'recommendation': recommendation,
            'reason': data.get('reasoning', ''),
            'alternative': data.get('suggested_alternative', ''),
            'similar_failure': data.get('similar_episodes', [{}])[0] if data.get('similar_episodes') else None,
            'confidence': data.get('confidence', 0.0),
        }
    
    async def learn_from_failure(
        self,
        goal: str,
        actions: List[str],
        error: str,
        tool_chain: Optional[List[str]] = None,
        resolution: Optional[str] = None,
        severity: int = 3,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Capture a failure for future learning.
        
        Args:
            goal: What you were trying to accomplish
            actions: List of actions taken before failure
            error: Error message/trace
            tool_chain: Tools used (e.g., ["git", "docker"])
            resolution: How it was fixed (if known)
            severity: 1=critical, 5=minor
            context: Environment info, OS, versions, etc.
        
        Returns:
            {
                'episode_id': int,
                'reflection_generated': bool,
                'root_cause': str,  # If reflection generated
                'resolution_strategy': str,  # If reflection generated
            }
        
        Example:
            await agent.learn_from_failure(
                goal="Deploy app",
                actions=["kubectl apply -f deployment.yaml"],
                error="ImagePullBackOff",
                tool_chain=["kubectl"],
                resolution="Fixed image tag in deployment.yaml",
                context={"k8s_version": "1.28"}
            )
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/v1/capture",
                headers=self.headers,
                json={
                    "episode_type": "failure",
                    "goal": goal,
                    "actions_taken": actions,
                    "error_trace": error,
                    "error_class": self._classify_error(error),
                    "tool_chain": tool_chain or ["unknown"],
                    "resolution": resolution,
                    "severity": severity,
                    "environment_info": context or {},
                    "tags": self._extract_tags(goal, error)
                }
            )
            response.raise_for_status()
            data = response.json()
        
        return {
            'episode_id': data.get('episode_id'),
            'reflection_generated': data.get('reflection_generated', False),
            'root_cause': data.get('reflection', {}).get('root_cause'),
            'resolution_strategy': data.get('reflection', {}).get('resolution_strategy'),
        }
    
    async def search_similar_failures(
        self,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar past failures.
        
        Args:
            query: Description of current issue
            limit: Max results to return
            min_similarity: Minimum similarity threshold (0.0-1.0)
        
        Returns:
            List of similar episodes with solutions
        
        Example:
            results = await agent.search_similar_failures(
                "docker container won't start",
                limit=3
            )
            for episode in results:
                print(f"Solution: {episode['resolution']}")
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/v1/search",
                headers=self.headers,
                json={
                    "query": query,
                    "k": limit,
                    "similarity_threshold": min_similarity,
                    "include_reflection": True
                }
            )
            response.raise_for_status()
            data = response.json()
        
        return [
            {
                'episode_id': ep['episode_id'],
                'goal': ep['goal'],
                'error': ep['error_trace'],
                'resolution': ep.get('resolution'),
                'root_cause': ep.get('reflection', {}).get('root_cause'),
                'fix_strategy': ep.get('reflection', {}).get('resolution_strategy'),
                'similarity': ep.get('scores', {}).get('combined', 0.0),
                'success_rate': ep.get('usage_stats', {}).get('fix_success_rate', 0.0)
            }
            for ep in data.get('results', [])
        ]
    
    def _classify_error(self, error: str) -> str:
        """Simple error classification heuristic."""
        error_lower = error.lower()
        
        if any(word in error_lower for word in ['permission', 'denied', 'forbidden', 'unauthorized']):
            return "permission_error"
        elif any(word in error_lower for word in ['network', 'connection', 'timeout', 'refused']):
            return "network_error"
        elif any(word in error_lower for word in ['not found', 'missing', 'enoent']):
            return "resource_error"
        elif any(word in error_lower for word in ['invalid', 'syntax', 'type error']):
            return "validation_error"
        elif any(word in error_lower for word in ['dependency', 'import', 'module']):
            return "dependency_error"
        else:
            return "unknown"
    
    def _extract_tags(self, goal: str, error: str) -> List[str]:
        """Extract relevant tags from goal and error."""
        tags = []
        text = (goal + " " + error).lower()
        
        # Common tools/technologies
        tools = ['docker', 'kubernetes', 'git', 'python', 'npm', 'postgres', 'redis']
        for tool in tools:
            if tool in text:
                tags.append(tool)
        
        # Severity indicators
        if any(word in text for word in ['production', 'critical', 'urgent']):
            tags.append('critical')
        
        return tags[:5]  # Limit to 5 tags


# ============================================================================
# Example Usage
# ============================================================================

async def example_coding_agent_workflow():
    """
    Example of how a coding agent would use Vritti.
    """
    agent = VrittiAgent(api_key="em_live_test_key_12345")
    
    # SCENARIO 1: Check before risky action
    print("=" * 60)
    print("SCENARIO 1: Gating (check before action)")
    print("=" * 60)
    
    result = await agent.check_action(
        action="rm -rf /tmp/*",
        goal="Clean temporary files",
        tool="shell",
        context={"disk_usage": "95%", "os": "linux"}
    )
    
    if result['safe']:
        print("✅ Action is SAFE - proceeding")
        # Would execute: os.system("rm -rf /tmp/*")
    else:
        print(f"⛔ Action BLOCKED")
        print(f"   Reason: {result['reason']}")
        if result['alternative']:
            print(f"   Suggested: {result['alternative']}")
    
    # SCENARIO 2: Learn from a failure
    print("\n" + "=" * 60)
    print("SCENARIO 2: Learning from Failure")
    print("=" * 60)
    
    captured = await agent.learn_from_failure(
        goal="Deploy application to Kubernetes",
        actions=[
            "docker build -t myapp:latest .",
            "docker push myapp:latest",
            "kubectl apply -f deployment.yaml"
        ],
        error="ImagePullBackOff: image 'myapp:latest' not found in registry",
        tool_chain=["docker", "kubectl"],
        resolution="Changed image tag to 'myapp:v1.0.0' to match pushed version",
        severity=2,
        context={
            "k8s_version": "1.28",
            "docker_version": "24.0.6"
        }
    )
    
    print(f"✅ Failure captured: Episode #{captured['episode_id']}")
    if captured['reflection_generated']:
        print(f"   Root Cause: {captured['root_cause']}")
        print(f"   Fix: {captured['resolution_strategy']}")
    
    # SCENARIO 3: Search for similar issues
    print("\n" + "=" * 60)
    print("SCENARIO 3: Search for Similar Failures")
    print("=" * 60)
    
    similar = await agent.search_similar_failures(
        query="kubernetes deployment fails with ImagePullBackOff",
        limit=3
    )
    
    print(f"Found {len(similar)} similar failures:")
    for i, episode in enumerate(similar, 1):
        print(f"\n{i}. {episode['goal']}")
        print(f"   Similarity: {episode['similarity']:.2%}")
        print(f"   Fix: {episode['fix_strategy'][:100]}...")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_coding_agent_workflow())
