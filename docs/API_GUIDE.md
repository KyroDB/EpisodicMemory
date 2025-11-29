# Vritti API Guide

Quick reference for integrating AI agents with Vritti.

## Authentication

Include API key in all requests:
```http
X-API-Key: em_live_your_api_key
```

## Core Endpoints

### 1. Capture Failure
`POST /api/v1/capture`

When your agent fails, capture it:
```json
{
  "episode_type": "failure",
  "goal": "Deploy to Kubernetes",
  "tool_chain": ["kubectl"],
  "actions_taken": ["kubectl apply -f deployment.yaml"],
  "error_trace": "ImagePullBackOff: image not found",
  "error_class": "resource_error",
  "resolution": "Fixed image tag",
  "tags": ["kubernetes", "production"]
}
```

**Response**:
```json
{
  "episode_id": 12345,
  "status": "captured"
}
```

### 2. Check Before Action (Gating)
`POST /api/v1/gate/reflect`

Before risky actions, ask if it's safe:
```json
{
  "proposed_action": "kubectl apply -f deployment.yaml",
  "goal": "Deploy application",
  "tool": "kubectl",
  "context": "Deploying to production",
  "current_state": {"cluster": "prod"}
}
```

**Response**:
```json
{
  "recommendation": "BLOCK",
  "reasoning": "Similar action failed before (96% match). Image tag 'latest' not found",
  "suggested_alternative": "Use specific tag: myapp:v1.2.3",
  "confidence": 0.95
}
```

**Recommendations**:
- `ALLOW` - Safe to proceed
- `BLOCK` - Don't do it, use suggestion
- `REWRITE` - Use alternative action
- `HINT` - Warning, proceed with caution

### 3. Search Past Failures
`POST /api/v1/search`

Find similar problems you've solved:
```json
{
  "query": "Fix ImagePullBackOff error",
  "k": 5,
  "include_reflection": true
}
```

**Response**:
```json
{
  "results": [{
    "episode_id": 987,
    "goal": "Deploy to Kubernetes",
    "similarity": 0.89,
    "reflection": {
      "root_cause": "Image tag doesn't exist",
      "resolution_strategy": "Verify tag exists before applying",
      "confidence_score": 0.95
    }
  }]
}
```

## Error Classes

Use correct classification for better search:
- `configuration_error` - Wrong config values
- `permission_error` - Access denied
- `network_error` - Connection issues  
- `resource_error` - File/image not found
- `dependency_error` - Missing packages
- `timeout_error` - Took too long
- `validation_error` - Invalid input
- `unknown` - Unclassified

## Python Example

```python
import httpx

async def check_action(action, goal):
    """Check if action is safe before executing."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/gate/reflect",
            headers={"X-API-Key": "em_live_your_key"},
            json={
                "proposed_action": action,
                "goal": goal,
                "tool": "shell",
                "context": "",
                "current_state": {}
            }
        )
        result = response.json()
        
        if result['recommendation'] == 'BLOCK':
            print(f"BLOCKED: {result['reasoning']}")
            return False
        
        return True

# Usage
if await check_action("rm -rf /", "Clean files"):
    os.system("rm -rf /")  # Only execute if safe
```

## Health Check

Verify Vritti is running:
```bash
curl http://localhost:8000/health
```

## API Documentation

Interactive API docs: `http://localhost:8000/docs`
