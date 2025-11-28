# Vritti

**Production-grade episodic memory for AI coding assistants**

Store and retrieve multi-modal failure episodes with semantic search, precondition matching, LLM-powered reflections, and intelligent pre-action gating.

---

## What It Does

Captures failed actions (code errors, deployment failures), generates LLM reflections, and provides pre-action gating to prevent agents from repeating mistakes.

**Example Flow**:
1. AI assistant fails to deploy → Episode captured with error trace
2. LLM generates reflection (root cause, resolution strategy)
3. Next time: Agent calls `/reflect` before similar action
4. System returns BLOCK/REWRITE/HINT based on past failures
5. Agent avoids the mistake or applies proven solution

---

## Core Features

- **Multi-modal storage**: Text, code, images, error traces
- **Semantic search**: Vector similarity + precondition matching (<50ms P99)
- **LLM reflections**: Tiered cost model (cheap/cached/premium)
- **Pre-action gating**: BLOCK, REWRITE, HINT, PROCEED recommendations
- **Skills promotion**: Successful fixes promoted to reusable skills
- **Multi-tenant**: API key authentication, namespace isolation
- **Production-ready**: Prometheus metrics, structured logging, Kubernetes health probes

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Configure
cp .env.example .env
# Set KyroDB connection and OpenRouter API key in .env

# Run
uvicorn src.main:app --port 8000
```

**API**: `http://localhost:8000/docs`

---

## Usage

### Capture Failure

```python
POST /api/v1/capture
Headers: {"X-API-Key": "em_live_your_key"}
Body: {
  "episode_type": "failure",
  "goal": "Deploy to production",
  "error_trace": "ImagePullBackOff: registry.io/app:v1.2.3",
  "error_class": "resource_error",
  "tool_chain": ["kubectl", "apply"],
  "resolution": "Changed image tag to match pushed version"
}
```

### Pre-Action Gating (Reflect)

```python
POST /api/v1/reflect
Headers: {"X-API-Key": "em_live_your_key"}
Body: {
  "goal": "Deploy with kubectl apply",
  "planned_action": "kubectl apply -f deployment.yaml",
  "environment": {"cluster": "production"}
}

# Response:
{
  "recommendation": "REWRITE",
  "confidence": 0.87,
  "reason": "Similar deployment failed with ImagePullBackOff",
  "suggested_action": "Verify image tag matches registry before applying",
  "supporting_episodes": [...]
}
```

### Search Similar Episodes

```python
POST /api/v1/search
Headers: {"X-API-Key": "em_live_your_key"}
Body: {
  "goal": "Deploy with kubectl",
  "current_state": {"cluster": "prod"},
  "k": 5,
  "min_similarity": 0.6
}
```

---

## Architecture

```
┌─────────────────┐
│    FastAPI      │ ← API Layer (auth, rate limiting, validation)
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│Ingestion│ │Gating  │ ← Pre-action decision engine
└────┬───┘ └────┬───┘
     │          │
     ▼          ▼
┌─────────────────┐
│Search Pipeline  │ ← Semantic + precondition matching
└────────┬────────┘
         │
┌────────▼────────┐
│     KyroDB      │ ← Vector database (384-dim embeddings)
└─────────────────┘
```

**Storage**: KyroDB (fast vector search, HNSW index)
**Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`, 384-dim)
**LLM**: OpenRouter (Grok 4.1 fast, Deepseek R1t2, kat-coder-pro)
**Observability**: Prometheus + Grafana

---

## LLM Reflection System

Vritti uses a **tiered reflection model** to balance cost and quality:

| Tier | Cost | Use Case |
|------|------|----------|
| Cheap | ~$0.001 | Simple failures, high volume |
| Cached | $0.00 | Similar episodes (>92% match) |
| Premium | ~$0.03 | Complex failures, consensus |

```bash
# Check daily LLM budget
curl http://localhost:8000/admin/budget \
  -H "X-Admin-API-Key: your-admin-key"
```

See [LLM Configuration](docs/LLM_CONFIGURATION.md) for details.

---

## Skills System

High-performing fixes are automatically promoted to **skills**:

1. Episode validated 3+ times with 90%+ success rate
2. LLM generates reusable skill (name, docstring, preconditions)
3. Skills surfaced in gating responses for quick application

```python
# Validate a fix worked
POST /api/v1/validate_fix
{
  "episode_id": 12345,
  "outcome": "success",
  "applied_suggestion": true
}

# If promoted, response includes:
{
  "promoted_to_skill": true,
  "skill_id": 456
}
```

---

## Deployment

### Docker Compose

```bash
docker-compose up -d
```

Includes: API, KyroDB (text + image), Prometheus, Grafana

### Kubernetes

```bash
kubectl apply -k k8s/production/
```

Includes: Auto-scaling, health probes, PodDisruptionBudgets, ServiceMonitor

---

## Development

```bash
# Run tests
pytest tests/ -v --cov=src

# Run integration tests (requires KyroDB)
pytest tests/integration/ -v

# Format & lint
black src/ tests/
ruff check src/ --fix

# Type check
mypy src/
```

---

## Performance

- **Search latency**: <50ms P99 (10K episodes)
- **Gating latency**: <100ms P99 (includes LLM cache check)
- **Throughput**: 1000 req/sec (single instance)
- **Storage**: ~500KB per episode (with reflection)

---

## Rate Limits

Tier-based rate limiting per customer:

| Tier | Capture | Search | Reflect |
|------|---------|--------|---------|
| Free | 10/min | 20/min | 10/min |
| Starter | 100/min | 200/min | 100/min |
| Pro | 500/min | 1000/min | 500/min |
| Enterprise | 2000/min | 5000/min | 2000/min |

---

## Documentation

- **[Architecture](docs/ARCHITECTURE.md)**: System design, data flow
- **[LLM Configuration](docs/LLM_CONFIGURATION.md)**: OpenRouter setup, cost management
- **[Deployment](docs/DEPLOYMENT.md)**: Production setup, Kubernetes
- **[Observability](docs/OBSERVABILITY.md)**: Metrics, logging, alerting
- **[Runbooks](docs/RUNBOOKS.md)**: Incident response procedures
- **[API Docs](http://localhost:8000/docs)**: Interactive OpenAPI

---

## License

Business Source License 1.1 - Free for non-production use
Commercial use requires license - Contact: [kishan@kyrodb.com]

---

**Stack**: Python 3.11+ • FastAPI • KyroDB • OpenRouter • Docker • Kubernetes
