# LLM Configuration Guide

This document covers the configuration and management of LLM providers used for reflection generation in Vritti.

---

## Overview

Vritti uses LLMs to generate structured reflections from failure episodes. These reflections include:
- Root cause analysis
- Preconditions for the failure
- Resolution strategies
- Generalization scores

The system supports a **tiered reflection model** that balances cost with quality:

| Tier | Provider | Model | Cost per Reflection | Use Case |
|------|----------|-------|---------------------|----------|
| Cheap | OpenRouter | Gemini Flash | ~$0.001 | Simple failures, high volume |
| Cached | Local Cache | N/A | $0.00 | Identical or highly similar episodes |
| Premium | OpenRouter | Claude Sonnet | ~$0.03 | Complex failures, consensus validation |

---

## Configuration

### Environment Variables

```bash
# Required for reflection generation
OPENROUTER_API_KEY=sk-or-v1-your-api-key

# Optional: Override default models
# Optional: Override default models
LLM_CHEAP_MODEL=google/gemini-flash-1.5            # Matches cheap tier default
LLM_CONSENSUS_MODEL_1=anthropic/claude-3.5-sonnet  # Premium tier model A
LLM_CONSENSUS_MODEL_2=openai/gpt-4o                # Premium tier model B

These environment variables intentionally mirror the documented defaults so operators can override them without editing `config.yaml`. Update the values here only if deployment needs different providers or versions.

# Optional: Adjust generation parameters
LLM_TEMPERATURE=0.7        # 0.0-1.0, default 0.7
LLM_MAX_TOKENS=1500        # Max response tokens, default 1500

# Budget controls
LLM_DAILY_WARNING_USD=10.0   # Log warning at this daily spend
LLM_DAILY_LIMIT_USD=50.0     # Block premium tier at this daily spend
```

### Configuration File

Alternatively, configure via `config.yaml`:

```yaml
llm:
  openrouter_api_key: ${OPENROUTER_API_KEY}
  cheap_model: google/gemini-flash-1.5
  consensus_model_1: anthropic/claude-3.5-sonnet
  consensus_model_2: openai/gpt-4o
  temperature: 0.7
  max_tokens: 1500
  daily_warning_usd: 10.0
  daily_limit_usd: 50.0
```

---

## Tiered Reflection System

### Tier Selection Logic

The system automatically selects the appropriate tier based on:

1. **Episode Complexity**
   - Simple errors (single tool, clear error) → Cheap tier
   - Complex failures (multi-tool, unclear cause) → Premium tier

2. **Cache Availability**
   - If a similar reflection exists (>0.92 similarity) → Cached tier
   - Cache lookup happens before any LLM call
   - **Cache Thresholds**:
     - `0.92`: Minimum similarity for cache hit (episode must match at this level to return cached reflection)
     - `0.85`: Minimum similarity for semantic clustering (used for grouping similar episodes during batch processing; individual matches must still reach 0.92 for cache hit)

3. **Routing Logic**
   - **Cheap Tier**: Default for all standard failures. Uses `google/gemini-flash-1.5`.
   - **Premium Tier**: Triggered only for:
     - `AuthenticationError`
     - `RateLimitError` (if persistent)
     - `DatabaseConnectionError`
     - Or when `episode_complexity_score > 0.7`
   - **Cached Tier**: Triggered when semantic similarity > 0.92.

3. **Budget Status**
   - If daily limit exceeded → Cheap tier only (premium blocked)
   - If approaching limit → Warning logged

4. **Explicit Override**
   - API callers can request specific tier: `tier=cheap|cached|premium`

### Cost Tracking

The system tracks daily costs and provides admin endpoints:

```bash
# Check current budget status
curl http://localhost:8000/admin/budget \
  -H "X-Admin-API-Key: your-admin-key"

# Response:
{
  "date": "2024-01-15",
  "daily_cost_usd": 12.50,
  "warning_threshold_usd": 10.0,
  "limit_threshold_usd": 50.0,
  "warning_triggered": true,
  "limit_exceeded": false,
  "budget_remaining_usd": 37.50,
  "premium_tier_blocked": false,
  "cost_by_tier": {
    "cheap": 2.50,
    "premium": 10.00
  },
  "count_by_tier": {
    "cheap": 2500,
    "cached": 1200,
    "premium": 350
  }
}
```

### Reflection Statistics

```bash
# Get detailed reflection statistics
curl http://localhost:8000/admin/reflection/stats \
  -H "X-Admin-API-Key: your-admin-key"

# Response:
{
  "total_cost_usd": 125.50,
  "total_reflections": 15000,
  "average_cost_per_reflection": 0.0084,
  "cost_savings_usd": 450.00,
  "cost_savings_percentage": 78.2,
  "cost_by_tier": {
    "cheap": 25.00,
    "cached": 0.00,
    "premium": 100.50
  },
  "percentage_by_tier": {
    "cheap": 60.0,
    "cached": 30.0,
    "premium": 10.0
  }
}
```

---

## OpenRouter Setup

### Getting an API Key

1. Sign up at [openrouter.ai](https://openrouter.ai)
2. Add credits to your account (minimum $5 recommended)
3. Generate an API key from the dashboard
4. Set `OPENROUTER_API_KEY` environment variable

### Rate Limits

OpenRouter rate limits vary by model and account tier:

| Account Type | Rate Limit | Recommendation |
|--------------|------------|----------------|
| Free | 10 req/min | Development only |
| Pay-as-you-go | 60 req/min | Small deployments |
| Enterprise | Custom | Production workloads |

The system automatically handles rate limit errors with exponential backoff.

### Supported Models

Any model available on OpenRouter can be used. Recommended:

**Cheap Tier (High Volume)**:
- `google/gemini-flash-1.5` - Fast, cheap, good quality
- `anthropic/claude-3-haiku` - Very fast, low cost
- `meta-llama/llama-3-8b-instruct` - Open source, self-hostable

**Premium Tier (Complex Failures)**:
- `anthropic/claude-3.5-sonnet` - Best quality, higher cost
- `openai/gpt-4o` - Strong reasoning
- `anthropic/claude-3-opus` - Maximum quality (expensive)

---

## Fallback Behavior

When LLM is unavailable:

1. **No API Key**: Episodes stored without reflection
2. **Rate Limited**: Retry with exponential backoff (max 3 retries)
3. **Budget Exceeded**: Cheap tier only, premium blocked
4. **API Error**: Episode stored, reflection queued for retry

### Graceful Degradation

The system continues to function without LLM:
- Episode storage works normally
- Search returns episodes (without reflection-based ranking)
- Gating uses precondition matching only

---

## Operational Guide

### Cost Optimization

1. **Enable caching**: High hit rates (30-40%) significantly reduce costs
2. **Use cheap tier first**: Route 80%+ of traffic to cheap tier
3. **Set budget limits**: Prevent runaway costs with daily limits
4. **Monitor metrics**: Track cost per customer, tier distribution

### Scaling Considerations

For high-volume deployments (>100 RPS):

1. **Reflection Workers**: Background reflection generation is handled asynchronously by `IngestionPipeline`. Scaling is primarily achieved through:
   - Horizontal scaling: Deploy multiple service instances behind a load balancer
   - Resource allocation: Increase CPU/memory for the service container
   - Note: There is no separate `reflection:` config section. All LLM configuration is under `llm:` as shown in the Configuration File section above.

2. **KyroDB Sharding**: Ensure KyroDB is provisioned with enough memory for vector index.

3. **Rate Limits**: Configure tier-based rate limits via environment variables or `config.yaml`:
   ```yaml
   # Example: Custom rate limits for Enterprise tier
   # Currently requires modifying src/rate_limits.py - future enhancement
   # will support configuration via environment variables
   ```
   For now, Enterprise customers should contact support to adjust `TIER_RATE_LIMITS` in `src/rate_limits.py`.

### Security

1. **Rotate API keys** monthly
2. **Use separate keys** for development and production
3. **Enable admin auth** (`ADMIN_API_KEY`) for budget endpoints
4. **Monitor for anomalies** in cost metrics

---

## Troubleshooting

### Common Issues

**"Reflection generation disabled"**
- Check `OPENROUTER_API_KEY` is set and valid
- Verify API key has credits

**"Premium tier blocked"**
- Daily budget exceeded
- Increase `LLM_DAILY_LIMIT_USD` or wait for next day

**"Rate limit exceeded"**
- Upgrade OpenRouter account tier
- Reduce reflection concurrency
- Enable more aggressive caching

**Slow reflection generation**
- Check OpenRouter status page
- Switch to faster model (Gemini Flash)
- Enable async reflection (don't block on LLM)

### Logs to Check

```bash
# Filter for LLM-related logs
grep -E "(reflection|llm|openrouter)" logs/app.log

# Check budget warnings
grep "budget_warning\|limit_exceeded" logs/app.log
```

---

## Metrics

Key Prometheus metrics for LLM monitoring:

```promql
# Reflection generation rate by tier
sum(rate(vritti_reflection_count_total[5m])) by (tier)

# Average cost per reflection
sum(rate(vritti_reflection_cost_usd_total[5m])) / sum(rate(vritti_reflection_count_total[5m]))

# Cache hit rate
sum(rate(vritti_reflection_count_total{tier="cached"}[5m])) / sum(rate(vritti_reflection_count_total[5m]))

# Daily cost trend
sum(increase(vritti_reflection_cost_usd_total[24h]))
```

---

## Related Documentation

- [Architecture](ARCHITECTURE.md) - System overview
- [Observability](OBSERVABILITY.md) - Metrics and logging
- [Runbooks](RUNBOOKS.md) - Incident response for LLM issues
- [API Guide](API_GUIDE.md) - Reflection API endpoints
