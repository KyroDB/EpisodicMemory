# Troubleshooting Guide

## Common Issues

### 401 Unauthorized
- **Cause**: Invalid or missing `X-API-Key`.
- **Fix**: Check your API key in `.env` or headers. Ensure it matches the hashed key in the database.

### 429 Too Many Requests
- **Cause**: Rate limit exceeded (default: 100 req/min).
- **Fix**: Implement exponential backoff in your client. Contact support to increase limits.

### "No candidates found"
- **Cause**:
  - Vector search threshold too high (`min_similarity`).
  - Preconditions filtered out all candidates.
  - Empty database.
- **Fix**:
  - Lower `min_similarity` (vector similarity threshold, range 0.0–1.0, default: 0.8) to 0.6 or 0.7.
  - Check `current_state` matches episode preconditions.
  - Ensure episodes are being ingested correctly.

### Slow Search Response (>200ms)
- **Cause**:
  - Cold start (model loading).
  - Large `k` value (e.g., k=100).
  - Network latency to KyroDB.
- **Fix**:
  - Keep the service warm.
  - Use `k=5` or `k=10` (number of results to return, default: 20, acceptable range: 1-100).
  - Check KyroDB connection.

## Health Checks

Check the system status:

```bash
# Local
curl http://localhost:8000/health

# Staging
curl https://staging-api.example.com/health  # Replace with ${STAGING_API_HOST}

# Production
curl https://api.example.com/health  # Replace with ${PRODUCTION_API_HOST}
```

**Note**: `HOST` and `PORT` are configurable via environment variables or deployment settings.

**Response**:
```json
{
  "status": "healthy",
  "components": {
    "kyrodb": "connected",      // Vector database for episodes
    "database": "connected",    // Metadata database (PostgreSQL/SQLite)
    "redis": "connected"        // Cache layer
  }
}
```

**Component Statuses**:
- `connected`: Service is healthy
- `disconnected`: Service is unreachable
- `degraded`: Service is partially available

**Troubleshooting disconnected components**:
1. **Verify service is running**:
   ```bash
   docker-compose ps  # Check if services are up
   ```
2. **Check service logs**:
   ```bash
   docker-compose logs <SERVICE_NAME>  # Replace <SERVICE_NAME> with: kyrodb, postgres, redis, etc.
   # Find service names: docker-compose config --services
   ```
3. **Confirm network/.env configuration**:
   - Check connection strings in `.env`
   - Verify firewall rules
   - Test network connectivity

## Logs

Logs are structured JSON. Look for `level="ERROR"` or `level="WARNING"`.

```bash
# Tail logs (replace <SERVICE_NAME> with your actual service name, e.g., "api", "episodic-memory")
# Find service name: docker-compose ps
docker-compose logs -f <SERVICE_NAME> | grep "ERROR"
```
