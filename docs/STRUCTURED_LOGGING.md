# Structured Logging

JSON-formatted logs for easy parsing and analysis.

## Configuration

Set in `.env`:
```bash
LOGGING_JSON_OUTPUT=true  # JSON format
LOGGING_LEVEL=INFO        # DEBUG/INFO/WARNING/ERROR
LOGGING_COLOR IZED=false   # Color output (development only)
```

## Log Format

Each log entry is JSON:
```json
{
  "timestamp": "2025-11-29T02:44:40.633Z",
  "level": "info",
  "service": "vritti",
  "version": "0.1.0",
  "request_id": "abc123",
  "customer_id": "customer-456",
  "endpoint": "/api/v1/capture",
  "method": "POST",
  "status_code": 201,
  "duration_ms": 85.3,
  "message": "Episode captured successfully"
}
```

## Viewing Logs

### Tail logs
```bash
tail -f logs/app.log
```

### Pretty print with jq
```bash
tail -f logs/app.log | jq
```

### Filter errors only
```bash
grep '"level":"error"' logs/app.log | jq
```

### Filter by customer
```bash
grep '"customer_id":"customer-123"' logs/app.log | jq
```

### Filter slow requests (>100ms)
```bash
jq 'select(.duration_ms > 100)' logs/app.log
```

## Log Levels

- **DEBUG**: Detailed diagnostic info
- **INFO**: General operational events
- **WARNING**: Minor issues, degraded performance
- **ERROR**: Errors that need attention

## Common Log Patterns

### Successful Request
```json
{
  "level": "info",
  "endpoint": "/api/v1/capture",
  "status_code": 201,
  "duration_ms": 82.5,
  "message": "Episode captured"
}
```

### Error
```json
{
  "level": "error",
  "endpoint": "/api/v1/search",
  "error_type": "kyrodb_error",
  "error": "Connection refused",
  "stack_trace": "...",
  "message": "Search failed"
}
```

### Slow Query Warning
```json
{
  "level": "warning",
  "endpoint": "/api/v1/search",
  "duration_ms": 250,
  "message": "Slow query detected"
}
```

## PII Redaction

Sensitive data is automatically redacted:
- API keys → `***KEY`
- Email addresses → `***@example.com`
- IP addresses → `***.***.***.***`
- Credit cards → `****-****-****-1234`

## Log Rotation

Logs rotate daily:
```
logs/
├── app.log (current)
├── app.log.2025-11-28
└── app.log.2025-11-27
```

Configure in `.env`:
```bash
LOGGING_MAX_FILE_SIZE_MB=100
LOGGING_MAX_BACKUP_COUNT=30
```

## Development Mode

For development, use colorized output:
```bash
LOGGING_JSON_OUTPUT=false
LOGGING_COLORIZED=true
LOG GING_LEVEL=DEBUG
```

Example output:
```
2025-11-29 08:16:48 INFO  [main] Episode captured episode_id=12345
```

## Production Best Practices

1. **Use JSON format** for log aggregation
2. **Set level to INFO** (not DEBUG) to reduce volume
3. **Collect logs centrally** (Loki, Elasticsearch, CloudWatch)
4. **Set up alerts** for ERROR level logs
5. **Monitor disk space** for log files

## Log Analysis

### Count requests by endpoint
```bash
jq -r '.endpoint' logs/app.log | sort | uniq -c | sort -rn
```

### Average latency by endpoint
```bash
jq -s 'group_by(.endpoint) | map({endpoint: .[0].endpoint, avg: (map(.duration_ms) | add / length)})' logs/app.log
```

### Error rate
```bash
total=$(grep -c '"level"' logs/app.log)
errors=$(grep -c '"level":"error"' logs/app.log)
echo "Error rate: $(echo "scale=2; $errors / $total * 100" | bc)%"
```

See `OBSERVABILITY.md` for metrics-based monitoring.
