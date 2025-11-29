# Observability & Monitoring

Track Vritti's health and performance.

## Quick Start

### View Metrics
```bash
curl http://localhost:8000/metrics
```

### Health Check
```bash
curl http://localhost:8000/health | jq
```

## Key Metrics

### Request Metrics
- `episodic_memory_http_request_duration_seconds` - Request latency (histogram)
- `episodic_memory_http_requests_total` - Total requests (counter)
- `episodic_memory_http_requests_active` - Active requests (gauge)

### KyroDB Metrics
- `episodic_memory_kyrodb_operation_duration_seconds` - KyroDB latency
- `episodic_memory_kyrodb_operations_total` - Total operations
- `episodic_memory_kyrodb_connection_healthy` - Connection health (1=healthy, 0=unhealthy)

### Business Metrics
- `episodic_memory_episodes_ingested_total` - Episodes captured
- `episodic_memory_searches_total` - Search requests
- `episodic_memory_gating_decision_total` - Gating calls

### Error Metrics
- `episodic_memory_errors_total` - Errors by type
- `episodic_memory_rate_limit_exceeded_total` - Rate limit violations

## Useful Queries

### Check Error Rate
```bash
curl -s http://localhost:8000/metrics | grep errors_total
```

### Check Request Latency
```bash
curl -s http://localhost:8000/metrics | grep http_request_duration
```

### Check KyroDB Health
```bash
curl -s http://localhost:8000/metrics | grep kyrodb_connection_healthy
```

## Prometheus Setup (Optional)

### prometheus.yml
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'vritti'
    static_configs:
      - targets: ['localhost:8000']
```

### Run Prometheus
```bash
prometheus --config.file=prometheus.yml
```

Access at: `http://localhost:9090`

## Grafana Dashboards (Optional)

Connect Grafana to Prometheus, then create dashboards for:
- Request rate & latency
- Error rates
- KyroDB performance
- Business metrics (episodes/searches)

## Health Checks

### Liveness Probe
```bash
curl http://localhost:8000/health/liveness
```

Returns 200 if service is running.

### Readiness Probe
```bash
curl http://localhost:8000/health/readiness
```

Returns 200 if service is ready (KyroDB connected, models loaded).

### Full Health
```bash
curl http://localhost:8000/health | jq
```

Shows all component statuses.

## Performance Targets

- **Search latency**: <50ms P99
- **Request latency**: <100ms P99
- **Error rate**: <1%
- **KyroDB connection**: 100% uptime

## Troubleshooting

**Metrics not appearing**:
```bash
curl http://localhost:8000/metrics | grep episodic_memory
```

**Prometheus not scraping**:
```bash
curl http://localhost:9090/api/v1/targets
```

**High latency**:
- Check KyroDB health
- Check metrics for slow operations
- See `RUNBOOKS.md` for detailed procedures

## Structured Logging

Logs are in JSON format (configurable via `.env`):
```bash
LOGGING_JSON_OUTPUT=true
LOGGING_LEVEL=INFO
```

View logs:
```bash
tail -f logs/app.log | jq
```

Filter errors:
```bash
grep '"level":"error"' logs/app.log | jq
```

See `STRUCTURED_LOGGING.md` for log format details.
