# Incident Response Runbooks

Quick procedures for common issues.

## 1. High Error Rate

**Symptoms**: Error rate >1%, HTTP 5xx responses

**Diagnosis**:
```bash
curl http://localhost:8000/metrics | grep episodic_memory_errors_total
curl http://localhost:8000/health | jq
```

**Fix**:
1. Check error type: validation/auth/kyrodb/llm
2. If auth errors: restart service
3. If KyroDB errors: see #3 below
4. If validation errors: check recent schema changes

## 2. High Latency

**Symptoms**: P99 >50ms

**Diagnosis**:
```bash
curl http://localhost:8000/metrics | grep http_request_duration
curl http://localhost:8000/metrics | grep kyrodb_operation_duration
```

**Fix**:
1. Check KyroDB health: `curl http://localhost:8000/health`
2. If KyroDB slow: restart KyroDB instances
3. If traffic spike: increase workers or add rate limiting

## 3. KyroDB Connection Failure

**Symptoms**: Health shows kyrodb unhealthy, all requests failing

**Diagnosis**:
```bash
lsof -i :50051  # Check text instance
lsof -i :50052  # Check image instance
```

**Fix**:
```bash
# Restart both instances
./target/release/kyrodb_server --config kyrodb_config.toml --data-dir ./data/kyrodb
./target/release/kyrodb_server --config kyrodb_config_images.toml

# Verify
curl http://localhost:8000/health | jq '.components[] | select(.name=="kyrodb")'
```

## 4. LLM Budget Exceeded

**Symptoms**: Daily cost >$50 (if using paid models)

**Note**: With OpenRouter free tier, this shouldn't happen unless you switched to paid models.

**Fix**:
```bash
# Check budget
curl http://localhost:8000/admin/budget | jq

# Increase limit if needed
export REFLECTION_MAX_COST_PER_DAY_USD=100
# Restart service
```

## 5. Reflection Generation Failures

**Symptoms**: Reflections not appearing on episodes

**Diagnosis**:
```bash
grep -i "openrouter\|llm" logs/app.log | tail -20
curl https://status.openrouter.ai/api/v2/status.json | jq
```

**Fix**:
1. Check API key: Test with OpenRouter directly
2. If key invalid: rotate key in `.env`
3. If timeout: increase `LLM_TIMEOUT_SECONDS`
4. Restart service

## 6. API Key Compromise

**Symptoms**: Unusual traffic, unexpected customer activity

**Immediate Actions**:
```bash
# Suspend customer
curl -X PATCH http://localhost:8000/api/v1/customers/$CUSTOMER_ID \
  -H "X-Admin-API-Key: $ADMIN_KEY" \
  -d '{"status": "suspended"}'

# Rotate key
curl -X POST http://localhost:8000/api/v1/customers/$CUSTOMER_ID/rotate-key \
  -H "X-Admin-API-Key: $ADMIN_KEY"
```

## 7. Service Unresponsive

**Symptoms**: High CPU/memory, OOM kills

**Fix**:
```bash
# Check resources
top -l 1 | grep uvicorn

# Restart service
pkill -f "uvicorn.*app" && sleep 2
uvicorn src.main:app --port 8000
```

## 8. Data Recovery

**Episode Recovery**:
```bash
# Export episodes for customer
python scripts/export_episodes.py --customer-id=$CUSTOMER_ID --output=backup.json
```

**Database Recovery**:
```bash
# Backup current
cp data/customers.db data/customers.db.bak

# Restore from backup
cp ${BACKUP_DIR}/customers.db.$(date +%Y%m%d) data/customers.db

# Verify
sqlite3 data/customers.db "PRAGMA integrity_check"
```

## Quick Health Check

```bash
# Full system status
curl http://localhost:8000/health | jq

# Check components
curl http://localhost:8000/health | jq '.components'

# Metrics
curl http://localhost:8000/metrics | grep -E "http_request|kyrodb_operation"
```

## Escalation

| Severity | Response | Contact |
|----------|----------|---------|
| Critical | 15 min | On-call engineer |
| High | 1 hour | Team lead |
| Medium | 4 hours | Ticket |
| Low | 24 hours | Normal queue |

**Contact**: kishan@kyrodb.com
