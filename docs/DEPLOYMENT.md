# Deployment Guide

Production deployment guide for EpisodicMemory (Phase 3 Week 8-10).

## Overview

EpisodicMemory uses a **hybrid cloud deployment model**:

- **Application (EpisodicMemory API)**: Containerized, runs on Kubernetes/cloud
- **Vector Database (KyroDB)**: Bare metal for maximum performance
- **Customer Database**: Cloud-managed PostgreSQL (or SQLite for development)

This guide covers:
1. Container image building
2. Local development with Docker Compose
3. Kubernetes deployment
4. CI/CD pipeline setup

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Cloud / Kubernetes                    │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │  EpisodicMemory API (3-10 replicas)              │  │
│  │  - Load balanced                                  │  │
│  │  - Auto-scaling                                   │  │
│  │  - Health checks                                  │  │
│  └──────────────────────────────────────────────────┘  │
│                         │                                │
│                         │ gRPC/TLS                       │
│                         ▼                                │
└──────────────────────────┬──────────────────────────────┘
                           │
                           │ VPN/Private Network
                           │
┌──────────────────────────▼──────────────────────────────┐
│                    Bare Metal Servers                    │
│                                                          │
│  ┌────────────────────┐    ┌────────────────────┐      │
│  │  KyroDB Text       │    │  KyroDB Image      │      │
│  │  Instance          │    │  Instance          │      │
│  │  (High IOPS SSD)   │    │  (High IOPS SSD)   │      │
│  └────────────────────┘    └────────────────────┘      │
└─────────────────────────────────────────────────────────┘
```

## Prerequisites

### Required

- **Docker**: 20.10+ (for building images)
- **Docker Compose**: 2.0+ (for local development)
- **Kubernetes**: 1.25+ (for production deployment)
- **kubectl**: Configured with cluster access
- **Git**: For source control

### Optional

- **Helm**: 3.0+ (for Kubernetes package management)
- **Skaffold**: For development workflows
- **Buildah/Podman**: Alternative to Docker

## Local Development

### Quick Start with Docker Compose

**1. Clone repository**:
```bash
git clone https://github.com/your-org/episodic-memory.git
cd episodic-memory
```

**2. Create environment file**:
```bash
cp .env.example .env
# Edit .env with your configuration
```

**3. Start services**:
```bash
docker-compose up -d
```

**Services started**:
- EpisodicMemory API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

**4. View logs**:
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f episodic-memory
```

**5. Stop services**:
```bash
docker-compose down
```

### Development with Hot-Reload

The `docker-compose.yml` uses `Dockerfile.dev` which mounts source code as a volume:

```bash
# Make code changes in src/
# Changes are reflected immediately (no rebuild needed)

# Restart if needed
docker-compose restart episodic-memory
```

### Running Tests

```bash
# Run tests in container
docker-compose exec episodic-memory pytest tests/ -v

# Run specific test file
docker-compose exec episodic-memory pytest tests/test_health.py -v
```

## Building Container Images

### Production Image

**Build**:
```bash
docker build -t episodic-memory:latest .
```

**Build with specific version**:
```bash
docker build -t episodic-memory:v0.1.0 .
```

**Build for multiple platforms** (multi-arch):
```bash
docker buildx build --platform linux/amd64,linux/arm64 \
  -t episodic-memory:latest \
  --push .
```

### Development Image

```bash
docker build -f Dockerfile.dev -t episodic-memory:dev .
```

### Image Size Optimization

**Multi-stage build** reduces image size:
- Builder stage: ~2.5 GB (includes build tools)
- Runtime stage: ~1.2 GB (slim Python + dependencies)

**Check image size**:
```bash
docker images | grep episodic-memory
```

**Analyze layers**:
```bash
docker history episodic-memory:latest
```

## Pushing to Registry

### Docker Hub

```bash
# Tag image
docker tag episodic-memory:latest your-org/episodic-memory:latest

# Push
docker push your-org/episodic-memory:latest
```

### AWS ECR

```bash
# Authenticate
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-west-2.amazonaws.com

# Tag image
docker tag episodic-memory:latest \
  123456789012.dkr.ecr.us-west-2.amazonaws.com/episodic-memory:latest

# Push
docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/episodic-memory:latest
```

### Google Container Registry (GCR)

```bash
# Authenticate
gcloud auth configure-docker

# Tag image
docker tag episodic-memory:latest gcr.io/your-project/episodic-memory:latest

# Push
docker push gcr.io/your-project/episodic-memory:latest
```

## Kubernetes Deployment

### Prerequisites

**1. Create namespace**:
```bash
kubectl create namespace episodic-memory
```

**2. Create ConfigMap**:
```bash
kubectl apply -f k8s/configmap.yaml -n episodic-memory
```

**3. Create Secrets** (IMPORTANT: Update with real values):
```bash
# Edit k8s/configmap.yaml with your secrets
# Then apply
kubectl apply -f k8s/configmap.yaml -n episodic-memory
```

### Deploy Application

**1. Deploy**:
```bash
kubectl apply -f k8s/deployment.yaml -n episodic-memory
```

**2. Verify deployment**:
```bash
# Check pods
kubectl get pods -n episodic-memory

# Check deployment
kubectl get deployment episodic-memory -n episodic-memory

# View logs
kubectl logs -f deployment/episodic-memory -n episodic-memory
```

**3. Check health**:
```bash
# Port-forward to access locally
kubectl port-forward deployment/episodic-memory 8000:8000 -n episodic-memory

# Test health endpoint
curl http://localhost:8000/health/liveness
curl http://localhost:8000/health/readiness
```

### Scaling

**Manual scaling**:
```bash
kubectl scale deployment episodic-memory --replicas=5 -n episodic-memory
```

**Auto-scaling** (HPA already configured in deployment.yaml):
```bash
# View HPA status
kubectl get hpa episodic-memory -n episodic-memory

# Describe HPA
kubectl describe hpa episodic-memory -n episodic-memory
```

### Rolling Updates

**Update image**:
```bash
kubectl set image deployment/episodic-memory \
  api-server=episodic-memory:v0.2.0 \
  -n episodic-memory
```

**Monitor rollout**:
```bash
kubectl rollout status deployment/episodic-memory -n episodic-memory
```

**Rollback if needed**:
```bash
kubectl rollout undo deployment/episodic-memory -n episodic-memory
```

## Configuration Management

### Environment Variables

Configuration is managed via:
1. **ConfigMap** (non-sensitive config)
2. **Secrets** (API keys, passwords, certificates)
3. **Environment variables** (override defaults)

### Updating Configuration

**Update ConfigMap**:
```bash
# Edit k8s/configmap.yaml
kubectl apply -f k8s/configmap.yaml -n episodic-memory

# Restart pods to pick up changes
kubectl rollout restart deployment/episodic-memory -n episodic-memory
```

**Update Secrets**:
```bash
# Create secret from file
kubectl create secret generic episodic-memory-secrets \
  --from-file=llm-api-key=./secrets/llm-api-key.txt \
  -n episodic-memory \
  --dry-run=client -o yaml | kubectl apply -f -

# Or edit directly (base64 encoded)
kubectl edit secret episodic-memory-secrets -n episodic-memory
```

## Monitoring

### Prometheus Metrics

Metrics are automatically scraped if using Prometheus Operator:

```yaml
annotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "8000"
  prometheus.io/path: "/metrics"
```

**Manual scraping** (if not using operator):
```bash
# Port-forward Prometheus
kubectl port-forward svc/prometheus 9090:9090

# Add scrape config pointing to episodic-memory service
```

### Grafana Dashboards

**Import dashboards**:
1. Access Grafana (port-forward or LoadBalancer)
2. Go to Dashboards → Import
3. Upload `grafana/dashboards/service-health-overview.json`
4. Upload `grafana/dashboards/business-metrics.json`

### Logs

**View logs** (Kubernetes):
```bash
# All pods
kubectl logs -l app=episodic-memory -n episodic-memory --tail=100

# Specific pod
kubectl logs episodic-memory-5d7f8c9b4-x7k2p -n episodic-memory -f
```

**Structured logging** (JSON output):
Logs are in JSON format for easy aggregation:
- ELK Stack (Elasticsearch + Logstash + Kibana)
- Grafana Loki
- AWS CloudWatch Logs
- Google Cloud Logging

## Troubleshooting

### Pod Not Starting

**Check pod status**:
```bash
kubectl describe pod <pod-name> -n episodic-memory
```

**Common issues**:
- Image pull errors: Check image name and registry credentials
- Resource limits: Pods may be evicted if limits too low
- Config errors: Check ConfigMap and Secrets

### Health Check Failures

**Check health endpoints**:
```bash
kubectl port-forward pod/<pod-name> 8000:8000 -n episodic-memory
curl http://localhost:8000/health/readiness
```

**Common issues**:
- KyroDB connection failures: Check network connectivity
- Database connection failures: Verify DATABASE_URL
- Missing models: Check embedding service initialization

### High Memory Usage

**Check resource usage**:
```bash
kubectl top pod -n episodic-memory
```

**Optimize**:
- Reduce `UVICORN_WORKERS`
- Set `EMBEDDING_DEVICE=cpu` if not using GPU
- Increase memory limits in deployment.yaml

## Security Best Practices

### Container Security

- ✓ Non-root user (UID 1000)
- ✓ Read-only root filesystem (Phase 3 final)
- ✓ No privileged containers
- ✓ Dropped capabilities

### Network Security

- ✓ TLS for KyroDB connections
- ✓ Network policies (restrict pod-to-pod traffic)
- ✓ Service mesh (Istio/Linkerd for mTLS) - Phase 4

### Secret Management

**DO NOT**:
- Commit secrets to Git
- Use default passwords
- Share API keys across environments

**DO**:
- Use Kubernetes Secrets or external secret management
- Rotate secrets regularly
- Use different secrets for each environment

## Performance Tuning

### Worker Configuration

```yaml
env:
  - name: UVICORN_WORKERS
    value: "4"  # Adjust based on CPU cores
```

**Rule of thumb**: `(2 x num_cores) + 1`

### Resource Limits

```yaml
resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: 2000m
    memory: 4Gi
```

Adjust based on:
- Request rate
- Model size (embeddings)
- Number of workers

### Connection Pooling

KyroDB connection pooling is configured via:
```yaml
env:
  - name: KYRODB_MAX_WORKERS
    value: "10"
```

## CI/CD Integration

See `docs/CI_CD.md` (Phase 3 Week 10) for:
- GitHub Actions workflow
- GitLab CI pipeline
- Jenkins pipeline
- Automated testing
- Image scanning
- Deployment automation

## Next Steps

**Phase 3 Week 10**: CI/CD Pipeline
- Automated builds on commit
- Container scanning for vulnerabilities
- Automated deployment to staging/production
- Blue-green deployments

**Phase 4**: Production hardening
- Circuit breakers
- Response caching
- Database connection pooling
- Advanced monitoring

## References

- [Dockerfile Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)
- [Multi-stage Builds](https://docs.docker.com/build/building/multi-stage/)
