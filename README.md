# Episodic Memory for AI Agents

Application-layer episodic memory system built on top of [KyroDB](https://github.com/KyroDB/KyroDB) - a high-performance vector database optimized for RAG workloads and AI agents.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Episodic Memory Service (Python FastAPI)                    │
│ - Multi-modal ingestion (text/code/images)                 │
│ - Precondition-aware retrieval                              │
│ - Automated hygiene (decay & promotion)                     │
└─────────────────────────────────────────────────────────────┘
                        ↓ (gRPC)
┌─────────────────────────────────────────────────────────────┐
│ KyroDB (2 instances)                                        │
│ - Text/code embeddings (384-dim)                           │
│ - Image embeddings (512-dim via CLIP)                      │
│ - 3-tier caching (71.7% hit rate)                          │
│ - <1ms P99 vector search                                    │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Multi-modal storage**: Text, code, and image embeddings
- **Namespace-based collections**: failures, skills, semantic_rules
- **Precondition matching**: LLM-powered relevance filtering
- **Temporal queries**: Filter by timestamp ranges
- **Automated hygiene**: Time-based decay and usage-based pruning
- **Pattern promotion**: Episodic → Semantic memory via clustering

## Performance Targets

- **<50ms P99** retrieval latency
- **10K-100K** episodes per collection
- **Multi-modal search** across text + images

## Quick Start

### Prerequisites

- Python 3.11+
- KyroDB running (2 instances: text + images)
- Redis (for Celery background jobs)

### Installation

```bash
# Clone repository
git clone https://github.com/KyroDB/EpisodicMemory.git
cd EpisodicMemory

# Install dependencies
poetry install

# Setup environment
cp .env.example .env
# Edit .env with KyroDB connection details
```

### Running Locally

```bash
# Terminal 1: Start KyroDB (text embeddings)
cd ../ProjectKyro
./target/release/kyrodb_server --port 50051 --data-dir ./data/kyrodb_text

# Terminal 2: Start KyroDB (image embeddings)
./target/release/kyrodb_server --port 50052 --data-dir ./data/kyrodb_images

# Terminal 3: Start Episodic Memory service
cd ../EpisodicMemory
poetry run uvicorn src.main:app --reload --port 8000
```

### API Usage

```python
import requests

# Capture an episode
response = requests.post("http://localhost:8000/api/capture", json={
    "goal": "Deploy web application to production",
    "tool_chain": ["kubectl", "docker"],
    "actions_taken": [
        "Built Docker image",
        "Pushed to registry",
        "Applied Kubernetes manifest"
    ],
    "error_trace": "ImagePullBackOff: failed to resolve image",
    "code_state_diff": "# git diff output",
    "screenshot_path": "./screenshots/deploy_error.png"
})
episode_id = response.json()["episode_id"]

# Search for relevant failures
response = requests.post("http://localhost:8000/api/search", json={
    "goal": "Deploy app with kubectl",
    "tool": "kubectl",
    "current_state": {"cluster": "production", "namespace": "default"},
    "k": 5
})
similar_failures = response.json()["results"]
```

## Project Status

- **Phase 0 (Week 1)**: Repository setup & infrastructure ✅ **IN PROGRESS**
- **Phase 1 (Week 2-3)**: Core ingestion pipeline
- **Phase 2 (Week 4-5)**: Retrieval with preconditions
- **Phase 3 (Week 6-7)**: Background hygiene (decay/promotion)

## Development

```bash
# Run tests
poetry run pytest

# Type checking
poetry run mypy src/

# Format code
poetry run black src/ tests/
poetry run isort src/ tests/
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

BSL License - see [LICENSE](LICENSE) for details.

## Related Projects

- [KyroDB](https://github.com/KyroDB/KyroDB) - High-performance vector database
