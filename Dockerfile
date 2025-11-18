# Multi-stage Dockerfile for EpisodicMemory API
# Production-optimized container image
#
# Build: docker build -t episodic-memory:latest .
# Run: docker run -p 8000:8000 episodic-memory:latest
#
# Image size optimization:
# - Multi-stage build (builder + runtime)
# - Production dependencies only
# - Non-root user
# - Minimal base image (python:3.11-slim)

# ============================================================================
# Stage 1: Builder - Compile dependencies and wheels
# ============================================================================
FROM python:3.11-slim as builder

# Install system dependencies for building Python packages
# gcc, g++: Required for compiling C extensions (bcrypt, grpcio)
# build-essential: Build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment for isolated dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only requirements first (layer caching optimization)
COPY requirements-prod.txt /tmp/requirements-prod.txt

# Install Python dependencies
# --no-cache-dir: Don't cache pip downloads (saves space)
# --disable-pip-version-check: Skip pip version check (faster)
RUN pip install --no-cache-dir --disable-pip-version-check \
    -r /tmp/requirements-prod.txt

# ============================================================================
# Stage 2: Runtime - Minimal production image
# ============================================================================
FROM python:3.11-slim

# Metadata labels (OCI standard)
LABEL maintainer="EpisodicMemory Team"
LABEL version="0.1.0"
LABEL description="Multi-modal episodic memory API for AI agents"

# Install only runtime system dependencies
# libgomp1: Required for PyTorch (OpenMP)
# ca-certificates: For HTTPS connections (KyroDB TLS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
# UID 1000: Standard non-root user ID
# No password: Service account
RUN useradd --create-home --shell /bin/bash --uid 1000 episodic && \
    mkdir -p /app/data /app/logs && \
    chown -R episodic:episodic /app

# Copy virtual environment from builder
COPY --from=builder --chown=episodic:episodic /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    # Disable PyTorch JIT (faster startup, lower memory)
    PYTORCH_JIT=0 \
    # Use CPU for inference (GPU support via device configuration)
    CUDA_VISIBLE_DEVICES="" \
    # Logging configuration (JSON output for production)
    LOGGING_JSON_OUTPUT=true \
    LOGGING_LEVEL=INFO

# Set working directory
WORKDIR /app

# Copy application code
# .dockerignore ensures only necessary files are copied
COPY --chown=episodic:episodic src/ /app/src/
COPY --chown=episodic:episodic requirements-prod.txt /app/

# Create necessary directories with correct permissions
RUN mkdir -p /app/data/screenshots /app/data/archive /app/logs && \
    chown -R episodic:episodic /app

# Switch to non-root user
USER episodic

# Expose port
EXPOSE 8000

# Health check (uses liveness probe endpoint)
# --interval: Check every 30 seconds
# --timeout: Timeout after 5 seconds
# --start-period: Wait 40 seconds before first check (model loading time)
# --retries: Fail after 3 consecutive failures
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8000/health/liveness || exit 1

# Default command: Run uvicorn server
# --host 0.0.0.0: Listen on all interfaces (required for Docker)
# --port 8000: Default port
# --workers 4: Number of worker processes (adjust based on CPU cores)
# --log-config: Uvicorn logging is disabled (we use structlog)
CMD ["uvicorn", "src.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--log-level", "warning"]

# Alternative: Development mode with auto-reload
# CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
