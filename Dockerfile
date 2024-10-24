# syntax=docker/dockerfile:1

# Build stage
FROM python:3.10-slim AS builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only the files needed for installation
COPY requirements.txt setup.py README.md ./
COPY aide ./aide

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install -e .

# Runtime stage
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        unzip \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 aide

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
# Copy the package files needed for the installation
COPY --from=builder /app /app
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Create and set permissions for logs and workspaces
RUN mkdir -p logs workspaces && \
    chown -R aide:aide /app

# Switch to non-root user
USER aide

# Set default command
ENTRYPOINT ["aide"]
