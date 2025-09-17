# Multi-stage build for optimized image size
FROM python:3.9-slim as builder

# Build stage - compile dependencies
WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Download ANTLR
RUN wget https://www.antlr.org/download/antlr-4.13.1-complete.jar \
    -O /usr/local/lib/antlr-4.13.1-complete.jar

# Copy requirements
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    default-jre-headless \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /usr/local/lib/antlr-4.13.1-complete.jar /usr/local/lib/

# Set PATH
ENV PATH=/root/.local/bin:$PATH
ENV ANTLR_JAR=/usr/local/lib/antlr-4.13.1-complete.jar

# Copy application code
COPY . /app/

# Generate ANTLR files (to parser/ from grammer/EdgeFlow.g4)
RUN test -f grammer/EdgeFlow.g4 && \
    java -jar $ANTLR_JAR -Dlanguage=Python3 -o parser grammer/EdgeFlow.g4 || true

# Create directories for models and outputs
RUN mkdir -p /app/models /app/outputs /app/logs /app/device_specs /app/configs

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command
ENTRYPOINT ["python", "edgeflowc.py"]
CMD ["--help"]
