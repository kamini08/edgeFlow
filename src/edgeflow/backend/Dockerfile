FROM python:3.9-slim

WORKDIR /app

# Install curl for healthchecks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install backend runtime deps
COPY src/edgeflow/backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy backend app and core modules used by the API
COPY src/edgeflow/backend /app/backend
COPY src/edgeflow/compiler/edgeflowc.py /app/edgeflowc.py
COPY src/edgeflow/parser /app/parser
COPY src/edgeflow/optimization/optimizer.py /app/optimizer.py
COPY src/edgeflow/ir/edgeflow_ast.py /app/edgeflow_ast.py
COPY src/edgeflow/compiler/code_generator.py /app/code_generator.py
COPY src/edgeflow/benchmarking/benchmarker.py /app/benchmarker.py
COPY src/edgeflow/reporting/reporter.py /app/reporter.py
COPY src/edgeflow/config/device_specs.py /app/device_specs.py
COPY src/edgeflow/analysis/initial_check.py /app/initial_check.py

EXPOSE 8000

# Health check endpoint with initial delay for startup
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
