FROM python:3.9-slim

WORKDIR /app

# Install curl for healthchecks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install backend runtime deps
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy backend app and core modules used by the API
COPY backend /app/backend
COPY edgeflowc.py /app/edgeflowc.py
COPY parser /app/parser
COPY parser.py /app/parser.py
COPY optimizer.py /app/optimizer.py
COPY edgeflow_ast.py /app/edgeflow_ast.py
COPY code_generator.py /app/code_generator.py
COPY benchmarker.py /app/benchmarker.py
COPY reporter.py /app/reporter.py
COPY device_specs.py /app/device_specs.py
COPY initial_check.py /app/initial_check.py

EXPOSE 8000

# Health check endpoint with initial delay for startup
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
