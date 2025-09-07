FROM python:3.11-slim

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

EXPOSE 8000
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
