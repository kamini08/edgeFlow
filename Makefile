.PHONY: pre-commit lint-backend lint-frontend test-backend test-frontend run-local

pre-commit: lint-backend lint-frontend test-backend test-frontend

lint-backend:
	cd backend && python3 -m black --check . && python3 -m isort --profile black --check-only . && python3 -m flake8 . && python3 -m mypy --ignore-missing-imports --follow-imports=skip .

lint-frontend:
	cd frontend && npm run lint && npm run type-check

test-backend:
	cd backend && python3 -m pytest --cov=backend --cov-report=term-missing -q

test-frontend:
	cd frontend && npm test -- --coverage

run-local:
	docker-compose up --build

# ------------------------------
# Docker operations
# ------------------------------

.PHONY: docker-build docker-up docker-down docker-logs docker-test docker-clean

docker-build:
	@echo "🔨 Building Docker images..."
	docker-compose build --parallel

docker-up:
	@echo "🚀 Starting EdgeFlow services..."
	docker-compose up -d
	@echo "✅ Services started!"
	@echo "   API: http://localhost:8000"
	@echo "   Frontend: http://localhost:3000"

docker-down:
	@echo "⏹️  Stopping EdgeFlow services..."
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-test:
	@echo "🧪 Running tests in Docker..."
	docker run --rm -v $(PWD):/app edgeflow:latest pytest tests/

docker-clean:
	@echo "🧹 Cleaning up Docker resources..."
	docker-compose down -v || true
	docker system prune -f || true
