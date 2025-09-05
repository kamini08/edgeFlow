.PHONY: pre-commit lint-backend lint-frontend test-backend test-frontend run-local

pre-commit: lint-backend lint-frontend test-backend test-frontend

lint-backend:
	cd backend && black --check . && isort --profile black --check-only . && flake8 . && mypy --ignore-missing-imports .

lint-frontend:
	cd frontend && npm run lint && npm run type-check

test-backend:
	cd backend && pytest --cov=backend --cov-report=term-missing -q

test-frontend:
	cd frontend && npm test -- --coverage

run-local:
	docker-compose up --build

