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
