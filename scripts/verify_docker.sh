#!/bin/bash
# scripts/verify_docker.sh

echo "🐳 Docker Module Verification"
echo "============================="

# Check Docker and Docker Compose
echo "Checking Docker installation..."
docker --version || { echo "❌ Docker not installed"; exit 1; }
docker-compose --version || { echo "❌ Docker Compose not installed"; exit 1; }

echo -e "\n🔨 Building Docker images..."
docker build -t edgeflow:test . || exit 1
docker build -t edgeflow-api:test -f backend/Dockerfile . || exit 1
docker build -t edgeflow-frontend:test -f frontend/Dockerfile frontend/ || exit 1

echo -e "\n📋 Validating docker-compose.yml..."
docker-compose config > /dev/null || exit 1

echo -e "\n🧪 Running Docker tests..."
pytest tests/test_docker.py -v -m docker || exit 1

echo -e "\n🚀 Testing service startup..."
docker-compose up -d
sleep 10
docker-compose ps
curl -f http://localhost:8000/api/health || { echo "❌ API health check failed"; exit 1; }
docker-compose down

echo -e "\n🧹 Cleaning up test images..."
docker rmi edgeflow:test edgeflow-api:test edgeflow-frontend:test || true

echo -e "\n✅ Docker setup complete and verified!"
echo "Ready to commit and create PR!"

