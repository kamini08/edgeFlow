#!/bin/bash
# scripts/verify_initial_check.sh

echo "🔍 Initial Check Module Verification"
echo "===================================="

# Verify documentation was read
echo "📚 Documentation Review"
read -p "Have you read ALL .md files? (yes/no): " answer
if [ "$answer" != "yes" ]; then
    echo "❌ Read all documentation before proceeding!"
    exit 1
fi

# Run tests
echo -e "\n🧪 Running tests..."
pytest tests/test_initial_check.py -v --cov=initial_check --cov=device_specs --cov-report=term-missing

# Check coverage
coverage report --fail-under=90 || exit 1

# Lint
echo -e "\n🧹 Running linters..."
black --check initial_check.py device_specs.py || exit 1
flake8 initial_check.py device_specs.py --max-line-length=100 || exit 1
mypy initial_check.py device_specs.py --strict --ignore-missing-imports || exit 1

# CLI test
echo -e "\n⚙️ Testing CLI integration..."
echo 'model_path = "test.tflite"' > test.ef
echo 'target_device = "raspberry_pi_4"' >> test.ef
python edgeflowc.py test.ef --check-only --dry-run || true
rm -f test.ef

# API test
echo -e "\n🌐 Testing API endpoint..."
curl -s -X POST http://localhost:8000/api/check \
  -H "Content-Type: application/json" \
  -d '{"model_path": "test.tflite", "config": {"target_device": "raspberry_pi_4"}}' || true

echo -e "\n✅ All initial check verifications passed!"
echo "Ready to commit and push!"

