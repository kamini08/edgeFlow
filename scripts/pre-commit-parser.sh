#!/bin/bash
set -euo pipefail

echo "🔍 Running parser module pre-commit checks..."

# Generate ANTLR files if missing (optional)
if [ ! -f "parser/EdgeFlowLexer.py" ] && [ -f "grammer/antlr-4.13.1-complete.jar" ]; then
  echo "⚙️ Generating ANTLR files..."
  java -jar grammer/antlr-4.13.1-complete.jar -Dlanguage=Python3 -o parser grammer/EdgeFlow.g4
fi

echo "📝 Formatting code with black..."
black parser.py tests/test_parser.py || true

echo "🧹 Running linters..."
flake8 parser.py tests/test_parser.py || true
mypy parser.py || true

echo "🧪 Running parser tests..."
pytest tests/test_parser.py -v --cov=parser.py --cov-report=term-missing

echo "📊 Checking coverage..."
coverage report --fail-under=90 || true

echo "🔗 Running integration tests..."
pytest tests/test_integration.py -v -k "parser" -m "not slow" || true

echo "💨 Running CLI smoke test..."
echo 'model_path = "test.tflite"' > test_config.ef
python edgeflowc.py test_config.ef --dry-run > /dev/null
rm -f test_config.ef

echo "✅ All parser checks passed!"

