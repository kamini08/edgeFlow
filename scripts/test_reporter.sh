#!/bin/bash
set -euo pipefail

echo "📊 Running Reporter Module Verification"
echo "========================================"

# Step 1: Verify all documentation was read (presence only)
echo "📚 Documentation checklist:"
for file in README.md Agents.md docs/*.md; do
    if [ -f "$file" ]; then
        echo "✓ Found: $file"
    fi
done

# Step 2: Run reporter tests
echo -e "\n🧪 Running reporter tests..."
source .venv/bin/activate 2>/dev/null || true
python -m pytest tests/test_reporter.py -v --cov=reporter --cov-report=term-missing

# Step 3: Check coverage
echo -e "\n🧮 Checking coverage..."
python -m coverage report --fail-under=90

# Step 4: Lint reporter module
echo -e "\n🧹 Running linters..."
python -m black --check reporter.py tests/test_reporter.py || (echo "Run: black reporter.py tests/test_reporter.py" && exit 1)
python -m flake8 reporter.py --max-line-length=100
python -m mypy reporter.py --strict --ignore-missing-imports

# Step 5: Integration test
echo -e "\n🔗 Running integration test..."
python - <<'PY'
from reporter import generate_report
import tempfile
import os

# Test with dummy data
stats = {'size_mb': 10.0, 'latency_ms': 20.0}
with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as f:
    report_path = f.name

generate_report(stats, stats, output_path=report_path)
assert os.path.exists(report_path)
os.unlink(report_path)
print('✅ Integration test passed')
PY

echo -e "\n✅ All reporter checks passed!"
echo "Ready to raise PR after running:"
echo "  git add -A"
echo "  git commit -m 'feat(day4): implement reporter module with comprehensive metrics'"
echo "  git push origin feature/day4-reporter"

