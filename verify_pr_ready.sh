#!/bin/bash
# Final PR Readiness Check for Day 2 Parser Module

echo "📋 Final PR Readiness Check"
echo "=========================="

READY=true

# Check git status
echo -n "Git status clean: "
if [ -z "$(git status --porcelain)" ]; then
    echo "✅"
else
    echo "❌ (commit or stash changes)"
    READY=false
fi

# Check branch
echo -n "On feature branch: "
BRANCH=$(git branch --show-current)
if [[ $BRANCH == feature/* ]] || [[ $BRANCH == day2/* ]]; then
    echo "✅ ($BRANCH)"
else
    echo "⚠️  Consider using feature branch naming"
fi

# Run all checks
echo -n "Parser tests pass: "
pytest tests/test_parser.py -q && echo "✅" || { echo "❌"; READY=false; }

echo -n "Integration tests pass: "
pytest tests/test_integration.py -q -k "parser" && echo "✅" || { echo "❌"; READY=false; }

echo -n "Linting passes: "
(black --check parser.py 2>/dev/null && flake8 parser.py 2>/dev/null) && echo "✅" || { echo "❌"; READY=false; }

echo -n "Type checking passes: "
mypy parser.py --ignore-missing-imports 2>/dev/null && echo "✅" || { echo "❌"; READY=false; }

echo -n "Coverage >90%: "
COV=$(pytest tests/test_parser.py --cov=parser --cov-report=term | grep TOTAL | awk '{print $4}' | sed 's/%//')
if [ "${COV%.*}" -ge 90 ]; then
    echo "✅ ($COV%)"
else
    echo "❌ ($COV%)"
    READY=false
fi

echo "=========================="
if [ "$READY" = true ]; then
    echo "🎉 Ready to raise PR!"
    echo ""
    echo "Next steps:"
    echo "1. git add -A"
    echo "2. git commit -m 'feat: implement EdgeFlow parser module with full integration'"
    echo "3. git push origin $BRANCH"
    echo "4. Create PR with template below"
else
    echo "❌ Not ready for PR. Fix issues above."
    exit 1
fi

