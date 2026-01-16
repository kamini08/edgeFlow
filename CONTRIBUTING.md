# Contributing to EdgeFlow

Thank you for your interest in contributing to EdgeFlow! This document provides guidelines and instructions for contributing.

## Code of Conduct

This project adheres to the Contributor Covenant [code of conduct](CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Issues

- Check existing issues before creating a new one
- Provide clear description, steps to reproduce, and expected vs actual behavior
- Include relevant logs, error messages, and environment details

### Submitting Pull Requests

1. **Fork the repository**
2. **Create a feature branch** from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Run tests and linters**
   ```bash
   pytest tests/
   black src/ tests/
   flake8 src/ tests/
   mypy src/ --ignore-missing-imports
   ```
5. **Commit your changes**
   - Use conventional commit format: `feat:`, `fix:`, `docs:`, `refactor:`, etc.
   - Write clear, descriptive commit messages
6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Open a Pull Request**
   - Provide clear description of changes
   - Reference related issues
   - Ensure CI passes

## Development Setup

### Prerequisites

- Python 3.11+
- Git

### Installation

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/edgeFlow.git
cd edgeFlow

# Install dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=edgeflow --cov-report=term

# Run specific test file
pytest tests/test_pipeline_v2.py -v
```

### Code Quality Standards

All contributions must meet these standards:

1. **Code Formatting**

   - Use `black` for Python code formatting
   - Line length: 100 characters max

   ```bash
   black src/ tests/
   ```

2. **Linting**

   - Pass `flake8` checks

   ```bash
   flake8 src/ tests/ --max-line-length=100
   ```

3. **Type Checking**

   - Pass `mypy` type checks

   ```bash
   mypy src/ --ignore-missing-imports
   ```

4. **Test Coverage**

   - Maintain test coverage ≥75%
   - Add tests for new features
   - Update tests when modifying existing code

5. **Documentation**
   - Add docstrings to all public APIs
   - Update README.md for user-facing changes
   - Include inline comments for complex logic

## Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `ci`: CI/CD changes
- `chore`: Maintenance tasks

### Examples

```
feat(compiler): add support for PyTorch models
fix(parser): handle edge case in quantization config
docs: update installation instructions
refactor(ir): simplify graph transformation logic
test(pipeline): add integration tests for optimization
```

## Project Structure Guidelines

- **src/edgeflow/**: Core source code (organized by feature/module)
- **tests/**: Test files (mirror src/ structure)
- **scripts/**: Build and utility scripts
- **.github/workflows/**: CI/CD configurations

## Pull Request Guidelines

### Before Submitting

- [ ] Code is formatted with `black`
- [ ] Linting passes (`flake8`)
- [ ] Type checking passes (`mypy`)
- [ ] All tests pass
- [ ] Test coverage maintained or improved
- [ ] Documentation updated
- [ ] Commit messages follow conventional format

### PR Description Template

```markdown
## Description

Brief description of changes

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing

Describe testing performed

## Checklist

- [ ] Code follows project style
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] CI passes
```

## Adding New Features

1. **Discuss first**: Open an issue to discuss major features
2. **Design**: Plan the implementation approach
3. **Implement**: Write code following project standards
4. **Test**: Add comprehensive tests
5. **Document**: Update documentation
6. **Review**: Submit PR for review

## Questions or Help

- Open an issue for questions
- Tag maintainers for assistance
- Join discussions in existing issues/PRs

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
