# CI/CD Workflows

This directory contains GitHub Actions workflows for continuous integration.

## Workflows

### `ci.yml`
Main CI workflow that runs on every push and pull request to `main` and `develop` branches.

**Jobs:**
1. **lint-and-format**: Checks code formatting and runs linting across Python 3.9, 3.10, and 3.11
2. **validate-imports**: Validates that all Python files can be imported without errors
3. **check-notebooks**: Validates Jupyter notebook JSON structure

**Checks:**
- Code formatting with Black
- Linting with flake8
- Python syntax validation
- Import validation
- Notebook structure validation

## Running Locally

To run the same checks locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Check formatting
black --check code/

# Run linting
flake8 code/

# Validate syntax
python -m py_compile code/*.py

# Validate imports (see scripts/validate_imports.py)
python scripts/validate_imports.py
```

## Configuration Files

- `.flake8`: Flake8 linting configuration
- `pyproject.toml`: Black and Pylint configuration
- `requirements.txt`: Python dependencies

