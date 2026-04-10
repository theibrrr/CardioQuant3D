# Contributing

## Setup

1. Create and activate a Python 3.11 environment.
2. Install the project in editable mode with development dependencies:

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

## Lint Commands

```bash
black --check .
isort --check-only .
```

## Pull Request Checklist

- Tests updated when behavior changes
- README updated when public behavior changes
- Lint passes locally
