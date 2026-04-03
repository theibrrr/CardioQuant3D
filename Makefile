.PHONY: install train evaluate infer api docker-build lint test clean

# ─── Installation ───────────────────────────────────────────────
install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pre-commit install

# ─── Training ───────────────────────────────────────────────────
train:
	python scripts/train.py

# ─── Evaluation ─────────────────────────────────────────────────
evaluate:
	python scripts/evaluate.py

# ─── Inference ──────────────────────────────────────────────────
infer:
	@echo "Usage: make infer INPUT=<path_to_nifti>"
	python scripts/infer.py --input $(INPUT)

# ─── API Server ─────────────────────────────────────────────────
api:
	uvicorn cardioquant3d.api.main:app --host 0.0.0.0 --port 8000 --reload

# ─── Docker ─────────────────────────────────────────────────────
docker-build:
	docker build -t cardioquant3d:latest .

docker-run:
	docker run -p 8000:8000 -v ./outputs:/app/outputs cardioquant3d:latest

# ─── Code Quality ───────────────────────────────────────────────
lint:
	black .
	isort .
	flake8 .
	mypy cardioquant3d/

lint-check:
	black --check .
	isort --check-only .
	flake8 .
	mypy cardioquant3d/

# ─── Testing ────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=cardioquant3d --cov-report=term-missing

# ─── Cleanup ────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf outputs/ mlruns/ .eggs/ *.egg-info dist/ build/
