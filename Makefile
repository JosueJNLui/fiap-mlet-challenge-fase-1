.PHONY: help install install-dev sync build docker-build docker-run run test test-e2e-httpie lint type-check format clean

# Variáveis
PYTHON := python3
UV := uv
DOCKER_IMAGE := fiap-mlet-challenge-fase-1
DOCKER_TAG := latest
DOCKER_CONTAINER := fiap-mlet
APP_PORT := 8000

help:
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║           FIAP MLET Challenge - Fase 1 Commands               ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "📦 UV Operations:"
	@echo "  make install           → Install dependencies (uv sync)"
	@echo "  make install-dev       → Install dependencies + dev tools"
	@echo "  make sync              → Update uv.lock (same as install)"
	@echo ""
	@echo "🐳 Docker Operations:"
	@echo "  make docker-build      → Build Docker image"
	@echo "  make docker-run        → Run container locally"
	@echo "  make docker-stop       → Stop running container"
	@echo ""
	@echo "🚀 Local Development:"
	@echo "  make run               → Run application with uvicorn"
	@echo "  make dev               → Run with auto-reload"
	@echo ""
	@echo "🧪 Testing & Code Quality:"
	@echo "  make test              → Run pytest"
	@echo "  make test-e2e-httpie   → Run isolated HTTPie E2E tests"
	@echo "  make test-verbose      → Run pytest with verbose output"
	@echo "  make test-cov          → Run pytest with coverage"
	@echo "  make type-check        → Run type checking (ty)"
	@echo "  make lint              → Run linting (ruff check)"
	@echo "  make format            → Format code (ruff format)"
	@echo "  make check             → Run all checks (lint + type-check + test)"
	@echo ""
	@echo "🧹 Utilities:"
	@echo "  make clean             → Remove cache, builds, and temp files"
	@echo "  make help              → Show this help message"
	@echo ""

# ============================================================================
# UV Operations
# ============================================================================

install:
	@echo "📦 Installing dependencies..."
	$(UV) sync --frozen

install-dev:
	@echo "📦 Installing dependencies with dev tools..."
	$(UV) sync --frozen --extra dev

sync: install

# ============================================================================
# Docker Operations
# ============================================================================

docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@echo "✅ Docker image built: $(DOCKER_IMAGE):$(DOCKER_TAG)"

docker-run: docker-build
	@echo "🐳 Running Docker container..."
	docker run -d \
		--name $(DOCKER_CONTAINER) \
		-p $(APP_PORT):8000 \
		$(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo "✅ Container running at http://localhost:$(APP_PORT)"
	@echo "   Container name: $(DOCKER_CONTAINER)"

docker-stop:
	@echo "🛑 Stopping Docker container..."
	docker stop $(DOCKER_CONTAINER) && docker rm $(DOCKER_CONTAINER) || echo "Container not running"
	@echo "✅ Container stopped"

# ============================================================================
# Local Development
# ============================================================================

run:
	@echo "🚀 Starting application..."
	$(UV) run uvicorn src.main:app --host 0.0.0.0 --port $(APP_PORT)

dev:
	@echo "🚀 Starting application with auto-reload..."
	$(UV) run uvicorn src.main:app --host 0.0.0.0 --port $(APP_PORT) --reload

# ============================================================================
# Testing & Code Quality
# ============================================================================

test:
	@echo "🧪 Running tests..."
	$(UV) run pytest

test-e2e-httpie:
	@echo "🧪 Running isolated HTTPie E2E tests..."
	$(UV) run pytest tests/e2e/test_httpie_api.py -q

test-verbose:
	@echo "🧪 Running tests (verbose)..."
	$(UV) run pytest -v

test-cov:
	@echo "🧪 Running tests with coverage..."
	$(UV) run pytest --cov=src --cov-report=xml --cov-report=html --cov-report=term-missing --cov-fail-under=80
	@echo "✅ Coverage reports generated in coverage.xml and htmlcov/index.html"

type-check:
	@echo "📝 Running type check..."
	$(UV) run ty check --exclude "notebooks/" --exclude "**/*.ipynb" --exclude ".venv/" --exclude "__pycache__/" --exclude ".pytest_cache/" --exclude "htmlcov/" --exclude "data/" --exclude "docs/" src/ tests/

lint:
	@echo "🔍 Running linter..."
	$(UV) run ruff check --exclude notebooks/ --exclude "**/*.ipynb" src/ tests/

format:
	@echo "💅 Formatting code..."
	$(UV) run ruff format .

check: lint type-check test
	@echo "✅ All checks passed!"

# ============================================================================
# Utilities
# ============================================================================

clean:
	@echo "🧹 Cleaning up..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "mlruns" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup complete"

# ============================================================================
# Default target
# ============================================================================

.DEFAULT_GOAL := help
