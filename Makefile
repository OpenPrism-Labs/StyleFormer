.PHONY: install dev-install test test-cov lint format clean clean-outputs train train-multi-attr prepare-data verify-data help

# Installation
install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

# Code quality
lint:
	ruff check src/ scripts/ tests/
	mypy src/

format:
	ruff format src/ scripts/ tests/
	ruff check --fix src/ scripts/ tests/

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html

# Data smoke checks (no model training)
train:
	python scripts/train.py

train-multi-attr:
	python scripts/train.py experiments=multi_attr

# Data
prepare-data:
	python scripts/prepare_data.py --dataset all

verify-data:
	python scripts/prepare_data.py --verify

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage

clean-outputs:
	rm -rf outputs/
	rm -rf logs/
	rm -rf wandb/

# Help
help:
	@echo "Available targets:"
	@echo "  install        - Install package"
	@echo "  dev-install    - Install with dev dependencies"
	@echo "  lint           - Run linters"
	@echo "  format         - Format code"
	@echo "  test           - Run tests"
	@echo "  train          - Inspect one data batch (no model training)"
	@echo "  train-multi-attr - Inspect multi-attribute data (no model training)"
	@echo "  prepare-data   - Show dataset download instructions"
	@echo "  verify-data    - Verify datasets"
	@echo "  clean          - Clean build artifacts"
	@echo "  clean-outputs  - Clean training outputs"
