.PHONY: install dev-install test lint format clean train prepare-data

# Installation
install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"
	pre-commit install

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

# Training
train:
	python scripts/train.py

train-multi-attr:
	python scripts/train.py experiment=multi_attr

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
	@echo "  train          - Run training (default config)"
	@echo "  train-multi-attr - Run multi-attribute training"
	@echo "  prepare-data   - Show dataset download instructions"
	@echo "  verify-data    - Verify datasets"
	@echo "  clean          - Clean build artifacts"
	@echo "  clean-outputs  - Clean training outputs"
