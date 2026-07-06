.DEFAULT_GOAL := help

COMPOSE := docker compose

.PHONY: help
help: ## Show available commands
	@echo "Usage: make <target>"
	@echo ""
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Install all dependencies (including dev)
	uv sync

.PHONY: env
env: ## Create .env from .env.example (if missing)
	@test -f .env && echo ".env already exists" || (cp .env.example .env && echo ".env created")

.PHONY: hooks
hooks: ## Install pre-commit hooks
	uv run pre-commit install

.PHONY: run
run: ## Run the app locally (http://localhost:7860)
	uv run python -m src.main

.PHONY: lint
lint: ## Check code style (ruff)
	uv run ruff check
	uv run ruff format --check

.PHONY: format
format: ## Auto-fix style and format code
	uv run ruff check --fix
	uv run ruff format

.PHONY: typecheck
typecheck: ## Run static type checking (ty)
	uv run ty check

.PHONY: test
test: ## Run tests with coverage
	uv run pytest

.PHONY: check
check: lint typecheck test ## Run all checks (lint + typecheck + test)

.PHONY: pre-commit
pre-commit: ## Run pre-commit hooks on all files
	uv run pre-commit run --all-files

.PHONY: build
build: ## Build the Docker image
	$(COMPOSE) build

.PHONY: up
up: ## Start the service in the background
	$(COMPOSE) up -d

.PHONY: down
down: ## Stop and remove containers
	$(COMPOSE) down

.PHONY: restart
restart: down up ## Restart the service

.PHONY: logs
logs: ## Tail service logs
	$(COMPOSE) logs -f

.PHONY: clean
clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
