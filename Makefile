.DEFAULT_GOAL := help

help:  ## Show available commands
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Install website and PyTorch learning dependencies
	uv sync --extra dev --group docs

docs-only:  ## Install only lightweight website dependencies
	uv sync --only-group docs --locked

serve:  ## Preview the notes site at http://127.0.0.1:3408/
	uv run --only-group docs --locked mkdocs serve

build:  ## Build the site strictly and catch broken navigation
	uv run --only-group docs --locked mkdocs build --strict

test:  ## Run model and RLCD tests
	uv run --extra dev python -m pytest -q

examples:  ## Run both guided examples
	uv run python -m examples.inspect_modernbert
	uv run python -m examples.train_rlcd_toy

.PHONY: help setup docs-only serve build test examples
