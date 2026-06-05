.PHONY: up down index serve test lint install clean

# ── Infrastructure ─────────────────────────────────────────────────
up:
	docker compose up -d
	@echo "FalkorDB: localhost:6379  |  Qdrant: localhost:6333"

down:
	docker compose down

restart: down up

# ── Dev setup ──────────────────────────────────────────────────────
install:
	python3.12 -m venv .venv
	. .venv/bin/activate && pip install -e ".[dev,api,eval,mcp]"

# ── Indexing ───────────────────────────────────────────────────────
# Usage: make index REPO=/path/to/target/repo
index:
	. .venv/bin/activate && hybrid-rag index $(REPO)

# ── Querying ───────────────────────────────────────────────────────
# Usage: make query Q="your question"
query:
	. .venv/bin/activate && hybrid-rag query "$(Q)"

# ── Status ─────────────────────────────────────────────────────────
status:
	. .venv/bin/activate && hybrid-rag status

# ── Tests ──────────────────────────────────────────────────────────
test:
	pytest tests/ -v --cov=hybrid_rag --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

# ── Lint ───────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/

# ── Cleanup ────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov/
