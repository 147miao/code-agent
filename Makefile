.PHONY: install test lint test-graph ci index graph api fmt

install:
	uv venv .venv --python 3.13
	uv pip install --python .venv/bin/python -e .
	uv pip install --python .venv/bin/python pytest pytest-asyncio ruff

test:
	PYTHONPATH=src .venv/bin/python -m pytest tests -q

test-graph:
	PYTHONPATH=src .venv/bin/python -m pytest tests/test_codegraph_neo4j.py -q -m integration

lint:
	.venv/bin/python -m ruff check src tests scripts

ci: lint test

index:
	PYTHONPATH=src .venv/bin/python scripts/index_code.py --root . --name codeagent --reindex

graph:
	PYTHONPATH=src .venv/bin/python scripts/build_graph.py --root . --name codeagent --reindex --with-llm

api:
	PYTHONPATH=src .venv/bin/uvicorn codeagent.api.main:app --host 127.0.0.1 --port 8000

fmt:
	.venv/bin/python -m ruff check src tests scripts --fix
