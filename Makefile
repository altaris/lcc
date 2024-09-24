SRC_PATH 	= nlnas
VENV_PATH	= venv
DOCS_PATH 	= docs/html

PDOC		= pdoc -d google --math
PYTHON		= python3.10

.ONESHELL:

all: format typecheck lint

.PHONY: docs
docs:
	-@mkdir $(DOCS_PATH) > /dev/null 2>&1
	PDOC_ALLOW_EXEC=1 uv run $(PDOC) --output-directory $(DOCS_PATH) $(SRC_PATH)

.PHONY: docs-browser
docs-browser:
	-@mkdir $(DOCS_PATH) > /dev/null 2>&1
	PDOC_ALLOW_EXEC=1 uv run $(PDOC) -p 8081 -n $(SRC_PATH)

.PHONY: format
format:
	uvx ruff check --select I --fix
	uvx ruff format

.PHONY: lint
lint:
	uvx ruff check

.PHONY: typecheck
typecheck:
	uv run mypy -p $(SRC_PATH)
