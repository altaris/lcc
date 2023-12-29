SRC_PATH 		= nlnas
VENV_PATH		= ./venv
DOCS_PATH 		= docs

PYTHON			= python3.11
PDOC			= $(PYTHON) -m pdoc -d google --math
BLACK			= $(PYTHON) -m black --line-length 79 --target-version py310
MYPY			= $(PYTHON) -m mypy
PYLINT			= $(PYTHON) -m pylint

.ONESHELL:

all: format typecheck lint

.PHONY: docs
docs:
	-@mkdir $(DOCS_PATH) > /dev/null 2>&1
	$(PDOC) --output-directory $(DOCS_PATH) $(SRC_PATH)

.PHONY: docs-browser
docs-browser:
	-@mkdir $(DOCS_PATH) > /dev/null 2>&1
	$(PDOC) $(SRC_PATH)

.PHONY: format
format:
	$(BLACK) $(SRC_PATH)
	$(BLACK) *.py

.PHONY: lint
lint:
	$(PYLINT) $(SRC_PATH)

.PHONY: typecheck
typecheck:
	$(MYPY) -p $(SRC_PATH)
	$(MYPY) *.py
