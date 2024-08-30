SRC_PATH 	= nlnas
VENV_PATH	= venv
DOCS_PATH 	= docs

PDOC		= pdoc -d google --math
PYTHON		= python3.10

.ONESHELL:

all: format typecheck lint

.PHONY: docs
docs:
	-@mkdir $(DOCS_PATH) > /dev/null 2>&1
	$(PYTHON) -m $(PDOC) --output-directory $(DOCS_PATH) $(SRC_PATH)

.PHONY: docs-browser
docs-browser:
	-@mkdir $(DOCS_PATH) > /dev/null 2>&1
	$(PYTHON) -m $(PDOC) -h 0.0.0.0 -p 8081 -n $(SRC_PATH)

.PHONY: format
format:
	$(PYTHON) -m isort $(SRC_PATH)
	$(PYTHON) -m black $(SRC_PATH)

.PHONY: lint
lint:
	$(PYTHON) -m pylint $(SRC_PATH)

.PHONY: profile
profile:
	-@mkdir prof > /dev/null 2>&1
	$(PYTHON) -m cProfile -o prof/$(shell date +%Y%m%d_%H%M%S).prof \
		-m $(SRC_PATH) run -n 1000

.PHONY: typecheck
typecheck:
	$(PYTHON) -m mypy -p $(SRC_PATH)
