DOCS_PATH 		= docs
SRC_PATH 		= nlnas
VENV			= ./venv
PDOC			= pdoc -d google --math

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
	# isort .
	black --line-length 79 --target-version py310 $(SRC_PATH)
	black --line-length 79 --target-version py310 *.py

.PHONY: lint
lint:
	pylint $(SRC_PATH)

.PHONY: typecheck
typecheck:
	mypy -p $(SRC_PATH)
	mypy *.py
