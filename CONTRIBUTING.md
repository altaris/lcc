# Dependencies

- `python3.10`,
- [`uv`](https://docs.astral.sh/uv/),
- `make` (optional).

# Documentation

Simply run

```sh
make docs
```

This will generate the HTML doc of the project, and the index file should be at
`docs/index.html`. To have it directly in your browser, run

```sh
make docs-browser
```

# Code quality

Don't forget to run

```sh
make
```

to format and check the code using [`ruff`](https://docs.astral.sh/ruff/) and
typecheck it using [mypy](http://mypy-lang.org/).
