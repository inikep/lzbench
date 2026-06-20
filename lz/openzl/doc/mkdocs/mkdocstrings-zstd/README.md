# mkdocstrings-zstd
Custom mkdocstrings handler for Zstd headers

## Deps

* See `pyproject.toml` for Python dependencies
* doxygen
* clang-format

## TODO

* Support Doxygen @ref in comments and replace with autoref tag. This should be straightforward.

## Wish

* Support Doxygen @ref in code blocks. This probably means hooking into Pygments, unless I can find something more clever.
* Collapsible descriptions
