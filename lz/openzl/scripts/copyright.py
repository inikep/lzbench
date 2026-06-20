#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.


import os
from typing import Optional, Tuple

COPYRIGHT = "Copyright (c) Meta Platforms, Inc. and affiliates."

ROOT = os.path.join(os.path.dirname(__file__), "..")

SUFFIX_TO_COMMENT = {
    "CMakeLists.txt": "#",
    ".cmake": "#",
    ".cmake.in": "#",
    ".h.cmake": "//",
    ".cpp": "//",
    ".hpp": "//",
    ".c": "//",
    ".h": "//",
    ".py": "#",
    ".pyi": "#",
    ".sh": "#",
    ".js": "//",
    ".ts": "//",
    ".yml": "#",
    ".toml": "#",
    ".tsx": "//",
    ".css": ("/*", "*/"),
    ".html.jinja": ("{#-", "-#}"),
    ".html": ("<!--", "-->"),
    "Makefile": "#",
    ".make": "#",
    ".thrift": "//",
    ".bzl": "#",
    "PACKAGE": "#",
    ".bat": "::",
    ".ps1": "#",
    "Doxyfile": "#",
    "static_docs_build_script": "#",
    "BUCK": "#",
    ".proto": "//",
}

DIRS = [
    "benchmark",
    "build",
    "cli",
    "contrib",
    "cpp",
    "custom_parsers",
    "custom_transforms",
    "deps",
    "doc",
    "examples",
    "include",
    "py",
    "scripts",
    "src",
    "tests",
    "tools",
    ".github",
]

FILES = [
    ".autodeps.toml",
    "BUCK",
    "CMakeLists.txt",
    "Makefile",
    "corpus_download.sh",
    "corpus_upload.sh",
    "defs.bzl",
    "PACKAGE",
]

EXCLUDE_PREFIXES = {
    "src/openzl/fse/",
    "examples/compress_data/",
    "cli/tests/sample_files/",
    "deps/",
    "doc/mkdocs/mkdocstrings-zstd/",
}

EXCLUDE_SUFFIXES = [
    "PACKAGE",
    "CODING_STYLE",
    ".md",
    ".gitignore",
    ".pyc",
    ".json",
    ".csv",
    ".svg",
    ".png",
    ".jpg",
    ".ai",
    ".eps",
    ".txt",
    "pdqsort-inl.h",
    "/xxhash.h",
    ".zlc",
    ".prettierignore",
    ".prettierrc",
    "yarn.lock",
]


def exclude_file(file: str) -> bool:
    file = os.path.relpath(file, ROOT)

    for prefix in EXCLUDE_PREFIXES:
        if file.startswith(prefix):
            return True

    for suffix in EXCLUDE_SUFFIXES:
        if file.endswith(suffix):
            return True

    return False


def get_comment_syntax(file: str) -> Optional[Tuple[str, str]]:
    best_suffix = ""
    for suffix in SUFFIX_TO_COMMENT.keys():
        if file.endswith(suffix) and len(suffix) > len(best_suffix):
            best_suffix = suffix
    if best_suffix != "":
        syntax = SUFFIX_TO_COMMENT[best_suffix]
        if isinstance(syntax, str):
            return (syntax, None)
        else:
            return syntax
    else:
        return None


def handle_file(file: str) -> None:
    with open(file, "rb") as f:
        src = f.read()

    try:
        src = src.decode()
    except UnicodeDecodeError:
        print(f"Skipping {file} due to UnicodeDecodeError")
        return

    comment_syntax = get_comment_syntax(file)
    if comment_syntax is None:
        print(f"Skipping {file} due to unknown suffix")
        return
    comment_begin, comment_end = comment_syntax

    dst = ""

    # Skip shebang
    if src.startswith("#!"):
        newline = src.find("\n")
        dst = src[: newline + 1]
        src = src[newline + 1 :]

    newline = src.find("\n")
    maybe_copyright = src[: newline + 1]
    src = src[newline + 1 :]

    has_copyright = (
        "Copyright" in maybe_copyright
        or "(c)" in maybe_copyright
        or "(C)" in maybe_copyright
        or "Meta" in maybe_copyright
        or "Facebook" in maybe_copyright
    )
    if has_copyright and not maybe_copyright.startswith(comment_begin):
        print(
            f"Skipping {file} due to incorrect comment character in copyright line: {maybe_copyright}"
        )
        return

    dst += f"{comment_begin} {COPYRIGHT}"
    if comment_end is not None:
        dst += f" {comment_end}"
    dst += "\n"
    if not has_copyright:
        dst += "\n"
        dst += maybe_copyright
    dst += src

    assert COPYRIGHT in dst

    with open(file, "w") as f:
        f.write(dst)


def main() -> None:
    for file in FILES:
        handle_file(os.path.join(ROOT, file))
    for directory in DIRS:
        for dirpath, _dirnames, filenames in os.walk(os.path.join(ROOT, directory)):
            for file in filenames:
                path = os.path.join(dirpath, file)
                if exclude_file(path):
                    continue

                handle_file(path)


if __name__ == "__main__":
    main()
