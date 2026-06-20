#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.


"""
Builds the docs in an OSS environment.

Requires:
    Python: mkdocs>=1.6.1
    Python: mkdocstrings>=0.27.0
    Python: mkdocs-material>=9.5.50
    System: doxygen
    System: clang-format
"""

import os
import sys
from pathlib import Path
from subprocess import call

MKDOCS_DIR = Path(__file__).parent
MKDOCSTRINGS_ZSTD = MKDOCS_DIR / "mkdocstrings-zstd" / "src"
MKDOCS_OPENZL = MKDOCS_DIR / "mkdocs-openzl" / "src"


def add_to_pythonpath(env, path: Path):
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (pythonpath + ":" if pythonpath else "") + str(path)


# Hack plugins into PYTHONPATH for mkdocs
env = os.environ.copy()
add_to_pythonpath(env, MKDOCSTRINGS_ZSTD)
add_to_pythonpath(env, MKDOCS_OPENZL)

# Grab MKDOCS from the env or default to the system mkdocs
MKDOCS = env.get("MKDOCS", "mkdocs")

ARGS = sys.argv[1:]
if len(ARGS) == 0:
    ARGS = ["build"]

sys.exit(call([MKDOCS] + ARGS, cwd=MKDOCS_DIR, env=env))
