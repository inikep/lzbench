#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.


import argparse
import copy
import importlib.resources
import os
import shutil
import subprocess
import tempfile


parser = argparse.ArgumentParser()
parser.add_argument("--pwd", type=str, required=True)
parser.add_argument("--test", action="store_true")
parser.add_argument(
    "--use-system-python-extension",
    action="store_true",
    help="Use system openzl Python extension instead of building with pip",
)
parser.add_argument("args", nargs="*", default=[])
args = parser.parse_args()

PWD = args.pwd

ARGS = args.args

if len(ARGS) == 0 or os.getenv("FB_INTERNAL") is not None:
    # The internal builder passes a storage prefix as the only argument
    ARGS = ["build"]

env = copy.deepcopy(os.environ)
if args.use_system_python_extension:
    env["OPENZL_USE_SYSTEM_PYTHON_EXTENSION"] = "1"

with importlib.resources.path(__package__, "mkdocs") as mkdocs:
    cmd = [str(mkdocs)] + ARGS

    if args.test:
        SITE_DIR = tempfile.mkdtemp()
        cmd += ["--site-dir", SITE_DIR]
    else:
        SITE_DIR = os.path.join(PWD, "build")

    try:
        subprocess.check_call(cmd, cwd=PWD)

        if not os.path.exists(os.path.join(SITE_DIR, "index.html")):
            raise RuntimeError("No index.html found in site directory")
    finally:
        if args.test:
            shutil.rmtree(SITE_DIR, ignore_errors=True)
