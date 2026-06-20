#!/usr/bin/env python3
# A script to invoke mkdocs with the correct environment.
# Additionally supports deploying via mike:
#   ./mkdocs deploy

import os
import sys
from pathlib import Path
from subprocess import call

example_dir = Path(__file__).parent
root_dir = example_dir.parent
zstd_handler_dir = root_dir / "src"
config_path = os.path.join(example_dir, "mkdocs.yml")

# Set PYTHONPATH for the mkdocstrings handler.
env = os.environ.copy()
path = env.get("PYTHONPATH")
env["PYTHONPATH"] = (path + ":" if path else "") + str(zstd_handler_dir)

args = sys.argv[1:]
sys.exit(call(["mkdocs"] + args, env=env))
