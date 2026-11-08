# ===========================================================================
# Local-runner header. Used in place of 00_header.py when assembling a script
# to run in this project's uv-managed .venv rather than on Kaggle.
#
# No pip subprocess here: dependencies are pinned in pyproject.toml and
# installed by `uv sync`, and `uv venv` environments don't ship pip by default,
# so a pip subprocess call would fail before the real code even runs.
# ===========================================================================

import json
import os
import time

import numpy as np
import torch

print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    print("running on CPU -- this is the local pipeline smoke test, not a real "
          "elicitation run", flush=True)
