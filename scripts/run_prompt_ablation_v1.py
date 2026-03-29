#!/usr/bin/env python3
"""Backward-compatible entry: writes ../v1结果.md. See run_prompt_ablation.py --prompt v1."""
from __future__ import annotations

import sys
from pathlib import Path

_scripts = Path(__file__).resolve().parent
if str(_scripts.parent) not in sys.path:
    sys.path.insert(0, str(_scripts.parent))
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from run_prompt_ablation import main

if __name__ == "__main__":
    main("v1")
