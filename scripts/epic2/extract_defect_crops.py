#!/usr/bin/env python3
"""Thin launcher for `udm-epic2` CLI (Epic 2 crop extraction)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from udm_epic2.cli_epic2 import app

if __name__ == "__main__":
    app()
