"""One-liner path setup for Jupyter notebooks (run from `notebooks/` or project root)."""
from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    cwd = Path.cwd().resolve()
    if cwd.name == "notebooks":
        return cwd.parent
    return cwd


def add_project_to_path() -> Path:
    root = project_root()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root
