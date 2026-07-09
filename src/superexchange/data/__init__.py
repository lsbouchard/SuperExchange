"""Bundled example datasets."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def example_data_path(name: str = "ErOEr1+.csv") -> Path:
    """Return a filesystem path for a bundled example dataset."""
    return Path(str(files(__package__).joinpath(name)))
