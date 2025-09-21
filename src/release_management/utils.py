"""
Core utilities for release management.

This module provides minimal functionality for file operations needed
for release creation in the benchmarking framework.
"""

import json
import typing as t
from pathlib import Path


def save_json(data: t.Any, file_path: Path, *, indent: int = 2) -> None:
    """
    Save data as JSON file.

    Args:
        data: Data to save (must be JSON serializable)
        file_path: Path to save JSON to
        indent: JSON indentation for readability

    """
    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)


def ensure_directory(directory: Path) -> None:
    """
    Ensure directory exists, creating it if necessary.

    Args:
        directory: Directory path to ensure exists

    """
    directory.mkdir(parents=True, exist_ok=True)