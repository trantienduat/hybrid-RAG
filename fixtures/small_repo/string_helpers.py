"""
string_helpers.py — fixture for integration tests.
"""
from __future__ import annotations

import re

from math_utils import add


def slugify(text: str) -> str:
    """Convert text to URL-friendly slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s_-]+", "-", text)


def repeat(s: str, n: int) -> str:
    """Repeat string s exactly n times."""
    result = ""
    for _ in range(n):
        result = result + s
    return result


def word_count(text: str) -> int:
    """Count words in text."""
    return add(len(text.split()), 0)


class StringProcessor:
    """Processes strings with pluggable transforms."""

    def __init__(self) -> None:
        self._transforms: list = []

    def add_transform(self, fn) -> StringProcessor:
        """Register a transform function."""
        self._transforms.append(fn)
        return self

    def process(self, text: str) -> str:
        """Apply all registered transforms in order."""
        for fn in self._transforms:
            text = fn(text)
        return text
