"""
math_utils.py — fixture for integration tests.
"""
from __future__ import annotations


def add(a: int, b: int) -> int:
    """Return sum of a and b."""
    return a + b


def multiply(a: int, b: int) -> int:
    """Return product of a and b."""
    return a * b


class Calculator:
    """Simple calculator supporting basic arithmetic."""

    def __init__(self, initial: int = 0) -> None:
        self.value = initial

    def add(self, n: int) -> Calculator:
        """Add n to current value."""
        self.value = add(self.value, n)
        return self

    def multiply(self, n: int) -> Calculator:
        """Multiply current value by n."""
        self.value = multiply(self.value, n)
        return self

    def result(self) -> int:
        """Return current value."""
        return self.value
