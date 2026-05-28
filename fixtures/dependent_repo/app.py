"""
app.py — dependent application.
"""
from math_utils import Calculator

class SuperCalculator(Calculator):
    """A calculator that inherits from math_utils.Calculator."""
    
    def square(self) -> "SuperCalculator":
        """Square the current value."""
        self.value = self.value * self.value
        return self
