from dataclasses import dataclass
from enum import Enum


class InputState(Enum):
    """Input signal states, syntax similar to paper"""
    RISING  = "↑"
    FALLING = "↓"
    LOW     = "0-"  # stable at 0
    HIGH    = "1-"  # stable at 1

@dataclass
class InputTransition:
    """One entry in the input sequence I = ((x_i, y_i, t_i))."""
    x: InputState   # input A
    y: InputState   # input B
    t: float        # transition time

@dataclass
class OutputTransition:
    o: int      # 0 or 1
    t_p: float    # Output transition time