from enum import Enum

class Signal(str, Enum):
    LONG = "A"
    SHORT = "S"
    FLAT = "F"
    NONE = ""
