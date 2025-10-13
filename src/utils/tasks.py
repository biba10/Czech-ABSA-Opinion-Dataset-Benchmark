from enum import Enum


class Task(Enum):
    """Task."""
    ASTE = "aste"
    ACOS = "acos"
    ASQP = "asqp"

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)
