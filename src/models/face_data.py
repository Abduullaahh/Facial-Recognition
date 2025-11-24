"""
Data models for face recognition.

This module defines data structures used for storing
and passing face encoding information.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class FaceData:
    """
    Data class for storing face encoding information.

    Attributes:
        encodings: List of 128-dimensional face encodings
        names: List of names corresponding to each encoding
    """

    encodings: List[List[float]]
    names: List[str]

    def __len__(self) -> int:
        """Return the number of known faces."""
        return len(self.names)

    def is_empty(self) -> bool:
        """Check if no faces are loaded."""
        return len(self.names) == 0

