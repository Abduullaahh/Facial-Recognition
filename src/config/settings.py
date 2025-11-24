"""
Configuration settings for the Facial Recognition System.

This module contains all configuration constants and settings
used throughout the application.
"""

from pathlib import Path
from typing import Final

# Directory and file paths
KNOWN_FACES_DIRECTORY: Final[str] = "known_faces"
ENCODINGS_PICKLE_FILE: Final[str] = "encodings.pkl"

# Image processing settings
SUPPORTED_IMAGE_EXTENSIONS: Final[set[str]] = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

# Camera settings
DEFAULT_CAMERA_INDEX: Final[int] = 0

# Face recognition settings
DEFAULT_TOLERANCE: Final[float] = 0.6  # Lower values = stricter matching

# FPS calculation settings
FPS_RESET_INTERVAL: Final[int] = 30  # Reset FPS calculation every N frames

# Display settings
WINDOW_NAME: Final[str] = "Facial Recognition"
QUIT_KEY: Final[str] = "q"

