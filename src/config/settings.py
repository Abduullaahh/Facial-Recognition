"""
Configuration settings for the Facial Recognition System.

This module contains all configuration constants and settings
used throughout the application.
"""

from pathlib import Path
from typing import Final

# Resolve paths from package location so the app works regardless of CWD
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent

# Directory and file paths (absolute)
KNOWN_FACES_DIRECTORY: Final[str] = str(_PROJECT_ROOT / "known_faces")
ENCODINGS_PICKLE_FILE: Final[str] = str(_PROJECT_ROOT / "encodings.pkl")

# Image processing settings
SUPPORTED_IMAGE_EXTENSIONS: Final[set[str]] = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

# Camera settings
DEFAULT_CAMERA_INDEX: Final[int] = 0

# Face recognition settings
DEFAULT_TOLERANCE: Final[float] = 0.6  # Lower values = stricter matching

# Resize frames before detection (1.0 = full resolution; lower = faster)
FRAME_PROCESS_SCALE: Final[float] = 0.25

# HOG is fast for CPU; "cnn" is slower but can be more accurate
FACE_DETECTION_MODEL: Final[str] = "hog"

# FPS calculation settings
FPS_RESET_INTERVAL: Final[int] = 30  # Reset FPS calculation every N frames

# Display settings
WINDOW_NAME: Final[str] = "Facial Recognition"
QUIT_KEY: Final[str] = "q"

