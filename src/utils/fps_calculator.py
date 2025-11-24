"""
FPS (Frames Per Second) calculator utility.

This module provides a class for calculating and tracking
frame rates in real-time video processing.
"""

import time
from typing import Final

from src.config import FPS_RESET_INTERVAL


class FPSCalculator:
    """
    Calculate and track frames per second for video processing.

    This class maintains internal state for FPS calculation,
    resetting the counter periodically for more responsive updates.
    """

    def __init__(self, reset_interval: int = FPS_RESET_INTERVAL) -> None:
        """
        Initialize the FPS calculator.

        Args:
            reset_interval: Number of frames before resetting FPS calculation
        """
        self.reset_interval: Final[int] = reset_interval
        self.frame_count: int = 0
        self.start_time: float = time.time()
        self.current_fps: float = 0.0

    def update(self) -> float:
        """
        Update FPS calculation based on elapsed time.

        Returns:
            Current FPS value
        """
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time

        if elapsed_time > 0:
            self.current_fps = self.frame_count / elapsed_time

        # Reset calculation periodically for more responsive updates
        if self.frame_count % self.reset_interval == 0:
            self.frame_count = 0
            self.start_time = time.time()

        return self.current_fps

    def get_fps(self) -> float:
        """
        Get the current FPS value without updating.

        Returns:
            Current FPS value
        """
        return self.current_fps

    def reset(self) -> None:
        """Reset the FPS calculator to initial state."""
        self.frame_count = 0
        self.start_time = time.time()
        self.current_fps = 0.0

