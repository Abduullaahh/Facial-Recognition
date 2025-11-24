"""
Visualization utilities for drawing annotations on video frames.

This module provides functions for drawing bounding boxes, labels,
and other visual elements on OpenCV frames.
"""

import cv2
from typing import Tuple


def draw_face_annotations(
    frame: cv2.Mat,
    face_location: Tuple[int, int, int, int],
    name: str,
    is_known: bool = True
) -> None:
    """
    Draw bounding box and label on the frame for a detected face.

    Args:
        frame: OpenCV image frame (BGR format)
        face_location: Tuple of (top, right, bottom, left) coordinates
        name: Name to display (person's name or "Unknown")
        is_known: If True, use green color; if False, use red color
    """
    top, right, bottom, left = face_location

    # Choose color based on recognition status
    color = (0, 255, 0) if is_known else (0, 0, 255)  # Green for known, Red for unknown

    # Draw bounding box rectangle
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # Draw label background rectangle for better text visibility
    cv2.rectangle(
        frame,
        (left, bottom - 35),
        (right, bottom),
        color,
        cv2.FILLED
    )

    # Draw text label
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(
        frame,
        name,
        (left + 6, bottom - 6),
        font,
        0.6,
        (255, 255, 255),
        1
    )


def draw_fps_counter(frame: cv2.Mat, fps: float) -> None:
    """
    Draw FPS counter in the top-left corner of the frame.

    Args:
        frame: OpenCV image frame (BGR format)
        fps: Current frames per second value
    """
    font = cv2.FONT_HERSHEY_DUPLEX
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        fps_text,
        (10, 30),
        font,
        0.7,
        (0, 255, 0),
        2
    )

