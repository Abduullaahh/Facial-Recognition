"""
Service for real-time facial recognition.

This module handles video capture, face detection, encoding comparison,
and displaying recognition results.
"""

import cv2
import face_recognition

from src.config import (
    DEFAULT_CAMERA_INDEX,
    DEFAULT_TOLERANCE,
    FACE_DETECTION_MODEL,
    FRAME_PROCESS_SCALE,
    QUIT_KEY,
    WINDOW_NAME,
)
from src.models.face_data import FaceData
from src.utils.fps_calculator import FPSCalculator
from src.utils.visualization import draw_face_annotations, draw_fps_counter


class RecognitionService:
    """
    Service for performing real-time facial recognition.

    Handles video capture, frame processing, face detection,
    encoding comparison, and result visualization.
    """

    def __init__(
        self,
        face_data: FaceData,
        camera_index: int = DEFAULT_CAMERA_INDEX,
        tolerance: float = DEFAULT_TOLERANCE,
        frame_scale: float = FRAME_PROCESS_SCALE,
        detection_model: str = FACE_DETECTION_MODEL
    ) -> None:
        """
        Initialize the recognition service.

        Args:
            face_data: FaceData object containing known encodings and names
            camera_index: Index of the camera to use
            tolerance: Lower values make recognition stricter
            frame_scale: Scale factor for detection (smaller = faster)
            detection_model: "hog" or "cnn" for face_locations
        """
        self.face_data = face_data
        self.camera_index = camera_index
        self.tolerance = tolerance
        self.frame_scale = max(0.1, min(1.0, frame_scale))
        self.detection_model = detection_model
        self.fps_calculator = FPSCalculator()

    def run_recognition(self) -> None:
        """
        Run real-time facial recognition using webcam feed.

        Processes video frames, detects faces, compares with known encodings,
        and displays results with bounding boxes and labels.
        """
        # Check if we have any known faces
        if self.face_data.is_empty():
            print("Error: No known faces loaded. Please add images to known_faces directory.")
            return

        # Initialize video capture
        video_capture = cv2.VideoCapture(self.camera_index)

        if not video_capture.isOpened():
            print(f"Error: Could not open camera {self.camera_index}.")
            return

        print("Starting real-time facial recognition...")
        print(f"Press '{QUIT_KEY}' to quit.")

        # Process video frames
        try:
            self._process_video_stream(video_capture)
        finally:
            # Clean up resources
            video_capture.release()
            cv2.destroyAllWindows()
            print("Recognition stopped.")

    def _process_video_stream(self, video_capture: cv2.VideoCapture) -> None:
        """
        Process video stream frames in a loop.

        Args:
            video_capture: OpenCV VideoCapture object
        """
        while True:
            # Read frame from camera
            ret, frame = video_capture.read()

            if not ret:
                print("Error: Failed to read frame from camera.")
                break

            # Process frame: detect and recognize faces
            self._process_frame(frame)

            # Calculate and draw FPS
            fps = self.fps_calculator.update()
            draw_fps_counter(frame, fps)

            # Display the resulting frame
            cv2.imshow(WINDOW_NAME, frame)

            # Break loop on quit key press
            if cv2.waitKey(1) & 0xFF == ord(QUIT_KEY):
                break

    def _process_frame(self, frame: cv2.Mat) -> None:
        """
        Process a single frame: detect faces and draw annotations.

        Args:
            frame: OpenCV image frame (BGR format)
        """
        inv_scale = 1.0 / self.frame_scale
        if self.frame_scale < 1.0:
            small = cv2.resize(frame, (0, 0), fx=self.frame_scale, fy=self.frame_scale)
        else:
            small = frame

        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        small_locations = face_recognition.face_locations(
            rgb_small,
            model=self.detection_model
        )
        face_locations = [
            (
                int(top * inv_scale),
                int(right * inv_scale),
                int(bottom * inv_scale),
                int(left * inv_scale),
            )
            for top, right, bottom, left in small_locations
        ]

        rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_full, face_locations)

        # Compare each detected face with known faces
        for face_encoding, face_location in zip(face_encodings, face_locations):
            name = self._identify_face(face_encoding)
            is_known = name != "Unknown"
            draw_face_annotations(frame, face_location, name, is_known)

    def _identify_face(self, face_encoding: list[float]) -> str:
        """
        Identify a face by comparing its encoding with known encodings.

        Args:
            face_encoding: 128-dimensional face encoding

        Returns:
            Name of the person if matched, "Unknown" otherwise
        """
        # Compare face encoding with known encodings
        matches = face_recognition.compare_faces(
            self.face_data.encodings,
            face_encoding,
            tolerance=self.tolerance
        )

        # Find the best match if any
        if True in matches:
            # Calculate face distances to find the closest match
            face_distances = face_recognition.face_distance(
                self.face_data.encodings,
                face_encoding
            )

            # Get the index of the best match (lowest distance)
            best_match_index = face_distances.argmin()

            # If match is found, use the corresponding name
            if matches[best_match_index]:
                return self.face_data.names[best_match_index]

        return "Unknown"

