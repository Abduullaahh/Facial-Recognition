"""
Webcam enrollment: capture a still and add it to known_faces.
"""

import os

import cv2
import face_recognition

from src.config import (
    DEFAULT_CAMERA_INDEX,
    FACE_DETECTION_MODEL,
    FRAME_PROCESS_SCALE,
    QUIT_KEY,
    WINDOW_NAME,
)
from src.services.encoding_service import EncodingService


class EnrollmentService:
    """Capture one face from the webcam and save an image for encoding."""

    def __init__(
        self,
        encoding_service: EncodingService,
        camera_index: int = DEFAULT_CAMERA_INDEX,
        frame_scale: float = FRAME_PROCESS_SCALE,
        detection_model: str = FACE_DETECTION_MODEL
    ) -> None:
        self.encoding_service = encoding_service
        self.camera_index = camera_index
        self.frame_scale = max(0.1, min(1.0, frame_scale))
        self.detection_model = detection_model

    def enroll_from_camera(self, person_name: str) -> bool:
        """
        Open the camera until the user saves one frame with exactly one face.

        Returns:
            True if an image was saved and encodings refreshed.
        """
        video_capture = cv2.VideoCapture(self.camera_index)
        if not video_capture.isOpened():
            print(f"Error: Could not open camera {self.camera_index}.")
            return False

        window = f"{WINDOW_NAME} - Enroll: {person_name}"
        print(f"Enrolling '{person_name}'. Align one face. SPACE = save, '{QUIT_KEY}' = quit.")
        inv_scale = 1.0 / self.frame_scale

        try:
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    print("Error: Failed to read frame from camera.")
                    break

                small = (
                    cv2.resize(frame, (0, 0), fx=self.frame_scale, fy=self.frame_scale)
                    if self.frame_scale < 1.0
                    else frame
                )
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                locs = face_recognition.face_locations(rgb_small, model=self.detection_model)
                n_faces = len(locs)

                hint = "One face: press SPACE to save"
                color = (0, 200, 255)
                if n_faces == 0:
                    hint = "No face detected"
                    color = (0, 165, 255)
                elif n_faces > 1:
                    hint = "Show only one face"
                    color = (0, 0, 255)

                for top, right, bottom, left in locs:
                    t, r, b, l = (
                        int(top * inv_scale),
                        int(right * inv_scale),
                        int(bottom * inv_scale),
                        int(left * inv_scale),
                    )
                    cv2.rectangle(frame, (l, t), (r, b), (255, 255, 0), 2)

                cv2.putText(
                    frame,
                    hint,
                    (10, 30),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.7,
                    color,
                    2,
                )
                cv2.imshow(window, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(QUIT_KEY):
                    print("Enrollment cancelled.")
                    break
                if key == ord(" ") and n_faces == 1:
                    path = self.encoding_service.next_enrollment_path(person_name)
                    if not cv2.imwrite(str(path), frame):
                        print(f"Error: could not write {path}")
                        continue
                    enc, _ = self.encoding_service._extract_encoding_from_image(
                        path, verbose=False
                    )
                    if enc is None:
                        try:
                            os.remove(path)
                        except OSError:
                            pass
                        print("Saved image had no usable face. Try again with better lighting.")
                        continue
                    print(f"Saved {path.name}. Refreshing encodings...")
                    self.encoding_service.refresh_encodings()
                    print("Done. You can run recognition now.")
                    return True
        finally:
            video_capture.release()
            cv2.destroyAllWindows()

        return False
