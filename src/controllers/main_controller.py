"""
Main controller for the Facial Recognition System.

This module serves as the entry point and orchestrates
the application's main workflow.
"""

from src.config import ENCODINGS_PICKLE_FILE, KNOWN_FACES_DIRECTORY
from src.services.encoding_service import EncodingService
from src.services.enrollment_service import EnrollmentService
from src.services.recognition_service import RecognitionService


class MainController:
    """
    Main controller that orchestrates the facial recognition application.

    Coordinates between encoding and recognition services to provide
    the complete facial recognition workflow.
    """

    def __init__(
        self,
        known_faces_dir: str = KNOWN_FACES_DIRECTORY,
        encodings_file: str = ENCODINGS_PICKLE_FILE
    ) -> None:
        """
        Initialize the main controller.

        Args:
            known_faces_dir: Path to directory containing known face images
            encodings_file: Path to pickle file storing encodings
        """
        self.encoding_service = EncodingService(
            known_faces_dir=known_faces_dir,
            encodings_file=encodings_file
        )

    def run(
        self,
        force_reload_encodings: bool = False,
        camera_index: int | None = None,
        tolerance: float | None = None,
        frame_scale: float | None = None,
        detection_model: str | None = None,
    ) -> None:
        """
        Run the facial recognition application.

        Loads known faces and starts real-time recognition.

        Args:
            force_reload_encodings: If True, regenerate encodings even if pickle exists
            camera_index: Camera device index (default from settings)
            tolerance: Match tolerance (default from settings)
            frame_scale: Detection resize scale (default from settings)
            detection_model: "hog" or "cnn"
        """
        face_data = self.encoding_service.load_known_faces(force_reload=force_reload_encodings)

        kwargs: dict = {}
        if camera_index is not None:
            kwargs["camera_index"] = camera_index
        if tolerance is not None:
            kwargs["tolerance"] = tolerance
        if frame_scale is not None:
            kwargs["frame_scale"] = frame_scale
        if detection_model is not None:
            kwargs["detection_model"] = detection_model

        recognition_service = RecognitionService(face_data=face_data, **kwargs)
        recognition_service.run_recognition()

    def enroll(
        self,
        raw_name: str,
        camera_index: int | None = None,
        frame_scale: float | None = None,
        detection_model: str | None = None,
    ) -> None:
        """Enroll a person from the webcam by saving one photo to known_faces."""
        stem = EncodingService.sanitize_person_name(raw_name)
        if stem is None:
            print(
                "Invalid name. Use letters, numbers, spaces, or hyphens "
                "(e.g. 'Alice' or 'Jane_Doe')."
            )
            return

        enroll_kwargs: dict = {}
        if camera_index is not None:
            enroll_kwargs["camera_index"] = camera_index
        if frame_scale is not None:
            enroll_kwargs["frame_scale"] = frame_scale
        if detection_model is not None:
            enroll_kwargs["detection_model"] = detection_model

        enrollment = EnrollmentService(self.encoding_service, **enroll_kwargs)
        enrollment.enroll_from_camera(stem)


def main() -> None:
    """
    Default CLI entry when module is run directly (legacy).
    Prefer recognition_app.py with argparse.
    """
    controller = MainController()
    controller.run(force_reload_encodings=False)


if __name__ == "__main__":
    main()
