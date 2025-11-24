"""
Main controller for the Facial Recognition System.

This module serves as the entry point and orchestrates
the application's main workflow.
"""

from src.config import ENCODINGS_PICKLE_FILE, KNOWN_FACES_DIRECTORY
from src.services.encoding_service import EncodingService
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

    def run(self, force_reload_encodings: bool = False) -> None:
        """
        Run the facial recognition application.

        Loads known faces and starts real-time recognition.

        Args:
            force_reload_encodings: If True, regenerate encodings even if pickle exists
        """
        # Load known face encodings
        face_data = self.encoding_service.load_known_faces(force_reload=force_reload_encodings)

        # Initialize and run recognition service
        recognition_service = RecognitionService(face_data=face_data)
        recognition_service.run_recognition()


def main() -> None:
    """
    Main entry point for the facial recognition application.

    Creates the main controller and runs the application.
    """
    controller = MainController()
    controller.run(force_reload_encodings=False)


if __name__ == "__main__":
    main()

