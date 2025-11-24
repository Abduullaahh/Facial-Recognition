"""
Service for managing face encoding operations.

This module handles loading face encodings from images,
saving them to disk, and retrieving them from cache.
"""

import os
import pickle
from pathlib import Path
from typing import Optional, Tuple

import face_recognition

from src.config import ENCODINGS_PICKLE_FILE, KNOWN_FACES_DIRECTORY, SUPPORTED_IMAGE_EXTENSIONS
from src.models.face_data import FaceData


class EncodingService:
    """
    Service for managing face encoding operations.

    Handles scanning directories for face images, extracting
    encodings, and persisting them to disk for efficient reuse.
    """

    def __init__(
        self,
        known_faces_dir: str = KNOWN_FACES_DIRECTORY,
        encodings_file: str = ENCODINGS_PICKLE_FILE
    ) -> None:
        """
        Initialize the encoding service.

        Args:
            known_faces_dir: Path to directory containing known face images
            encodings_file: Path to pickle file storing encodings
        """
        self.known_faces_dir = known_faces_dir
        self.encodings_file = encodings_file

    def load_known_faces(self, force_reload: bool = False) -> FaceData:
        """
        Load face encodings from known_faces directory or load from pickle file.

        Scans the known_faces directory for images, extracts 128D face encodings,
        and saves them to a pickle file for future use.

        Args:
            force_reload: If True, regenerate encodings even if pickle exists

        Returns:
            FaceData object containing encodings and names
        """
        # Check if encodings file exists and force_reload is False
        if os.path.exists(self.encodings_file) and not force_reload:
            return self._load_from_pickle()

        # Generate encodings from images
        return self._generate_from_images()

    def _load_from_pickle(self) -> FaceData:
        """
        Load encodings from pickle file.

        Returns:
            FaceData object containing encodings and names
        """
        print(f"Loading encodings from {self.encodings_file}...")
        with open(self.encodings_file, "rb") as f:
            known_encodings, known_names = pickle.load(f)
        print(f"Loaded {len(known_names)} known face(s).")
        return FaceData(encodings=known_encodings, names=known_names)

    def _generate_from_images(self) -> FaceData:
        """
        Generate encodings by scanning directory for images.

        Returns:
            FaceData object containing encodings and names
        """
        print(f"Scanning {self.known_faces_dir} directory for face images...")
        known_encodings: list[list[float]] = []
        known_names: list[str] = []

        # Create known_faces directory if it doesn't exist
        known_faces_path = Path(self.known_faces_dir)
        if not known_faces_path.exists():
            os.makedirs(known_faces_path, exist_ok=True)
            print(f"Created {self.known_faces_dir} directory. Please add face images there.")
            return FaceData(encodings=known_encodings, names=known_names)

        # Iterate through all files in the directory
        for image_path in known_faces_path.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                encoding, name = self._extract_encoding_from_image(image_path)
                if encoding is not None:
                    known_encodings.append(encoding)
                    known_names.append(name)

        if len(known_encodings) == 0:
            print("No face encodings found. Please add images to the known_faces directory.")
            return FaceData(encodings=known_encodings, names=known_names)

        # Save encodings to pickle file for future use
        self._save_to_pickle(known_encodings, known_names)
        print("Face encoding process completed successfully.")

        return FaceData(encodings=known_encodings, names=known_names)

    def _extract_encoding_from_image(
        self,
        image_path: Path
    ) -> Tuple[Optional[list[float]], str]:
        """
        Extract face encoding from a single image file.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (encoding or None, person_name)
        """
        # Extract person name from filename (without extension)
        person_name = image_path.stem

        # Load the image using face_recognition library
        image = face_recognition.load_image_file(str(image_path))

        # Find face locations in the image
        face_locations = face_recognition.face_locations(image)

        if len(face_locations) == 0:
            print(f"Warning: No face detected in {image_path.name}")
            return None, person_name

        # Extract face encodings (128-dimensional vectors)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        # If multiple faces in one image, use the first one
        if len(face_encodings) > 0:
            print(f"Loaded face encoding for: {person_name} ({image_path.name})")
            return face_encodings[0], person_name
        else:
            print(f"Warning: Could not extract encoding from {image_path.name}")
            return None, person_name

    def _save_to_pickle(
        self,
        encodings: list[list[float]],
        names: list[str]
    ) -> None:
        """
        Save encodings and names to pickle file.

        Args:
            encodings: List of face encodings
            names: List of corresponding names
        """
        print(f"Saving {len(names)} face encoding(s) to {self.encodings_file}...")
        with open(self.encodings_file, "wb") as f:
            pickle.dump((encodings, names), f)

