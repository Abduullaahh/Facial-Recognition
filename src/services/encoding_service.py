"""
Service for managing face encoding operations.

This module handles loading face encodings from images,
saving them to disk, and retrieving them from cache.
"""

import os
import pickle
import re
from pathlib import Path
from typing import Any, Optional, Tuple

import face_recognition

from src.config import ENCODINGS_PICKLE_FILE, KNOWN_FACES_DIRECTORY, SUPPORTED_IMAGE_EXTENSIONS
from src.models.face_data import FaceData

# Pickle format: legacy (encodings, names) or current (encodings, names, manifest)
Manifest = dict[str, float]


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

    def _build_manifest(self, known_faces_path: Path) -> Manifest:
        """Map relative image path (posix) -> mtime for change detection."""
        manifest: Manifest = {}
        if not known_faces_path.is_dir():
            return manifest
        for image_path in known_faces_path.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                rel = image_path.relative_to(known_faces_path).as_posix()
                manifest[rel] = image_path.stat().st_mtime
        return manifest

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
        if os.path.exists(self.encodings_file) and not force_reload:
            return self._load_from_pickle()

        return self._generate_from_images()

    def _load_from_pickle(self) -> FaceData:
        """
        Load encodings from pickle file; regenerate if corrupt or known_faces changed.

        Returns:
            FaceData object containing encodings and names
        """
        print(f"Loading encodings from {self.encodings_file}...")
        try:
            with open(self.encodings_file, "rb") as f:
                data: Any = pickle.load(f)
        except (OSError, pickle.PickleError, EOFError) as e:
            print(f"Warning: could not read encodings file ({e}). Regenerating from images.")
            return self._generate_from_images()

        known_faces_path = Path(self.known_faces_dir)
        current_manifest = self._build_manifest(known_faces_path)

        if isinstance(data, tuple) and len(data) == 3:
            known_encodings, known_names, saved_manifest = data
            if not isinstance(saved_manifest, dict):
                print("Invalid manifest in pickle. Regenerating...")
                return self._generate_from_images()
            if saved_manifest == current_manifest:
                print(f"Loaded {len(known_names)} known face(s).")
                return FaceData(encodings=list(known_encodings), names=list(known_names))
            print("Known faces folder changed since last save. Regenerating encodings...")
            return self._generate_from_images()

        if isinstance(data, tuple) and len(data) == 2:
            known_encodings, known_names = data
            print(f"Loaded {len(known_names)} known face(s) (legacy pickle; no auto-sync).")
            print("Tip: re-save with current app or use --reload after changing images.")
            return FaceData(encodings=list(known_encodings), names=list(known_names))

        print("Unrecognized pickle format. Regenerating...")
        return self._generate_from_images()

    def _generate_from_images(self) -> FaceData:
        """
        Generate encodings by scanning directory for images.

        Returns:
            FaceData object containing encodings and names
        """
        print(f"Scanning {self.known_faces_dir} directory for face images...")
        known_encodings: list[list[float]] = []
        known_names: list[str] = []

        known_faces_path = Path(self.known_faces_dir)
        if not known_faces_path.exists():
            os.makedirs(known_faces_path, exist_ok=True)
            print(f"Created {self.known_faces_dir} directory. Add face images or use 'enroll'.")
            return FaceData(encodings=known_encodings, names=known_names)

        for image_path in known_faces_path.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                encoding, name = self._extract_encoding_from_image(image_path, verbose=True)
                if encoding is not None:
                    known_encodings.append(encoding)
                    known_names.append(name)

        if len(known_encodings) == 0:
            print("No face encodings found. Add images to known_faces or use 'enroll'.")
            return FaceData(encodings=known_encodings, names=known_names)

        manifest = self._build_manifest(known_faces_path)
        self._save_to_pickle(known_encodings, known_names, manifest)
        print("Face encoding process completed successfully.")

        return FaceData(encodings=known_encodings, names=known_names)

    def _extract_encoding_from_image(
        self,
        image_path: Path,
        *,
        verbose: bool = True
    ) -> Tuple[Optional[list[float]], str]:
        """
        Extract face encoding from a single image file.

        Args:
            image_path: Path to the image file
            verbose: If False, skip progress prints (e.g. enrollment verification)

        Returns:
            Tuple of (encoding or None, person_name)
        """
        person_name = image_path.stem

        try:
            image = face_recognition.load_image_file(str(image_path))
        except OSError as e:
            if verbose:
                print(f"Warning: could not load {image_path.name}: {e}")
            return None, person_name

        face_locations = face_recognition.face_locations(image)

        if len(face_locations) == 0:
            if verbose:
                print(f"Warning: No face detected in {image_path.name}")
            return None, person_name

        face_encodings = face_recognition.face_encodings(image, face_locations)

        if len(face_encodings) > 0:
            if verbose and len(face_encodings) > 1:
                print(f"Note: multiple faces in {image_path.name}; using the first.")
            if verbose:
                print(f"Loaded face encoding for: {person_name} ({image_path.name})")
            return face_encodings[0], person_name

        if verbose:
            print(f"Warning: Could not extract encoding from {image_path.name}")
        return None, person_name

    def _save_to_pickle(
        self,
        encodings: list[list[float]],
        names: list[str],
        manifest: Manifest
    ) -> None:
        """
        Save encodings, names, and file manifest to pickle file.

        Args:
            encodings: List of face encodings
            names: List of corresponding names
            manifest: Relative path -> mtime map for cache invalidation
        """
        print(f"Saving {len(names)} face encoding(s) to {self.encodings_file}...")
        with open(self.encodings_file, "wb") as f:
            pickle.dump((encodings, names, manifest), f)

    def refresh_encodings(self) -> FaceData:
        """Force rebuild from disk and rewrite pickle."""
        return self._generate_from_images()

    @staticmethod
    def sanitize_person_name(raw: str) -> Optional[str]:
        """Return a safe filename stem or None if invalid."""
        cleaned = raw.strip()
        if not cleaned:
            return None
        cleaned = re.sub(r"[^\w\s-]", "", cleaned, flags=re.UNICODE)
        cleaned = re.sub(r"[-\s]+", "_", cleaned).strip("_")
        if not cleaned or not re.match(r"^[\w]+$", cleaned):
            return None
        if len(cleaned) > 64:
            cleaned = cleaned[:64]
        return cleaned

    def next_enrollment_path(self, stem: str) -> Path:
        """Return a non-colliding path like known_faces/{stem}.jpg."""
        base = Path(self.known_faces_dir)
        base.mkdir(parents=True, exist_ok=True)
        candidate = base / f"{stem}.jpg"
        if not candidate.exists():
            return candidate
        i = 1
        while True:
            candidate = base / f"{stem}_{i}.jpg"
            if not candidate.exists():
                return candidate
            i += 1
