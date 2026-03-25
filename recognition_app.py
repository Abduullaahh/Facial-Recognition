"""
Facial recognition CLI: run live recognition or enroll from webcam.
"""

import argparse
import sys

from src.config import (
    DEFAULT_CAMERA_INDEX,
    DEFAULT_TOLERANCE,
    FACE_DETECTION_MODEL,
    FRAME_PROCESS_SCALE,
)
from src.controllers.main_controller import MainController


def main() -> None:
    camera_parent = argparse.ArgumentParser(add_help=False)
    camera_parent.add_argument(
        "--camera",
        "-c",
        type=int,
        default=None,
        metavar="N",
        help=f"Camera index (default: {DEFAULT_CAMERA_INDEX})",
    )
    camera_parent.add_argument(
        "--scale",
        type=float,
        default=None,
        help=(
            "Frame scale for detection only, 0.1-1.0 "
            f"(default: {FRAME_PROCESS_SCALE}; smaller is faster)"
        ),
    )
    camera_parent.add_argument(
        "--model",
        choices=("hog", "cnn"),
        default=None,
        help=f"Detection model (default: {FACE_DETECTION_MODEL})",
    )

    parser = argparse.ArgumentParser(
        parents=[camera_parent],
        description="Real-time face recognition and webcam enrollment.",
    )
    parser.add_argument(
        "--tolerance",
        "-t",
        type=float,
        default=None,
        help=f"Recognition tolerance; lower = stricter (default: {DEFAULT_TOLERANCE})",
    )
    parser.add_argument(
        "--reload",
        "-r",
        action="store_true",
        help="Rebuild encodings from known_faces even if encodings.pkl exists",
    )

    sub = parser.add_subparsers(dest="action", metavar="COMMAND")
    enroll_p = sub.add_parser(
        "enroll",
        parents=[camera_parent],
        help="Save one webcam photo into known_faces and refresh encodings",
    )
    enroll_p.add_argument("name", help="Person label (filename stem, e.g. Alice or Jane_Doe)")

    args = parser.parse_args()
    if args.scale is not None and not (0.1 <= args.scale <= 1.0):
        print("Error: --scale must be between 0.1 and 1.0.", file=sys.stderr)
        sys.exit(1)

    controller = MainController()

    if args.action == "enroll":
        controller.enroll(
            args.name,
            camera_index=args.camera,
            frame_scale=args.scale,
            detection_model=args.model,
        )
        return

    controller.run(
        force_reload_encodings=args.reload,
        camera_index=args.camera,
        tolerance=args.tolerance,
        frame_scale=args.scale,
        detection_model=args.model,
    )


if __name__ == "__main__":
    main()
