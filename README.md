# Real-Time Facial Recognition System

A modular Python application for real-time facial recognition using a webcam, built with OpenCV and the `face_recognition` library.

## Features

- **Real-time recognition**: Live video with labeled bounding boxes (green = known, red = unknown)
- **Multiple faces**: Detects and labels several people in one frame
- **Persistent encodings**: Caches 128-D encodings in `encodings.pkl` and **invalidates** when files under `known_faces/` change
- **Webcam enrollment**: Add a person without copying files manually (`enroll` command)
- **Performance**: Optional downscaled detection, FPS overlay, HOG or CNN face detection

## Requirements

- Python 3.10+
- Webcam access
- Dependencies: see `requirements.txt` (`opencv-python`, `face-recognition`, `dlib`)

## Installation

```bash
pip install -r requirements.txt
```

**dlib / `face_recognition` on Windows** often needs [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/). If pip fails, try:

```bash
conda install -c conda-forge dlib
pip install opencv-python face-recognition
```

## Quick start

1. **Add people** — either:
   - Put photos in `known_faces/` (filename stem = display name, e.g. `alice.jpg` → `alice`), or  
   - Run enrollment (see below).

2. **Run recognition**

   ```bash
   python recognition_app.py
   ```

3. Press **`q`** to quit the video window.

Paths (`known_faces/`, `encodings.pkl`) are resolved from the **project root**, so you can run the app from another working directory if you use the same install.

## Command-line interface

```text
python recognition_app.py [options]              # live recognition (default)
python recognition_app.py enroll <name> [options] # capture one photo from webcam
```

### Global options (recognition and enroll where applicable)

| Option | Description |
|--------|-------------|
| `-c`, `--camera N` | Camera index (default: `0`) |
| `-t`, `--tolerance` | Match tolerance; **lower** = stricter (default: `0.6`) |
| `--scale` | Detection resize factor `0.1`–`1.0` (default: `0.25`; smaller = faster) |
| `--model` | `hog` (fast, CPU) or `cnn` (slower, often better) |
| `-r`, `--reload` | Rebuild encodings from disk (recognition only) |

### Examples

```bash
# Live recognition, second camera, stricter matching
python recognition_app.py -c 1 -t 0.5

# Force full rebuild of encodings.pkl from known_faces/
python recognition_app.py --reload

# Enroll "Jane_Doe" from the default camera
python recognition_app.py enroll Jane_Doe
```

During **enroll**: show **one** face, press **Space** to save; press **`q`** to cancel. The image is checked for a usable face; if not, it is discarded.

## Known faces folder

Place images under `known_faces/` (supported: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`). Subfolders are scanned. Prefer one clear face per image; if several faces appear, the first encoding is used.

The repo includes an empty `known_faces/` (via `.gitkeep`) so the path exists after clone.

## Project structure

```text
Facial-Recognition/
├── recognition_app.py          # CLI entry
├── requirements.txt
├── encodings.pkl               # Generated (gitignored)
├── known_faces/                # Your enrollment images
├── src/
│   ├── config/settings.py      # Defaults and paths
│   ├── models/face_data.py
│   ├── utils/                  # FPS, drawing
│   ├── services/
│   │   ├── encoding_service.py
│   │   ├── enrollment_service.py
│   │   └── recognition_service.py
│   └── controllers/main_controller.py
└── README.md
```

## How it works

1. **Encoding** (`EncodingService`): Reads images from `known_faces/`, computes encodings, writes `encodings.pkl` with a **manifest** (file paths + modification times). On the next run, if the folder no longer matches the manifest, encodings are rebuilt automatically. Older pickles without a manifest still load; use `--reload` after bulk changes if needed.

2. **Recognition** (`RecognitionService`): Loads encodings, captures video, runs detection (optionally on a downscaled frame for speed), compares encodings, draws boxes and names.

## Configuration (`src/config/settings.py`)

| Setting | Role |
|---------|------|
| `DEFAULT_CAMERA_INDEX` | Default camera |
| `DEFAULT_TOLERANCE` | Default match tolerance |
| `FRAME_PROCESS_SCALE` | Default scale for **detection** only |
| `FACE_DETECTION_MODEL` | `hog` or `cnn` default |
| Paths | `KNOWN_FACES_DIRECTORY`, `ENCODINGS_PICKLE_FILE` (absolute under project root) |

Override behavior from the CLI without editing code (`--camera`, `--tolerance`, `--scale`, `--model`).

## Troubleshooting

- **No faces in images**: Use front-facing, well-lit photos; check supported extensions.
- **Camera not opening**: Close other apps using the camera; try `-c 1` or `-c 2`.
- **Poor accuracy**: Add more or clearer `known_faces` images; lower `-t` slightly; try `--model cnn` if CPU allows.
- **Legacy `encodings.pkl`**: After changing images, run once with `--reload` if encodings seem stale.

## License

Provided as-is for educational and development use.
