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

## How to use

### First-time setup

1. Use **Python 3.10+** and a working webcam.
2. Open a terminal in the project folder (`Facial-Recognition`) and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. If **`dlib`** fails to install on Windows, install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) or use the **conda** snippet in [Installation](#installation) above.

Paths like `known_faces/` and `encodings.pkl` are resolved from the **project root**, so you can run commands from another working directory as long as this project is the one you installed.

You need **at least one enrolled face** before live recognition can label anyone. Use either photos on disk or the webcam enroll flow below.

### Add people (enrollment)

#### Option A — Photos in `known_faces/`

1. Copy images into the **`known_faces`** folder (next to `recognition_app.py`).
2. The **filename (without extension)** is the on-screen label, e.g. `Alice.jpg` → **Alice**.
3. Prefer clear, front-facing, well-lit shots; **one main face per image** works best (if several faces are detected, the first is used).

#### Option B — Capture from the webcam

```bash
python recognition_app.py enroll Alice
```

Replace `Alice` with a safe label: letters, numbers, spaces, or hyphens (e.g. `Jane_Doe`). Then:

1. Show **one** face in the frame.
2. When the hint says you are clear to save, press **Space** to capture.
3. Press **`q`** to cancel without saving.

The app checks that the saved image contains a usable face; otherwise the file is removed and you can try again. Encodings are refreshed after a successful save.

### Run live recognition

```bash
python recognition_app.py
```

- A video window opens using the default camera.
- **Green** box + name: matched **known** person.
- **Red** box + **Unknown**: face not matched (or below tolerance).
- Press **`q`** to quit.

### Common tasks (CLI)

| Goal | Example |
|------|---------|
| Use a different camera | `python recognition_app.py -c 1` (try `0`, `1`, `2`, …) |
| Stricter matching | `python recognition_app.py -t 0.5` (lower = stricter) |
| Faster processing (smaller detection image) | `python recognition_app.py --scale 0.2` |
| More accurate detection, slower on CPU | `python recognition_app.py --model cnn` |
| Rebuild `encodings.pkl` after manual file changes | `python recognition_app.py --reload` |

For every flag:

```bash
python recognition_app.py --help
python recognition_app.py enroll --help
```

### Typical workflows

- **First run:** Enroll with `python recognition_app.py enroll <name>` *or* add images under `known_faces/`, then start recognition with `python recognition_app.py`.
- **After adding or removing images in `known_faces/`:** The next start usually reloads encodings automatically (manifest). If labels look wrong, run once with **`--reload`**.

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

- **“No known faces loaded”** — Nothing usable was found in `known_faces/` (empty folder, bad images, or no face detected). Add photos or run `enroll` again.
- **No faces in images**: Use front-facing, well-lit photos; check supported extensions.
- **Camera not opening**: Close other apps using the camera; try `-c 1` or `-c 2`.
- **Poor accuracy**: Add more or clearer `known_faces` images; lower `-t` slightly; try `--model cnn` if CPU allows.
- **Legacy `encodings.pkl`**: After changing images, run once with `--reload` if encodings seem stale.

## License

Provided as-is for educational and development use.
