# Real-Time Facial Recognition System

A complete, modular Python application for real-time facial recognition using a webcam feed.

## Features

- 🎥 **Real-time Recognition**: Live video processing with webcam support
- 👥 **Multiple Face Detection**: Identifies multiple individuals in a single frame
- 💾 **Persistent Encodings**: Saves face encodings to avoid recalculation
- 📊 **Performance Tracking**: Displays FPS counter for performance monitoring
- 🎨 **Visual Feedback**: Color-coded bounding boxes (green for known, red for unknown)

## Requirements

- Python 3.10 or higher
- Webcam/camera access
- OpenCV and face_recognition libraries (see requirements.txt)

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

**Note**: Installing `dlib` (dependency of `face_recognition`) may require additional system dependencies:
- **Windows**: Visual C++ Build Tools may be needed
- **Linux**: `cmake` and `libboost` may be required
- **macOS**: Usually installs without issues via pip

If you encounter issues with `dlib` installation, consider using a pre-built wheel or conda:

```bash
# Using conda (recommended for easier dlib installation)
conda install -c conda-forge dlib
pip install opencv-python face-recognition
```

## Usage

### 1. Prepare Known Faces

Create a `known_faces/` directory in the project root and add images of people you want to recognize:

```
known_faces/
├── alice.jpg
├── bob.jpg
├── charlie.png
└── ...
```

**Important**: 
- Use clear, front-facing photos for best results
- File names (without extension) will be used as person names
- Only one face per image is recommended (first face found will be used)

### 2. Run the Application

```bash
python recognition_app.py
```

The application will:
1. Scan the `known_faces/` directory (first run only, or if `encodings.pkl` doesn't exist)
2. Extract face encodings and save them to `encodings.pkl`
3. Start the webcam feed
4. Display real-time recognition results

### 3. Controls

- Press **'q'** to quit the application

## Project Structure

```
Facial Recognition/
├── recognition_app.py      # Main entry point
├── requirements.txt        # Python dependencies
├── encodings.pkl          # Generated face encodings (auto-created)
├── known_faces/           # Directory for known face images (create this)
├── src/                   # Source code directory
│   ├── config/           # Configuration settings
│   │   ├── __init__.py
│   │   └── settings.py   # App configuration constants
│   ├── models/           # Data models
│   │   ├── __init__.py
│   │   └── face_data.py  # FaceData dataclass
│   ├── utils/            # Utility functions
│   │   ├── __init__.py
│   │   ├── visualization.py  # Drawing functions
│   │   └── fps_calculator.py # FPS tracking utility
│   ├── services/         # Business logic
│   │   ├── __init__.py
│   │   ├── encoding_service.py   # Face encoding management
│   │   └── recognition_service.py # Real-time recognition
│   └── controllers/      # Entry points
│       ├── __init__.py
│       └── main_controller.py # Main application controller
└── README.md             # This file
```

## Architecture

The application follows a modular architecture with clear separation of concerns:

- **`src/config/`**: Centralized configuration settings
- **`src/models/`**: Data models (FaceData dataclass)
- **`src/utils/`**: Reusable utility functions (visualization, FPS calculation)
- **`src/services/`**: Core business logic (encoding and recognition services)
- **`src/controllers/`**: Application entry points and orchestration

## How It Works

1. **Enrollment Phase** (`EncodingService.load_known_faces()`):
   - Scans `known_faces/` directory for images
   - Extracts 128-dimensional face encodings using the `face_recognition` library
   - Saves encodings and names to `encodings.pkl` for future use

2. **Recognition Phase** (`RecognitionService.run_recognition()`):
   - Loads encodings from `encodings.pkl`
   - Captures frames from webcam
   - Detects faces in each frame
   - Compares detected faces with known encodings
   - Displays results with colored bounding boxes and labels

## Configuration

Configuration settings can be modified in `src/config/settings.py`:

- **Camera Index**: `DEFAULT_CAMERA_INDEX` (default: 0)
- **Recognition Tolerance**: `DEFAULT_TOLERANCE` (lower = stricter, default: 0.6)
- **Known Faces Directory**: `KNOWN_FACES_DIRECTORY` (default: "known_faces")
- **Encodings File**: `ENCODINGS_PICKLE_FILE` (default: "encodings.pkl")

You can also customize settings programmatically when initializing services in `src/controllers/main_controller.py`.

## Troubleshooting

### No faces detected in images
- Ensure images contain clear, front-facing faces
- Check that images are in supported formats (JPG, PNG, BMP, GIF)

### Camera not working
- Verify camera permissions
- Try changing `camera_index` (0, 1, 2, etc.)
- Check if another application is using the camera

### Recognition accuracy issues
- Use high-quality, well-lit images for enrollment
- Adjust the `tolerance` parameter in `run_recognition()`
- Ensure faces are clearly visible in the webcam feed

## License

This project is provided as-is for educational and development purposes.

