# Object Detection using OpenCV and YOLO in Rust

## Overview
This project implements real-time object detection using the YOLO (You Only Look Once) deep learning model with OpenCV in Rust. The system loads a pre-trained YOLOv4 model, processes video frames, and detects objects with bounding boxes and confidence scores.

## Features
- Loads YOLOv4 model weights and configuration.
- Reads class labels from a `coco.names` file.
- Processes video frames for real-time object detection.
- Applies non-maximum suppression (NMS) to reduce false positives.
- Displays detected objects with class labels and confidence scores.

## Installation
### Prerequisites
Ensure you have the following installed:
- **Rust** (latest stable version) with Cargo
- **OpenCV** (compiled with DNN module support)
- **opencv-rust** bindings

### Setting Up the Project
Clone the repository and navigate to the project directory:
```sh
 git clone https://github.com/lazycodebaker/opencv_yolo_tracking.git
 cd object-detection-rust
```

Install dependencies:
```sh
 cargo build
```

## Usage
### Prepare YOLO Model Files
Download the necessary YOLOv4 files and place them inside a `yolo` directory:
- [`yolov4.weights`](https://github.com/AlexeyAB/darknet)
- [`yolov4.cfg`](https://github.com/AlexeyAB/darknet)
- [`coco.names`](https://github.com/pjreddie/darknet/blob/master/data/coco.names)
- Video file (`0.mp4`) for testing

### Run the Object Detection
```sh
 cargo run
```
This will process the video and print detected object classes with confidence scores.

## Project Structure
```
.
├── src
│   ├── main.rs        # Main application file 
├── yolo
│   ├── yolov4.weights # YOLO model weights
│   ├── yolov4.cfg     # YOLO model configuration
│   ├── coco.names     # Class labels
│   ├── 0.mp4          # Sample video
├── Cargo.toml         # Rust dependencies and project metadata
└── README.md          # Project documentation
```

## Configuration
Modify the `YOLO_FOLDER_PATH` constant in `main.rs` to point to the correct location of the YOLO model files.

```rust
const YOLO_FOLDER_PATH: &str = "/path/to/yolo/folder";
```

## Future Enhancements
- **Real-time Webcam Support**: Implementing live detection using a webcam.
- **Improved Performance**: Optimize processing speed using GPU acceleration.
- **Bounding Box Visualization**: Draw detected objects on frames using OpenCV.

## License
This project is licensed under the MIT License. Feel free to use and modify it as needed.

## Acknowledgments
- OpenCV Team
- YOLO Developers (AlexeyAB, PJ Reddie)
- Rust OpenCV Bindings Community

# yolo_tracking
