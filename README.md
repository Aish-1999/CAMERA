## Overview

This project implements a real-time posture detection system by integrating an ESP32-CAM with computer vision and machine learning.

The ESP32 streams live video over WiFi, which is processed in Python using MediaPipe for human pose estimation. The detected body landmarks are analyzed to classify posture (e.g., standing, sitting, hand raised), and the results are displayed through a web interface built with Flask.

The system demonstrates a complete pipeline combining embedded systems, AI-based vision, and web technologies for real-time human activity recognition.

---

### Key Highlights

* Real-time video streaming from ESP32
* AI-based pose detection using MediaPipe
* Posture classification using geometric analysis
* Smooth and stable predictions
* Web-based interface for live monitoring

---

### System Pipeline

ESP32 Camera → Python (OpenCV) → MediaPipe (Pose Detection) → Posture Logic → Flask Web UI
