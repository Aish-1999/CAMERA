from flask import Flask, Response
import cv2
import mediapipe as mp
import math
from collections import deque

app = Flask(__name__)

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    enable_segmentation=False
)

# ESP32 stream URL
STREAM_URL = "http://192.168.4.1:81/stream"
cap = cv2.VideoCapture(STREAM_URL)

# Smoothing history
history = deque(maxlen=10)

def get_stable_status(new_status):
    history.append(new_status)
    return max(set(history), key=history.count)

# Angle calculation
def calculate_angle(a, b, c):
    angle = math.degrees(
        math.atan2(c.y - b.y, c.x - b.x) -
        math.atan2(a.y - b.y, a.x - b.x)
    )
    return abs(angle)

def generate_frames():
    global cap

    while True:
        success, frame = cap.read()

        if not success:
            print("⚠️ Reconnecting to ESP32...")
            cap.release()
            cap = cv2.VideoCapture(STREAM_URL)
            continue

        try:
            # Resize for performance
            frame = cv2.resize(frame, (480, 320))

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            status = "NO PERSON"

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                # Key points
                hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
                knee = lm[mp_pose.PoseLandmark.LEFT_KNEE]
                ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
                wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST]
                shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]

                # Calculate knee angle
                angle = calculate_angle(hip, knee, ankle)

                # Better posture logic
                if wrist.y < shoulder.y:
                    status = "HAND RAISED"
                elif angle < 100:
                    status = "SITTING"
                else:
                    status = "STANDING"

                # Apply smoothing
                status = get_stable_status(status)

                # Draw skeleton
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

            # Color based on status
            if status == "STANDING":
                color = (0, 255, 0)
            elif status == "SITTING":
                color = (0, 0, 255)
            else:
                color = (255, 255, 0)

            # Show text
            cv2.putText(frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Encode frame
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            print("Error:", e)
            continue


@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>ESP32 Pose AI</title>
    </head>
    <body style="background:black; color:white; text-align:center;">
        <h2>AI Posture Detection</h2>
        <img src="/video" width="400" style="border:3px solid white;">
        <p style="color:gray;">Green = Standing | Red = Sitting | Yellow = Hand Raised</p>
    </body>
    </html>
    """


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)