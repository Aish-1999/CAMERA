from flask import Flask, Response
import cv2
import mediapipe as mp

app = Flask(__name__)

# Load MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,  # faster
    enable_segmentation=False
)

# ESP32 stream
STREAM_URL = "http://192.168.4.1:81/stream"
cap = cv2.VideoCapture(STREAM_URL)


def generate_frames():
    global cap

    while True:
        success, frame = cap.read()

        # 🔁 Auto-reconnect if stream breaks
        if not success:
            print("⚠️ Reconnecting to ESP32 stream...")
            cap.release()
            cap = cv2.VideoCapture(STREAM_URL)
            continue

        try:
            # Resize for performance
            frame = cv2.resize(frame, (480, 320))

            # Convert to RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run pose detection
            results = pose.process(img_rgb)

            status = "NO PERSON"

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                # Key joints
                wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST].y
                shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y
                hip = lm[mp_pose.PoseLandmark.LEFT_HIP].y
                knee = lm[mp_pose.PoseLandmark.LEFT_KNEE].y

                # Simple posture logic
                if wrist < shoulder:
                    status = "HAND RAISED"
                elif abs(hip - knee) < 0.08:
                    status = "SITTING"
                else:
                    status = "STANDING"

                # Draw skeleton
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

            # Show status text
            cv2.putText(frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Encode frame
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Stream to browser
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
        <h2>ESP32 Pose Detection</h2>
        <img src="/video" width="400" style="border:3px solid white;">
        <p>Shows: Standing / Sitting / Hand Raised</p>
    </body>
    </html>
    """


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)