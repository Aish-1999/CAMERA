import cv2
import mediapipe as mp

# IMPORTANT: correct initialization
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose()

cap = cv2.VideoCapture("http://192.168.4.1:81/stream")

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Cannot receive video")
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow("ESP32 Pose", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()