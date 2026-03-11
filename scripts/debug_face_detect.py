"""Debug face detection on live camera frame.

Usage: python scripts/debug_face_detect.py [camera_index]
"""
import sys

sys.path.insert(0, "src")

import cv2
from glados.robot.face_id import FaceRecognizer

camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

print("Loading FaceRecognizer...")
rec = FaceRecognizer("models/Face", face_db_dir="faces/")
print(f"Face DB: {rec.db.names}")

cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print(f"ERROR: Camera {camera_index} not available")
    sys.exit(1)

ret, frame = cap.read()
cap.release()

if not ret or frame is None:
    print("ERROR: Failed to capture frame")
    sys.exit(1)

print(f"Frame: {frame.shape} (H×W×C)")

# Test detection with different thresholds
saved_thresh = rec._SCORE_THRESH
for thresh in [0.3, 0.4, 0.5, 0.6, 0.65]:
    rec._SCORE_THRESH = thresh
    faces = rec.detect(frame)
    print(f"  threshold={thresh:.2f} → {len(faces)} faces: {faces}")
rec._SCORE_THRESH = saved_thresh

# Test full recognition pipeline
results = rec.recognize(frame)
print(f"\nRecognition results: {results}")

# Save frame for manual inspection
cv2.imwrite("/tmp/debug_frame.jpg", frame)
print("\nSaved /tmp/debug_frame.jpg")
