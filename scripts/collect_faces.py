#!/usr/bin/env python3
"""Collect face photos from camera for FaceID training.

Usage:
    python scripts/collect_faces.py <name> [--count 15] [--interval 0.5]

Opens the camera and saves face crops when SPACE is pressed,
or automatically at the given interval. Shows a live preview
with face detection boxes.

Photos are saved to faces/<name>/photo_NN.jpg
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect face photos for FaceID")
    parser.add_argument("name", help="Person name (folder name in faces/)")
    parser.add_argument("--count", type=int, default=15, help="Target number of photos")
    parser.add_argument("--interval", type=float, default=0.0,
                        help="Auto-capture interval in seconds (0 = manual with SPACE)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    out_dir = Path("faces") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Count existing photos
    existing = sorted(out_dir.glob("*.jpg"))
    start_idx = len(existing) + 1

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}")
        sys.exit(1)

    # OpenCV face detector (fast, good enough for capture)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    print(f"Collecting faces for '{args.name}' → {out_dir}/")
    print(f"Target: {args.count} photos (already have {len(existing)})")
    if args.interval > 0:
        print(f"Auto-capture every {args.interval}s")
    else:
        print("Press SPACE to capture, Q to quit")
    print("Tips: vary angle, lighting, expression, with/without glasses")

    saved = 0
    last_capture = 0.0
    idx = start_idx

    while saved < args.count:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        display = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        status = f"Saved: {saved}/{args.count} | Faces: {len(faces)}"
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Collect Faces", display)

        key = cv2.waitKey(1) & 0xFF

        should_capture = False
        if key == ord(" "):
            should_capture = True
        elif key == ord("q"):
            break
        elif args.interval > 0 and time.time() - last_capture >= args.interval:
            should_capture = True

        if should_capture and len(faces) > 0:
            # Save full frame (FaceID will detect and crop itself)
            path = out_dir / f"photo_{idx}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"  Saved {path.name}")
            saved += 1
            idx += 1
            last_capture = time.time()

    cap.release()
    cv2.destroyAllWindows()

    total = len(list(out_dir.glob("*.jpg")))
    print(f"\nDone! {total} photos in {out_dir}/")
    print("Restart the robot to re-index faces.")


if __name__ == "__main__":
    main()
