"""
Standalone Desktop Real-Time Webcam Face Attendance Scanner
===========================================================
High-performance direct OpenCV webcam scanner for offline classroom/lab deployment.
Runs at 30+ FPS with live bounding boxes, recognition, and attendance logging.

Usage:
    python run_webcam.py
    python run_webcam.py --camera 0 --threshold 0.60
"""

import argparse
import time
import cv2
import numpy as np
from core.attendance_engine import AttendancePipeline


def run_scanner(camera_idx: int = 0, threshold: float = 0.60):
    print("=" * 65)
    print(" DeepFaceAttend: Desktop Real-Time Attendance Scanner")
    print("=" * 65)
    print(f"Connecting to Camera Index {camera_idx}...")

    pipeline = AttendancePipeline(similarity_threshold=threshold)
    print(f"Enrolled students in database: {len(pipeline.gallery)}")

    cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[Warning] Failed to open camera with CAP_DSHOW, retrying default backend...")
        cap = cv2.VideoCapture(camera_idx)

    if not cap.isOpened():
        print(f"[Error] Could not open video capture device {camera_idx}.")
        print("Please check your camera connection or use the Streamlit web dashboard.")
        return

    # Set camera resolution to 640x480 for smooth 30+ FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window_name = "DeepFaceAttend - Real-Time Scanner [Press 'q' to exit]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    prev_time = time.time()
    fps = 0.0
    frame_count = 0

    print("\nScanner Controls:")
    print("  'q' - Exit application")
    print("  's' - Save snapshot of current frame")
    print("  'r' - Reload registered student database")
    print("-" * 65)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[Error] Failed to read frame from camera.")
                break

            frame_count += 1
            curr_time = time.time()
            dt = curr_time - prev_time
            if dt >= 1.0:
                fps = frame_count / dt
                frame_count = 0
                prev_time = curr_time

            # Mirror frame horizontally for intuitive mirror feedback
            flipped = cv2.flip(frame, 1)

            # Process frame through Deep Learning Pipeline
            annotated, events = pipeline.process_frame(flipped, mark_attendance=True, threshold=threshold)

            # Draw HUD Header
            hud_bar = np.zeros((45, annotated.shape[1], 3), dtype=np.uint8)
            hud_bar[:] = (30, 30, 30)

            status_text = f"FPS: {fps:.1f} | Enrolled: {len(pipeline.gallery)} | Faces Detected: {len(events)}"
            cv2.putText(hud_bar, status_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 1, cv2.LINE_AA)
            cv2.putText(hud_bar, "DeepFaceAttend", (annotated.shape[1] - 180, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            display = np.vstack([hud_bar, annotated])

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[Exit] User terminated scanner session.")
                break
            elif key == ord('s'):
                snap_name = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(snap_name, annotated)
                print(f"[Snapshot] Saved current frame to {snap_name}")
            elif key == ord('r'):
                pipeline.refresh_gallery()
                print(f"[Database] Reloaded gallery. Total enrolled: {len(pipeline.gallery)}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepFaceAttend Desktop Scanner")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--threshold", type=float, default=0.60, help="Cosine similarity match threshold (default: 0.60)")
    args = parser.parse_args()

    run_scanner(camera_idx=args.camera, threshold=args.threshold)
