"""
Demo Data Seeder for DeepFaceAttend
===================================
Enrolls demo students with generated face portraits so the system can be
evaluated and demonstrated immediately out-of-the-box.
"""

import os
import sys

# Ensure root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from core.attendance_engine import AttendancePipeline


def create_synthetic_face(skin_tone=(180, 200, 230), eye_color=(50, 40, 30), hair_color=(30, 30, 30)):
    """Generates a clean synthetic face portrait to serve as demo enrollment."""
    img = np.ones((250, 250, 3), dtype=np.uint8) * 240

    # Face Oval
    cv2.ellipse(img, (125, 135), (70, 95), 0, 0, 360, skin_tone, -1)

    # Hair
    cv2.ellipse(img, (125, 65), (75, 45), 0, 0, 360, hair_color, -1)

    # Eyes
    cv2.circle(img, (95, 115), 10, (255, 255, 255), -1)
    cv2.circle(img, (155, 115), 10, (255, 255, 255), -1)
    cv2.circle(img, (95, 115), 5, eye_color, -1)
    cv2.circle(img, (155, 115), 5, eye_color, -1)

    # Eyebrows
    cv2.line(img, (80, 100), (110, 100), hair_color, 3)
    cv2.line(img, (140, 100), (170, 100), hair_color, 3)

    # Nose
    pts = np.array([[125, 125], [120, 150], [130, 150]], np.int32)
    cv2.polylines(img, [pts], False, (140, 160, 190), 2)

    # Mouth / Smile
    cv2.ellipse(img, (125, 175), (25, 12), 0, 0, 180, (100, 100, 190), 3)

    return img


def seed_demo_students():
    print("Seeding demo students into DeepFaceAttend database...")
    pipeline = AttendancePipeline()

    demo_profiles = [
        {
            "roll": "21CS001",
            "name": "Aarav Sharma",
            "dept": "Computer Science & Engineering",
            "skin": (170, 195, 220),
            "hair": (20, 20, 20)
        },
        {
            "roll": "21AI015",
            "name": "Priya Patel",
            "dept": "Artificial Intelligence & Data Science",
            "skin": (185, 210, 235),
            "hair": (40, 30, 20)
        },
        {
            "roll": "21IT028",
            "name": "Rohan Verma",
            "dept": "Information Technology",
            "skin": (160, 185, 210),
            "hair": (15, 15, 15)
        }
    ]

    os.makedirs("data/registered_faces", exist_ok=True)

    for p in demo_profiles:
        face_img = create_synthetic_face(skin_tone=p["skin"], hair_color=p["hair"])
        photo_path = f"data/registered_faces/{p['roll']}.jpg"
        cv2.imwrite(photo_path, face_img)

        # Extract 512-D embedding
        emb = pipeline.extract_embedding(face_img)

        # Register
        ok, msg = pipeline.db.register_student(
            roll_number=p["roll"],
            name=p["name"],
            department=p["dept"],
            embedding=emb,
            photo_path=photo_path
        )
        print(f"[{'OK' if ok else 'FAIL'}] {msg}")

    pipeline.refresh_gallery()
    print(f"\nSeeding complete! Total enrolled students: {len(pipeline.gallery)}")


if __name__ == "__main__":
    seed_demo_students()
