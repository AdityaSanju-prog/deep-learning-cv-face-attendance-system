"""
Attendance Engine & Deep CV Pipeline Orchestrator
================================================
Combines:
- Unit I: Image preprocessing, aspect resizing, tensor normalization
- Unit III: Transfer learning MobileNetV2 embedding extraction & cosine matching
- Unit IV: Deep Single-Stage SSD Face Detection with NMS
- Unit VI: Passive anti-spoofing liveness check & Grad-CAM explainability
- Database: SQLite persistent enrollment & attendance logging.
"""

from typing import Tuple, List, Dict, Any, Optional
import os
import cv2
import numpy as np
import torch

from modules.unit1_foundations import ImagePreprocessor
from modules.unit3_transfer_learning import TransferFaceEmbeddingNet, FaceBiometricMatcher
from modules.unit4_object_detection import UniversalFaceDetector, BoundingBox
from modules.unit6_explainability_generative import AntiSpoofingLivenessDetector, GradCAM
from core.database import AttendanceDatabase


class AttendancePipeline:
    """
    End-to-end Computer Vision & Deep Learning Attendance Orchestrator.
    Handles face registration, real-time multi-face scanning, spoof rejection,
    and automatic database logging.
    """

    def __init__(self, db_path: str = "data/attendance.db",
                 similarity_threshold: float = 0.60,
                 device: str = "cpu"):
        self.device = torch.device(device)
        self.similarity_threshold = similarity_threshold

        # Initialize Sub-modules
        self.detector = UniversalFaceDetector(ssd_conf=0.5)
        self.preprocessor = ImagePreprocessor()
        self.liveness_detector = AntiSpoofingLivenessDetector(blur_threshold=50.0)
        self.matcher = FaceBiometricMatcher()
        self.db = AttendanceDatabase(db_path)

        # Initialize Deep Feature Extractor (Unit III)
        self.embedding_net = TransferFaceEmbeddingNet(
            backbone_name="mobilenet_v2",
            embedding_dim=512,
            pretrained=True,
            freeze_backbone=True
        ).to(self.device)
        self.embedding_net.eval()

        # Initialize Grad-CAM Explainability (Unit VI)
        self.grad_cam = GradCAM(self.embedding_net, self.embedding_net.features[-1])

        # Cache gallery embeddings in memory
        self.gallery = self.db.get_gallery_embeddings()

    def refresh_gallery(self):
        """Reloads gallery embeddings from the database."""
        self.gallery = self.db.get_gallery_embeddings()

    def extract_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Extracts 512-D L2-normalized embedding vector from a cropped face image.
        """
        # Preprocessing: BGR -> RGB -> Letterbox (224, 224) -> ImageNet Normalize
        rgb = self.preprocessor.bgr_to_rgb(face_bgr)
        letterboxed, _, _ = self.preprocessor.letterbox_resize(rgb, (224, 224))
        tensor = self.preprocessor.normalize_tensor(letterboxed, mode="imagenet")
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.embedding_net(tensor, return_normalized=True)
        return emb.cpu().numpy().flatten()

    def register_new_student(self, image_bgr: np.ndarray, roll_number: str,
                             name: str, department: str) -> Tuple[bool, str, Optional[np.ndarray]]:
        """
        Enrolls a new student from an uploaded photo or webcam snapshot.

        Returns:
            (success, message, face_crop)
        """
        if image_bgr is None or image_bgr.size == 0:
            return False, "Invalid image provided.", None

        # Detect face
        boxes = self.detector.detect_faces(image_bgr)
        if len(boxes) == 0:
            return False, "No face detected in the photo. Please ensure good lighting and face camera directly.", None
        if len(boxes) > 1:
            return False, f"Multiple faces detected ({len(boxes)}). Registration requires exactly one clear face.", None

        # Crop face
        box = boxes[0]
        face_crop = self.preprocessor.crop_bounding_box(image_bgr, box.xyxy, margin=0.15)

        # Save face image
        os.makedirs("data/registered_faces", exist_ok=True)
        photo_path = f"data/registered_faces/{roll_number.upper().strip()}.jpg"
        cv2.imwrite(photo_path, face_crop)

        # Extract 512-D embedding
        embedding = self.extract_embedding(face_crop)

        # Store in database
        ok, msg = self.db.register_student(
            roll_number=roll_number,
            name=name,
            department=department,
            embedding=embedding,
            photo_path=photo_path
        )

        if ok:
            self.refresh_gallery()
        return ok, msg, face_crop

    def process_frame(self, frame_bgr: np.ndarray,
                      mark_attendance: bool = True,
                      threshold: Optional[float] = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Processes a single video or camera frame:
        1. Detects all faces using Single-Stage SSD detector
        2. Performs passive anti-spoofing liveness check
        3. Extracts deep embedding and matches against enrolled student gallery
        4. Logs attendance in SQLite database for verified students
        5. Overlays rich visual annotations with color-coded status boxes.

        Returns:
            annotated_frame: BGR image with drawn HUD and bounding boxes
            detection_events: List of metadata dicts for all detected individuals
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr, []

        sim_thresh = threshold if threshold is not None else self.similarity_threshold
        annotated = frame_bgr.copy()
        h, w = frame_bgr.shape[:2]

        boxes = self.detector.detect_faces(frame_bgr)
        events = []

        for box in boxes:
            xmin, ymin, xmax, ymax = box.xyxy
            face_crop = self.preprocessor.crop_bounding_box(frame_bgr, (xmin, ymin, xmax, ymax), margin=0.08)

            # 1. Anti-Spoofing Liveness Check (Unit VI)
            is_live, liveness_score, liveness_metrics = self.liveness_detector.check_liveness(face_crop)

            # 2. Extract Embedding & Match (Unit III)
            query_emb = self.extract_embedding(face_crop)
            matched_roll, sim_score, all_candidates = self.matcher.match_against_gallery(
                query_emb, self.gallery, threshold=sim_thresh
            )

            # 3. Decision Logic & Database Logging
            status_label = ""
            box_color = (0, 255, 0)  # Default Green
            student_info = None

            if not is_live:
                # Spoof Attack Detected!
                box_color = (0, 0, 255)  # Bright Red
                status_label = f"SPOOF ALERT ({liveness_score:.2f})"
                event = {
                    "roll_number": "UNKNOWN",
                    "name": "Spoof Suspected",
                    "department": "N/A",
                    "status": "Spoof",
                    "confidence": round(sim_score, 3),
                    "liveness_score": round(liveness_score, 3),
                    "bbox": (xmin, ymin, xmax, ymax)
                }
            elif matched_roll is not None:
                # Genuine Enrolled Student Recognized!
                student_info = self.db.get_student_by_roll(matched_roll)
                name = student_info["name"] if student_info else matched_roll
                dept = student_info["department"] if student_info else "N/A"

                marked = False
                log_msg = ""
                if mark_attendance:
                    marked, log_msg, _ = self.db.mark_attendance(matched_roll, sim_score, liveness_score)

                box_color = (0, 200, 0)  # Green
                status_label = f"{name} ({sim_score * 100:.1f}%)"
                event = {
                    "roll_number": matched_roll,
                    "name": name,
                    "department": dept,
                    "status": "Present" if marked else "Already Marked",
                    "confidence": round(sim_score, 3),
                    "liveness_score": round(liveness_score, 3),
                    "message": log_msg,
                    "bbox": (xmin, ymin, xmax, ymax)
                }
            else:
                # Unregistered / Unknown Face
                box_color = (0, 215, 255)  # Amber / Yellow
                status_label = f"Unknown ({sim_score * 100:.1f}%)"
                event = {
                    "roll_number": "UNKNOWN",
                    "name": "Unregistered Visitor",
                    "department": "N/A",
                    "status": "Unknown",
                    "confidence": round(sim_score, 3),
                    "liveness_score": round(liveness_score, 3),
                    "bbox": (xmin, ymin, xmax, ymax)
                }

            events.append(event)

            # 4. Draw HUD Box
            self._draw_detection_box(annotated, xmin, ymin, xmax, ymax, status_label, box_color)

        return annotated, events

    def _draw_detection_box(self, img: np.ndarray, xmin: int, ymin: int, xmax: int, ymax: int,
                            label: str, color: Tuple[int, int, int]):
        """Renders rounded corners bounding box with banner badge."""
        # Main bounding rectangle
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)

        # Top label badge background
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.55
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        badge_y1 = max(0, ymin - text_h - 10)
        badge_y2 = ymin
        badge_x2 = min(img.shape[1], xmin + text_w + 12)

        cv2.rectangle(img, (xmin, badge_y1), (badge_x2, badge_y2), color, cv2.FILLED)
        cv2.putText(img, label, (xmin + 6, ymin - 5), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    def explain_face(self, face_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates Grad-CAM visual explanation showing which facial features
        were activated for recognition (Unit VI).

        Returns:
            face_resized: Resized face BGR image
            heatmap: 2D normalized heatmap
            overlay: Blended Grad-CAM heatmap over the face
        """
        rgb = self.preprocessor.bgr_to_rgb(face_bgr)
        letterboxed, _, _ = self.preprocessor.letterbox_resize(rgb, (224, 224))
        tensor = self.preprocessor.normalize_tensor(letterboxed, mode="imagenet")
        tensor = tensor.unsqueeze(0).to(self.device).requires_grad_(True)

        heatmap = self.grad_cam.generate_heatmap(tensor)
        overlay = self.grad_cam.overlay_heatmap(letterboxed, heatmap, alpha=0.45)

        return letterboxed, heatmap, overlay


if __name__ == "__main__":
    print("[Attendance Engine Test]")
    pipeline = AttendancePipeline("data/test_pipeline.db")
    print("Gallery loaded, count:", len(pipeline.gallery))

    # Test dummy enrollment
    dummy_face = (np.random.rand(240, 240, 3) * 255).astype(np.uint8)
    # Add an oval to simulate face
    cv2.circle(dummy_face, (120, 120), 80, (200, 180, 150), -1)

    emb = pipeline.extract_embedding(dummy_face)
    print(f"Extracted embedding shape: {emb.shape}")

    # Test Grad-CAM
    orig, heat, over = pipeline.explain_face(dummy_face)
    print(f"Grad-CAM generated successfully! Overlay shape: {over.shape}")

    del pipeline
    import gc
    gc.collect()
    try:
        if os.path.exists("data/test_pipeline.db"):
            os.remove("data/test_pipeline.db")
    except Exception:
        pass
    print("Attendance Engine module verified successfully!")
