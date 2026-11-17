"""
Comprehensive Unit & Integration Test Suite
===========================================
Validates Units I through VI, database consistency, and end-to-end attendance flow.
"""

import os
import sys
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import cv2

from modules.unit1_foundations import ImagePreprocessor, SingleLayerPerceptron
from modules.unit2_cnn_training import CustomFaceCNN, calculate_conv_output_dim, calculate_receptive_field
from modules.unit3_transfer_learning import TransferFaceEmbeddingNet, FaceBiometricMatcher
from modules.unit4_object_detection import BoundingBox, calculate_iou, non_maximum_suppression, UniversalFaceDetector
from modules.unit5_segmentation_transformers import LightweightFaceUNet, FaceVisionTransformer
from modules.unit6_explainability_generative import GradCAM, AntiSpoofingLivenessDetector
from core.database import AttendanceDatabase
from core.attendance_engine import AttendancePipeline


class TestDeepFaceAttendPipeline(unittest.TestCase):

    def setUp(self):
        self.test_db = "data/test_suite.db"
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
        self.db = AttendanceDatabase(self.test_db)

    def tearDown(self):
        del self.db
        import gc
        gc.collect()
        if os.path.exists(self.test_db):
            try:
                os.remove(self.test_db)
            except Exception:
                pass

    # --- Unit I Tests ---
    def test_unit1_preprocessor(self):
        preproc = ImagePreprocessor()
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        letterboxed, r, (pw, ph) = preproc.letterbox_resize(img, (224, 224))
        self.assertEqual(letterboxed.shape, (224, 224, 3))
        tensor = preproc.normalize_tensor(letterboxed, mode="imagenet")
        self.assertEqual(tensor.shape, (3, 224, 224))

    def test_unit1_perceptron(self):
        p = SingleLayerPerceptron(input_dim=2)
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 0, 1])  # AND gate
        for _ in range(50):
            p.train_step(X, y)
        preds = p.forward(X, activation="step")
        self.assertEqual(preds.shape, (4,))

    # --- Unit II Tests ---
    def test_unit2_custom_cnn(self):
        model = CustomFaceCNN(embedding_dim=128)
        x = torch.randn(2, 3, 112, 112)
        emb = model(x)
        self.assertEqual(emb.shape, (2, 128))
        norm = torch.norm(emb[0], p=2).item()
        self.assertAlmostEqual(norm, 1.0, places=3)

    def test_unit2_receptive_field(self):
        layers = [{"kernel_size": 3, "stride": 1}, {"kernel_size": 2, "stride": 2}]
        rf = calculate_receptive_field(layers)
        self.assertEqual(rf, 4)

    # --- Unit III Tests ---
    def test_unit3_transfer_net_and_matcher(self):
        net = TransferFaceEmbeddingNet(backbone_name="mobilenet_v2", embedding_dim=512)
        net.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            emb1 = net(x).numpy().flatten()
            emb2 = net(x).numpy().flatten()

        matcher = FaceBiometricMatcher()
        sim_identical = matcher.cosine_similarity(emb1, emb2)
        self.assertAlmostEqual(sim_identical, 1.0, places=3)

        gallery = {"STUDENT_01": emb1}
        match_id, sim, _ = matcher.match_against_gallery(emb1, gallery, threshold=0.7)
        self.assertEqual(match_id, "STUDENT_01")

    # --- Unit IV Tests ---
    def test_unit4_iou_and_nms(self):
        b1 = BoundingBox(0, 0, 10, 10, confidence=0.9)
        b2 = BoundingBox(0, 0, 10, 10, confidence=0.8)
        self.assertAlmostEqual(calculate_iou(b1, b2), 1.0, places=3)

        suppressed = non_maximum_suppression([b1, b2], iou_threshold=0.5)
        self.assertEqual(len(suppressed), 1)
        self.assertEqual(suppressed[0].confidence, 0.9)

    # --- Unit V Tests ---
    def test_unit5_unet_and_vit(self):
        unet = LightweightFaceUNet()
        vit = FaceVisionTransformer()
        x = torch.randn(1, 3, 112, 112)

        mask = unet(x)
        self.assertEqual(mask.shape, (1, 1, 112, 112))

        vit_emb = vit(x)
        self.assertEqual(vit_emb.shape, (1, 128))

    # --- Unit VI Tests ---
    def test_unit6_gradcam_and_liveness(self):
        net = TransferFaceEmbeddingNet(backbone_name="mobilenet_v2")
        grad_cam = GradCAM(net, net.features[-1])
        x = torch.randn(1, 3, 224, 224, requires_grad=True)
        heatmap = grad_cam.generate_heatmap(x)
        self.assertEqual(heatmap.shape, (224, 224))

        liveness = AntiSpoofingLivenessDetector()
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        is_real, score, _ = liveness.check_liveness(img)
        self.assertIsInstance(is_real, bool)

    # --- Database & Attendance Flow Tests ---
    def test_database_registration_and_dedup(self):
        dummy_emb = np.random.randn(512).astype(np.float32)
        dummy_emb /= np.linalg.norm(dummy_emb)

        ok, msg = self.db.register_student("CS501", "Bob Johnson", "CSE", dummy_emb)
        self.assertTrue(ok)

        # 1st check-in should succeed
        ok1, msg1, _ = self.db.mark_attendance("CS501", 0.95, 0.88)
        self.assertTrue(ok1)

        # 2nd check-in on same day should be rejected as duplicate
        ok2, msg2, _ = self.db.mark_attendance("CS501", 0.96, 0.89)
        self.assertFalse(ok2)
        self.assertIn("already marked", msg2.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
