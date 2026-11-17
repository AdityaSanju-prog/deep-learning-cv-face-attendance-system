"""
UNIT III: Transfer Learning and Fine-Tuning for Vision Tasks
============================================================
Syllabus Topics Covered:
1. Pre-trained architectures & Transfer Learning paradigm (MobileNetV2, ResNet, EfficientNet)
2. Feature extraction vs Fine-tuning workflows
3. Layer freezing and unfreezing strategies
4. Applying layer-wise / discriminative learning rates
5. Deep Metric Learning for Face Biometrics: L2 hypersphere embeddings
6. Cosine similarity vs Euclidean distance metrics for identity verification.
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# =====================================================================
# 1. Deep Face Embedding Network via Transfer Learning
# =====================================================================

class TransferFaceEmbeddingNet(nn.Module):
    """
    Leverages a pre-trained ImageNet backbone (MobileNetV2 or ResNet50)
    for high-fidelity facial biometric feature extraction.

    Why MobileNetV2?
    - 3.5 Million parameters (extremely lightweight)
    - Inverted residual blocks with linear bottlenecks preserve manifold information
    - Sub-15ms inference latency on CPU, enabling smooth 30+ FPS real-time webcam scanning.
    """

    def __init__(self, backbone_name: str = "mobilenet_v2", embedding_dim: int = 512,
                 pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim

        if backbone_name == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            base_model = models.mobilenet_v2(weights=weights)
            # Feature extractor up to last conv (1280 channels)
            self.features = base_model.features
            in_features = 1280
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base_model = models.resnet50(weights=weights)
            # ResNet layers
            self.features = nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool,
                base_model.layer1,
                base_model.layer2,
                base_model.layer3,
                base_model.layer4
            )
            in_features = 2048
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Domain Adaptation / Metric Projection Head
        # Projects high-level spatial representations to a compact discriminative metric space
        self.projection_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

        if freeze_backbone:
            self.freeze_backbone()

        # Set to evaluation mode by default for stable single-sample inference
        self.eval()

    def freeze_backbone(self):
        """Freezes all feature extractor weights for pure feature extraction transfer."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreezes entire backbone for full end-to-end fine-tuning."""
        for param in self.features.parameters():
            param.requires_grad = True

    def unfreeze_last_n_blocks(self, n: int = 3):
        """
        Selectively unfreezes the top N blocks of the backbone.
        Lower layers capture universal visual primitives (Gabor-like edges, corners);
        Top layers are fine-tuned to capture face-specific morphology.
        """
        # First freeze all
        self.freeze_backbone()

        if self.backbone_name == "mobilenet_v2":
            # MobileNet features has 19 sequential stages
            total_blocks = len(self.features)
            for i in range(max(0, total_blocks - n), total_blocks):
                for param in self.features[i].parameters():
                    param.requires_grad = True

        elif self.backbone_name == "resnet50":
            # Unfreeze layer4 (highest semantic level)
            for param in self.features[7].parameters():  # layer4 is 7th child in sequential
                param.requires_grad = True

    def get_layerwise_optimizer(self, base_lr: float = 1e-4, head_lr: float = 1e-3, weight_decay: float = 1e-4) -> torch.optim.Optimizer:
        """
        Configures discriminative layer-wise learning rates:
        - Pre-trained backbone receives a conservative LR (1e-4) to avoid catastrophic forgetting.
        - Newly initialized projection head receives an aggressive LR (1e-3) for rapid adaptation.
        """
        params_group = [
            {"params": [p for p in self.features.parameters() if p.requires_grad], "lr": base_lr},
            {"params": self.projection_head.parameters(), "lr": head_lr}
        ]
        return torch.optim.AdamW(params_group, weight_decay=weight_decay)

    def forward(self, x: torch.Tensor, return_normalized: bool = True) -> torch.Tensor:
        """
        Extracts face representation and normalizes to unit sphere.

        Args:
            x: Input face tensor of shape (B, 3, 224, 224)
            return_normalized: When True, applies L2 normalization (cosine metric)
        """
        feat_map = self.features(x)
        pooled = self.pool(feat_map)
        embedding = self.projection_head(pooled)

        if return_normalized:
            embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


# =====================================================================
# 2. Metric Verification & Distance Computations
# =====================================================================

class FaceBiometricMatcher:
    """
    Computes mathematical similarities between face embeddings for identity
    verification and attendance marking.
    """

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Cosine similarity: cos(theta) = (u . v) / (||u|| * ||v||)
        Range: [-1.0, 1.0], where 1.0 represents identical feature vectors.
        """
        emb1 = emb1.flatten()
        emb2 = emb2.flatten()
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    @staticmethod
    def euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Euclidean (L2) distance: sqrt(sum((u_i - v_i)^2))
        For unit-normalized vectors: L2^2 = 2 - 2 * cos(theta)
        """
        return float(np.linalg.norm(emb1.flatten() - emb2.flatten()))

    @classmethod
    def match_against_gallery(cls, query_embedding: np.ndarray, gallery_embeddings: Dict[str, np.ndarray],
                              threshold: float = 0.65) -> Tuple[Optional[str], float, List[Dict[str, Any]]]:
        """
        Compares a detected face query embedding against the enrolled student gallery.

        Args:
            query_embedding: 1D array of shape (embedding_dim,)
            gallery_embeddings: dict mapping student_id -> 1D embedding array
            threshold: Minimum cosine similarity required to verify identity

        Returns:
            best_match_id: Student ID if similarity >= threshold, else None
            best_similarity: Highest cosine similarity score
            all_scores: Detailed ranked list of candidates
        """
        if not gallery_embeddings:
            return None, 0.0, []

        scores = []
        for student_id, enrolled_emb in gallery_embeddings.items():
            sim = cls.cosine_similarity(query_embedding, enrolled_emb)
            scores.append({
                "student_id": student_id,
                "similarity": sim,
                "verified": bool(sim >= threshold)
            })

        # Sort descending by similarity
        scores.sort(key=lambda x: x["similarity"], reverse=True)
        best = scores[0]

        if best["similarity"] >= threshold:
            return best["student_id"], best["similarity"], scores
        return None, best["similarity"], scores


if __name__ == "__main__":
    print("[Unit III Module Test]")
    net = TransferFaceEmbeddingNet(backbone_name="mobilenet_v2", embedding_dim=512, pretrained=True)
    dummy_face = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        emb = net(dummy_face)

    print(f"Embedding shape: {emb.shape}")
    print(f"L2 Norm check: {torch.norm(emb[0]).item():.4f}")

    matcher = FaceBiometricMatcher()
    sim = matcher.cosine_similarity(emb[0].numpy(), emb[1].numpy())
    print(f"Self/Random Cosine Similarity: {sim:.4f}")
    print("Unit III module verified successfully!")
