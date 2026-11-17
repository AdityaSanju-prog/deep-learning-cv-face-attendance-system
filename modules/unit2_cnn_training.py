"""
UNIT II: Convolutional Neural Networks and Training Techniques
==============================================================
Syllabus Topics Covered:
1. CNN Basics: convolution operation, filters, feature maps, padding, stride,
   pooling, and receptive fields calculation.
2. Modern CNN architectures: Evolution from LeNet -> AlexNet -> VGG -> ResNet -> Inception.
3. Regularization techniques: Batch Normalization and Dropout.
4. Data augmentation methods: Random flips, rotation, color jitter, affine transforms.
5. Training dynamics: Learning rate scheduling (StepLR, CosineAnnealingLR),
   weight initialization (He/Kaiming, Xavier/Glorot), and Early Stopping.
6. Custom CNN Implementation: `CustomFaceCNN` for facial feature representation.
"""

import math
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image


# =====================================================================
# 1. Mathematical Convolution Operation & Receptive Field
# =====================================================================

def calculate_conv_output_dim(in_size: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1) -> int:
    """
    Computes spatial output dimension of a convolutional layer:
    O = floor((W - K - (K-1)*(D-1) + 2*P) / S) + 1
    """
    effective_kernel = kernel_size + (kernel_size - 1) * (dilation - 1)
    return math.floor((in_size - effective_kernel + 2 * padding) / stride) + 1


def calculate_receptive_field(layers_config: List[Dict[str, int]]) -> int:
    """
    Computes theoretical cumulative receptive field across a sequence of layers:
    RF_{i+1} = RF_i + (K_{i+1} - 1) * cumulative_stride_i

    Args:
        layers_config: list of dicts with keys 'kernel_size' and 'stride'
    """
    rf = 1
    cumulative_stride = 1
    for layer in layers_config:
        k = layer.get("kernel_size", 1)
        s = layer.get("stride", 1)
        rf += (k - 1) * cumulative_stride
        cumulative_stride *= s
    return rf


def manual_2d_convolution(input_mat: np.ndarray, kernel: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    Performs 2D discrete spatial cross-correlation/convolution from first principles
    to demonstrate kernel filtering and feature map generation.
    """
    if padding > 0:
        input_mat = np.pad(input_mat, ((padding, padding), (padding, padding)), mode='constant')

    in_h, in_w = input_mat.shape
    k_h, k_w = kernel.shape
    out_h = (in_h - k_h) // stride + 1
    out_w = (in_w - k_w) // stride + 1

    output = np.zeros((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            r_start = i * stride
            c_start = j * stride
            patch = input_mat[r_start:r_start + k_h, c_start:c_start + k_w]
            output[i, j] = np.sum(patch * kernel)
    return output


# =====================================================================
# 2. Modern CNN Architectures Overview
# =====================================================================

def cnn_architectures_chronology() -> List[Dict[str, Any]]:
    """
    Chronological breakdown of milestone CNN architectures highlighting
    innovations, receptive fields, and skip connections.
    """
    return [
        {
            "name": "LeNet-5 (1998)",
            "key_innovations": "Introduced Conv-Pool-Conv-Pool-FC paradigm for digit recognition",
            "activation": "Tanh & Sigmoid",
            "params": "~60K"
        },
        {
            "name": "AlexNet (2012)",
            "key_innovations": "ReLU activation (mitigated vanishing gradients), Dropout (p=0.5), GPU multi-processing",
            "activation": "ReLU",
            "params": "~60M"
        },
        {
            "name": "VGG-16 / VGG-19 (2014)",
            "key_innovations": "Factorized large 7x7/5x5 kernels into stacks of small 3x3 kernels (lower params, deeper non-linearity)",
            "activation": "ReLU",
            "params": "~138M"
        },
        {
            "name": "Inception-v1/GoogLeNet (2014)",
            "key_innovations": "Multi-scale parallel convolutional kernels (1x1, 3x3, 5x5) and 1x1 bottleneck dimension reduction",
            "activation": "ReLU",
            "params": "~7M"
        },
        {
            "name": "ResNet (2015)",
            "key_innovations": "Residual identity skip connections (H(x) = F(x) + x) solving gradient degradation in ultra-deep networks (50-152 layers)",
            "activation": "ReLU + Batch Normalization",
            "params": "~25M (ResNet-50)"
        },
        {
            "name": "MobileNetV2 (2018)",
            "key_innovations": "Inverted residuals with linear bottlenecks & Depthwise Separable Convolutions; ultra-fast for edge attendance cameras",
            "activation": "ReLU6",
            "params": "~3.5M"
        }
    ]


# =====================================================================
# 3. Custom Deep CNN for Face Representation (`CustomFaceCNN`)
# =====================================================================

class CustomFaceCNN(nn.Module):
    """
    Custom Deep Convolutional Neural Network designed for facial feature
    extraction and metric projection.

    Architecture highlights:
    - 4 Convolutional stages with 3x3 kernels
    - Batch Normalization after every Conv2D to stabilize internal covariate shift
    - LeakyReLU activation to prevent 'dying ReLU' neurons
    - Spatial Max Pooling (2x2, stride 2) for translation invariance
    - Adaptive Average Pooling to decouple input resolution from linear layer size
    - Dropout (p=0.4) for regularization against overfitting
    - L2-normalized 128-D embedding projection
    """

    def __init__(self, in_channels: int = 3, embedding_dim: int = 128, dropout_p: float = 0.4):
        super().__init__()

        # Conv Stage 1: (C, 112, 112) -> (32, 56, 56)
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Conv Stage 2: (32, 56, 56) -> (64, 28, 28)
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Conv Stage 3: (64, 28, 28) -> (128, 14, 14)
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Conv Stage 4: (128, 14, 14) -> (256, 7, 7)
        self.stage4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Fully Connected Metric Projection Head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Applies Kaiming (He) normal initialization for convolutional weights
        and Xavier (Glorot) uniform for linear projection layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor, return_normalized: bool = True) -> torch.Tensor:
        """
        Forward pass producing facial feature representation.

        Args:
            x: Input face tensor of shape (B, C, H, W)
            return_normalized: If True, projects onto unit hypersphere (L2-norm = 1.0)
        """
        feat = self.stage1(x)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        feat = self.stage4(feat)
        embedding = self.fc(feat)

        if return_normalized:
            # L2 hypersphere normalization: ||v|| = 1.0
            embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


# =====================================================================
# 4. Data Augmentation Pipeline for Face Biometrics
# =====================================================================

def get_face_augmentation_pipeline(target_size: Tuple[int, int] = (112, 112)) -> Dict[str, T.Compose]:
    """
    Constructs robust training and inference data augmentation transforms.
    Training transforms simulate real-world classroom variations:
    - Horizontal head turns (random horizontal flip)
    - Head tilt (slight random rotation +/- 10 degrees)
    - Lighting variations (ColorJitter on brightness, contrast, saturation)
    - Motion blur or camera focus blur (GaussianBlur)
    - Shift/crop invariance (RandomAffine)
    """
    train_transform = T.Compose([
        T.Resize(target_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=(-10, 10)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = T.Compose([
        T.Resize(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return {"train": train_transform, "val": val_transform}


# =====================================================================
# 5. Training Dynamics: LR Scheduling & Early Stopping
# =====================================================================

class EarlyStopping:
    """
    Monitors validation loss/metric and halts training when improvement
    plateaus, preventing catastrophic overfitting.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, current_score: float) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            return False

        if self.mode == "min":
            improvement = (self.best_score - current_score) > self.min_delta
        else:
            improvement = (current_score - self.best_score) > self.min_delta

        if improvement:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


if __name__ == "__main__":
    print("[Unit II Module Test]")
    model = CustomFaceCNN(embedding_dim=128)
    dummy_input = torch.randn(4, 3, 112, 112)
    embeddings = model(dummy_input)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Norm of first embedding: {torch.norm(embeddings[0], p=2).item():.4f}")

    layers = [
        {"kernel_size": 3, "stride": 1},
        {"kernel_size": 2, "stride": 2},
        {"kernel_size": 3, "stride": 1},
        {"kernel_size": 2, "stride": 2},
    ]
    rf = calculate_receptive_field(layers)
    print(f"Theoretical Receptive Field: {rf} pixels")
    print("Unit II module verified successfully!")
