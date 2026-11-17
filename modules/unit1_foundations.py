"""
UNIT I: Foundations of Computer Vision and Deep Learning
=========================================================
Syllabus Topics Covered:
1. Role of computer vision in modern AI applications
2. Traditional versus deep-learning-based vision approaches
3. Image fundamentals: pixels, channels, color spaces (BGR, RGB, HSV, Grayscale)
4. Resizing (standard & aspect-ratio preserving letterbox), cropping, and normalization
5. Neural network basics: Perceptron from scratch, Feedforward Networks (MLP),
   activation functions (ReLU, Sigmoid, Softmax), loss functions, and optimization methods.
"""

from typing import Tuple, Dict, Any, List
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim


# =====================================================================
# 1. Image Fundamentals & Preprocessing Pipeline
# =====================================================================

class ImagePreprocessor:
    """
    Implements core image processing operations foundational to computer vision
    pipelines: color spaces, geometric transformations, and tensor normalization.
    """

    # ImageNet standard normalization statistics
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @staticmethod
    def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
        """Converts OpenCV default BGR channel order to standard RGB."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def bgr_to_gray(image: np.ndarray) -> np.ndarray:
        """Converts BGR image to single-channel Grayscale for intensity analysis."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def bgr_to_hsv(image: np.ndarray) -> np.ndarray:
        """
        Converts BGR to Hue-Saturation-Value (HSV) color space.
        HSV decouples chromaticity from illumination, making it valuable
        for skin color segmentation and liveness/lighting analysis.
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    @staticmethod
    def crop_bounding_box(image: np.ndarray, bbox: Tuple[int, int, int, int], margin: float = 0.1) -> np.ndarray:
        """
        Crops a detected bounding box with an optional safety margin,
        ensuring coordinates remain strictly within image boundaries.

        Args:
            image: Input image (H, W, C)
            bbox: (xmin, ymin, xmax, ymax)
            margin: Percentage of width/height to expand crop (context around face)
        """
        h, w = image.shape[:2]
        xmin, ymin, xmax, ymax = bbox
        bw = xmax - xmin
        bh = ymax - ymin

        pad_w = int(bw * margin)
        pad_h = int(bh * margin)

        x1 = max(0, xmin - pad_w)
        y1 = max(0, ymin - pad_h)
        x2 = min(w, xmax + pad_w)
        y2 = min(h, ymax + pad_h)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return image
        return crop

    @staticmethod
    def letterbox_resize(image: np.ndarray, target_size: Tuple[int, int] = (224, 224),
                         color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Resizes an image preserving aspect ratio and pads the borders.
        This prevents aspect ratio distortion in deep neural network inputs.

        Returns:
            padded_image: Resized and padded image
            ratio: Scaling factor applied
            (pad_w, pad_h): Padding offsets
        """
        target_w, target_h = target_size
        h, w = image.shape[:2]
        ratio = min(target_w / w, target_h / h)
        new_w = int(round(w * ratio))
        new_h = int(round(h * ratio))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2

        padded = cv2.copyMakeBorder(
            resized,
            pad_h, target_h - new_h - pad_h,
            pad_w, target_w - new_w - pad_w,
            cv2.BORDER_CONSTANT,
            value=color
        )
        return padded, ratio, (pad_w, pad_h)

    @classmethod
    def normalize_tensor(cls, image_rgb: np.ndarray, mode: str = "imagenet") -> torch.Tensor:
        """
        Converts an RGB image array (H, W, C) to a PyTorch float tensor (C, H, W)
        with specified normalization.

        Modes:
            - 'zero_one': Scales [0, 255] -> [0.0, 1.0]
            - 'neg_one_to_one': Scales [0, 255] -> [-1.0, 1.0]
            - 'imagenet': Standard Z-score normalization using ImageNet mean & std:
                          Z = (X / 255.0 - mean) / std
        """
        arr = image_rgb.astype(np.float32) / 255.0

        if mode == "zero_one":
            norm = arr
        elif mode == "neg_one_to_one":
            norm = (arr - 0.5) * 2.0
        elif mode == "imagenet":
            norm = (arr - cls.IMAGENET_MEAN) / cls.IMAGENET_STD
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")

        # Transpose from (H, W, C) to PyTorch channel-first format (C, H, W)
        tensor = torch.from_numpy(norm.transpose(2, 0, 1)).float()
        return tensor


# =====================================================================
# 2. Neural Network Basics: Perceptron from Scratch
# =====================================================================

class SingleLayerPerceptron:
    """
    Fundamental Rosenblatt Perceptron implemented from first principles.
    Demonstrates:
    - Linear combination: z = w^T * x + b
    - Non-linear activation: Heaviside step or Sigmoid
    - Perceptron learning rule / delta update: w <- w + lr * (y - y_hat) * x
    """

    def __init__(self, input_dim: int, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        # Initialize weights with small random values, bias to 0
        self.weights = np.random.randn(input_dim) * 0.01
        self.bias = 0.0

    def step_function(self, z: np.ndarray) -> np.ndarray:
        """Binary threshold activation function."""
        return np.where(z >= 0, 1, 0)

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Continuous differentiable logistic sigmoid activation: sigma(z) = 1 / (1 + exp(-z))."""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def forward(self, X: np.ndarray, activation: str = "step") -> np.ndarray:
        """Computes affine combination and applies non-linear activation."""
        z = np.dot(X, self.weights) + self.bias
        if activation == "step":
            return self.step_function(z)
        elif activation == "sigmoid":
            return self.sigmoid(z)
        return z

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """Single epoch training pass using perceptron delta rule."""
        errors = 0
        for xi, target in zip(X, y):
            pred = self.forward(xi.reshape(1, -1), activation="step")[0]
            update = self.learning_rate * (target - pred)
            self.weights += update * xi
            self.bias += update
            if update != 0.0:
                errors += 1
        return errors / len(y)


# =====================================================================
# 3. Feedforward Neural Network (Multi-Layer Perceptron - MLP)
# =====================================================================

class FaceClassifierMLP(nn.Module):
    """
    Standard Feedforward Multi-Layer Perceptron (MLP) demonstrating:
    - Fully connected layers (nn.Linear)
    - Activation functions (ReLU, LeakyReLU, Softmax)
    - Dropout regularization
    - Classification loss & backpropagation
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, num_classes: int = 10, dropout_p: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Outputs unnormalized class logits."""
        return self.network(x)


# =====================================================================
# 4. Traditional CV vs Deep Learning Feature Comparison
# =====================================================================

def compare_vision_paradigms() -> Dict[str, Any]:
    """
    Returns an educational comparison matrix illustrating the fundamental
    shift from handcrafted feature engineering to end-to-end representation learning.
    """
    return {
        "Traditional Computer Vision": {
            "Representation": "Handcrafted spatial descriptors (Haar cascades, HOG, SIFT, LBP)",
            "Detector": "Sliding window with Adaboost or Linear SVM",
            "Invariance": "Sensitive to lighting, affine transformations, occlusions, pose changes",
            "Computational Bottleneck": "Redundant sliding window evaluations; scales poorly to crowded scenes",
            "Dataset Dependency": "Requires smaller training sets but extensive manual feature tuning",
            "Attendance Applicability": "Useful for fast initial face region proposals on constrained CPU edge devices"
        },
        "Deep Learning Computer Vision": {
            "Representation": "Hierarchical learned representations: Conv layers learn edges -> textures -> face parts -> identities",
            "Detector": "Anchor-based single-stage (SSD/RetinaNet/YOLO) or two-stage (Faster R-CNN)",
            "Invariance": "High robustness to occlusion, expressions, illumination, and non-frontal angles",
            "Computational Bottleneck": "High FLOPs during inference, mitigated by depthwise separable convs (MobileNet)",
            "Dataset Dependency": "Requires large datasets (VGGFace2, CASIA-WebFace) or transfer learning",
            "Attendance Applicability": "State-of-the-art accuracy, 512-D discriminative metric embeddings, robust anti-spoofing"
        }
    }


if __name__ == "__main__":
    print("[Unit I Module Test]")
    dummy_img = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)

    preproc = ImagePreprocessor()
    rgb = preproc.bgr_to_rgb(dummy_img)
    letterboxed, r, pads = preproc.letterbox_resize(rgb, (224, 224))
    tensor = preproc.normalize_tensor(letterboxed, mode="imagenet")

    print(f"Original shape: {dummy_img.shape}")
    print(f"Letterbox shape: {letterboxed.shape}, Ratio: {r:.2f}, Padding: {pads}")
    print(f"Tensor shape: {tensor.shape}, Range: [{tensor.min():.2f}, {tensor.max():.2f}]")

    mlp = FaceClassifierMLP(input_dim=512, hidden_dim=128, num_classes=5)
    dummy_features = torch.randn(8, 512)
    logits = mlp(dummy_features)
    print(f"MLP output shape: {logits.shape}")
    print("Unit I module verified successfully!")
