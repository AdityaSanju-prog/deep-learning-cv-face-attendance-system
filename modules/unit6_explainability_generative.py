"""
UNIT VI: Generative Vision Models and Interpretability
======================================================
Syllabus Topics Covered:
1. Generative Deep Learning foundations:
   - Variational Autoencoders (VAE): Latent probabilistic manifold, reparameterization trick (z = mu + sigma * eps), ELBO loss
   - Generative Adversarial Networks (GANs): Minimax game, DCGAN, CycleGAN (cycle consistency), StyleGAN (style modulation)
   - Super-resolution & image restoration
2. Vision-Language Models: CLIP (Contrastive Language-Image Pre-training)
3. Model Interpretability:
   - Grad-CAM (Gradient-weighted Class Activation Mapping) for explaining CNN facial verification decisions
   - Saliency maps
4. Biometric Security: Passive Anti-Spoofing / Liveness Detection
   - Prevents photo-print and smartphone screen replay attacks using frequency domain and Laplacian texture analysis.
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. Grad-CAM (Gradient-Weighted Class Activation Mapping)
# =====================================================================

class GradCAM:
    """
    Implements Grad-CAM for Deep Vision Architectures (MobileNetV2, Custom CNN, ResNet).

    Mathematical Foundation:
    1. Feature maps A^k are extracted from target conv layer.
    2. Gradients of output score y^c with respect to feature maps A^k are computed:
       alpha_k^c = (1 / Z) * sum_i sum_j (d y^c / d A_{ij}^k)  [Global Average Pooling of Gradients]
    3. Grad-CAM heatmap is a weighted combination passed through ReLU:
       L_{Grad-CAM}^c = ReLU( sum_k alpha_k^c * A^k )
    4. The heatmap reveals facial regions (eyes, nose, contour) critical to identification!
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Register forward and backward hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor: torch.Tensor, target_index: Optional[int] = None) -> np.ndarray:
        """
        Computes Grad-CAM saliency heatmap for the given input face tensor.

        Args:
            input_tensor: Tensor of shape (1, C, H, W)
            target_index: Index of target feature/class to explain. If None, uses max output.

        Returns:
            heatmap: 2D numpy array [0.0, 1.0] of shape (H, W)
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]

        if target_index is None:
            target_index = torch.argmax(output[0]).item()

        # Backward pass with respect to target output
        score = output[0, target_index]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture activations/gradients.")

        # Compute importance weights alpha_k via Global Average Pooling of gradients
        # Shape of gradients: (1, K, H_feat, W_feat)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, K, 1, 1)

        # Weighted combination of feature maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H_feat, W_feat)

        # Apply ReLU to retain only features that positively correlate with the identity
        cam = F.relu(cam)

        # Normalize CAM to [0.0, 1.0]
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Resize heatmap back to input resolution (H, W)
        in_h, in_w = input_tensor.shape[2], input_tensor.shape[3]
        heatmap = cv2.resize(cam, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        return heatmap

    @staticmethod
    def overlay_heatmap(image_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5,
                        colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Converts 2D float heatmap to color Jet colormap and blends with original image.
        """
        heatmap_uint8 = np.uint8(255 * heatmap)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)

        # Match dimensions
        h, w = image_bgr.shape[:2]
        colored_heatmap = cv2.resize(colored_heatmap, (w, h))

        overlay = cv2.addWeighted(image_bgr, 1.0 - alpha, colored_heatmap, alpha, 0)
        return overlay


# =====================================================================
# 2. Anti-Spoofing & Liveness Detection (Passive Biometrics)
# =====================================================================

class AntiSpoofingLivenessDetector:
    """
    Passive Liveness & Anti-Spoofing Detection for Face Attendance.
    Prevents presentation attacks:
    1. Printed 2D paper photo attacks
    2. Smartphone/tablet digital screen replay attacks

    Techniques:
    - High-frequency texture analysis (Laplacian spatial variance)
    - Frequency Domain Fast Fourier Transform (FFT) high-frequency power ratio
      (Physical digital screens produce moire fringes and high-frequency artifacts).
    """

    def __init__(self, blur_threshold: float = 60.0, fft_ratio_threshold: float = 0.08):
        self.blur_threshold = blur_threshold
        self.fft_ratio_threshold = fft_ratio_threshold

    def compute_laplacian_variance(self, face_bgr: np.ndarray) -> float:
        """Computes Laplacian focus measure to detect flat paper/blur."""
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = float(laplacian.var())
        return variance

    def compute_fft_frequency_energy(self, face_bgr: np.ndarray) -> float:
        """
        Analyzes 2D discrete Fourier Transform spectrum.
        Screens exhibit distinct periodic spatial frequencies (Moiré pattern).
        """
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if h < 20 or w < 20:
            return 0.0

        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)

        # Calculate high frequency energy ratio
        cy, cx = h // 2, w // 2
        radius = min(h, w) // 4

        # Mask low frequencies
        y, x = np.ogrid[:h, :w]
        mask = (x - cx) ** 2 + (y - cy) ** 2 > radius ** 2

        high_freq_energy = np.mean(magnitude_spectrum[mask])
        total_energy = np.mean(magnitude_spectrum) + 1e-8
        ratio = float(high_freq_energy / total_energy)
        return ratio

    def check_liveness(self, face_bgr: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Evaluates facial crop for liveness.

        Returns:
            is_real: True if genuine live face, False if spoof
            liveness_score: Normalized confidence [0.0, 1.0]
            metrics: Diagnostic metrics dictionary
        """
        if face_bgr.size == 0 or face_bgr.shape[0] < 20 or face_bgr.shape[1] < 20:
            return False, 0.0, {"error": "Invalid face patch"}

        laplacian_var = self.compute_laplacian_variance(face_bgr)
        fft_ratio = self.compute_fft_frequency_energy(face_bgr)

        # Normalization and heuristic scoring
        norm_laplacian = min(1.0, laplacian_var / 250.0)
        norm_fft = min(1.0, fft_ratio / 0.8)

        # Composite score
        liveness_score = 0.6 * norm_laplacian + 0.4 * norm_fft

        # Genuine face criteria
        is_real = bool(laplacian_var >= self.blur_threshold and liveness_score >= 0.40)

        metrics = {
            "laplacian_variance": round(laplacian_var, 2),
            "fft_energy_ratio": round(fft_ratio, 4),
            "liveness_score": round(liveness_score, 3),
            "status": "LIVE" if is_real else "SPOOF_SUSPECTED"
        }
        return is_real, liveness_score, metrics


# =====================================================================
# 3. Generative Vision Models Overview
# =====================================================================

def generative_models_summary() -> Dict[str, Any]:
    """
    Syllabus overview of generative architectures and vision-language alignment.
    """
    return {
        "Variational Autoencoder (VAE)": {
            "Core Concept": "Learns probabilistic mapping to latent distribution q_phi(z|x) ~ N(mu, sigma^2)",
            "Key Trick": "Reparameterization trick: z = mu + sigma * epsilon, epsilon ~ N(0, I) enables backpropagation",
            "Loss Function": "Evidence Lower Bound (ELBO): L(theta, phi) = E[log p_theta(x|z)] - D_KL(q_phi(z|x) || p(z))"
        },
        "Generative Adversarial Networks (GANs)": {
            "Minimax Game": "min_G max_D V(D, G) = E[log D(x)] + E[log(1 - D(G(z)))]",
            "DCGAN": "Deep convolutional architectural constraints (strided convs, batch norm, LeakyReLU)",
            "CycleGAN": "Unpaired image-to-image translation with cycle consistency loss: ||F(G(x)) - x||_1",
            "StyleGAN": "Style modulation via AdaIN (Adaptive Instance Normalization) across multi-resolution synthesis layers"
        },
        "Vision-Language Models (CLIP)": {
            "Architecture": "Dual image encoder (ViT/ResNet) and text encoder (Transformer)",
            "Objective": "InfoNCE symmetric cross-entropy contrastive loss aligning visual & textual embeddings",
            "Face Application": "Zero-shot facial attribute verification (e.g., 'a photo of an enrolled student wearing glasses')"
        }
    }


if __name__ == "__main__":
    print("[Unit VI Module Test]")
    from modules.unit3_transfer_learning import TransferFaceEmbeddingNet

    model = TransferFaceEmbeddingNet(backbone_name="mobilenet_v2")
    # In MobileNetV2, features[-1] is the last inverted residual conv block
    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)

    dummy_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)
    heatmap = grad_cam.generate_heatmap(dummy_tensor)
    print(f"Grad-CAM Heatmap Shape: {heatmap.shape}, Range: [{heatmap.min():.2f}, {heatmap.max():.2f}]")

    liveness = AntiSpoofingLivenessDetector()
    dummy_face_bgr = (np.random.rand(112, 112, 3) * 255).astype(np.uint8)
    is_real, score, metrics = liveness.check_liveness(dummy_face_bgr)
    print(f"Liveness Check: is_real={is_real}, score={score}, metrics={metrics}")
    print("Unit VI module verified successfully!")
