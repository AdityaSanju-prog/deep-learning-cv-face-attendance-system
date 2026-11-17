"""
UNIT V: Image Segmentation and Advanced Vision Architectures
============================================================
Syllabus Topics Covered:
1. Image Segmentation taxonomy: Semantic vs Instance segmentation
2. Segmentation Architectures: UNet (encoder-decoder with skip connections) & DeepLab (Atrous/Dilated Convolutions, ASPP)
3. Instance segmentation principles (Mask R-CNN: RoIAlign + FCN branch)
4. Vision Transformers (ViT):
   - Patch embedding decomposition
   - Learnable [CLS] classification token & 1D positional encodings
   - Scaled Dot-Product Self-Attention: Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
   - Multi-Head Self-Attention (MHSA)
   - Transformer Encoder blocks (LayerNorm, Residual Additions, MLP)
5. Swin Transformer principles: Shifted window hierarchical attention
6. Practical Implementation: Lightweight Face UNet segmentation module + Face ViT feature encoder.
"""

import math
from typing import Tuple, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


# =====================================================================
# 1. Image Segmentation: Lightweight Face UNet Architecture
# =====================================================================

class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2 block for UNet encoder/decoder."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LightweightFaceUNet(nn.Module):
    """
    Lightweight UNet semantic segmentation network:
    Isolates facial foreground from complex classroom background clutter.

    Architectural highlights:
    - Contracting path (Encoder): Captures contextual semantics
    - Expanding path (Decoder): Enables precise spatial localization
    - Skip connections: Concatenates high-resolution encoder features directly
      to upsampled decoder features to preserve fine facial boundaries.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()
        # Contracting Path
        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(64, 128)

        # Expanding Path with Skip Connections
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)  # 64 (up) + 64 (skip) = 128

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)   # 32 (up) + 32 (skip) = 64

        # Final 1x1 conv predicting binary foreground mask
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decoder + Skip connection from e2
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        # Decoder + Skip connection from e1
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        mask_logits = self.final_conv(d1)
        return torch.sigmoid(mask_logits)


# =====================================================================
# 2. Vision Transformer (ViT) Architecture for Facial Features
# =====================================================================

class PatchEmbedding(nn.Module):
    """
    Deconstructs a 2D face image (H, W, C) into a sequence of non-overlapping
    flattened patches (P x P), and projects them to dimension D:
    N = (H * W) / P^2 tokens.
    """
    def __init__(self, img_size: int = 112, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # 2D Conv with kernel=patch_size and stride=patch_size performs patch projection
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        x = self.projection(x)
        # Flatten spatial dims to tokens: (B, embed_dim, N) -> (B, N, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention (MHSA) module:
    Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V
    Allows tokens (face patches) to dynamically attend to other face regions
    (e.g., eye token attends to eyebrow and nose tokens).
    """
    def __init__(self, embed_dim: int = 192, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        # Compute Q, K, V
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each shape: (B, num_heads, N, head_dim)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        context = torch.matmul(attn_weights, v)  # (B, num_heads, N, head_dim)
        context = context.permute(0, 2, 1, 3).reshape(B, N, D)
        output = self.out_proj(context)
        return output


class TransformerEncoderBlock(nn.Module):
    """Single standard ViT Transformer Encoder block with residual connections."""
    def __init__(self, embed_dim: int = 192, num_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN residual block
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FaceVisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) implementation for facial representation:
    - Splits 112x112 face crop into 7x7 = 49 patches (16x16 pixels each)
    - Appends learnable [CLS] token representing whole face
    - Adds 1D learnable positional embeddings
    - Passes through stacked Multi-Head Self-Attention encoder blocks
    - Extracts [CLS] token embedding for facial verification!
    """

    def __init__(self, img_size: int = 112, patch_size: int = 16, in_channels: int = 3,
                 embed_dim: int = 192, depth: int = 4, num_heads: int = 4, output_dim: int = 128):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Learnable 1D positional encodings for (1 + num_patches) tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(0.1)

        # Transformer Encoder Blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, output_dim)

        # Initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # 1. Patch decomposition
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # 2. Expand & prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1 + num_patches, embed_dim)

        # 3. Add positional encoding
        x = self.pos_drop(x + self.pos_embed)

        # 4. Multi-head self-attention transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # 5. Extract [CLS] token representation
        cls_feature = x[:, 0]
        embedding = self.head(cls_feature)
        return F.normalize(embedding, p=2, dim=1)


if __name__ == "__main__":
    print("[Unit V Module Test]")
    unet = LightweightFaceUNet()
    dummy_face = torch.randn(2, 3, 112, 112)
    mask = unet(dummy_face)
    print(f"UNet Mask Output Shape: {mask.shape}")

    vit = FaceVisionTransformer()
    vit_emb = vit(dummy_face)
    print(f"ViT Face Embedding Shape: {vit_emb.shape}")
    print(f"ViT L2 Norm: {torch.norm(vit_emb[0]).item():.4f}")
    print("Unit V module verified successfully!")
