"""
Evaluation metrics for super-resolution: PSNR and SSIM
"""

import torch
import torch.nn.functional as F
from math import log10
import numpy as np
from skimage.metrics import structural_similarity as ssim


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_value: float = 1.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR)
    
    PSNR = 20 * log10(MAX) - 10 * log10(MSE)
    
    Args:
        img1: First image tensor [B, C, H, W] or [C, H, W] in range [0, 1]
        img2: Second image tensor [B, C, H, W] or [C, H, W] in range [0, 1]
        max_value: Maximum pixel value (default: 1.0)
    
    Returns:
        PSNR value in dB
    """
    # Handle batch dimension
    if img1.dim() == 4:
        mse = torch.mean((img1 - img2) ** 2, dim=(1, 2, 3))
    else:
        mse = torch.mean((img1 - img2) ** 2)
    
    # Avoid division by zero
    mse = torch.clamp(mse, min=1e-10)
    
    if mse.dim() == 0:
        psnr = 20 * log10(max_value) - 10 * log10(mse.item())
    else:
        psnr = 20 * log10(max_value) - 10 * torch.log10(mse)
        psnr = psnr.mean().item()
    
    return psnr


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Calculate Structural Similarity Index (SSIM)
    
    SSIM measures perceptual similarity between images
    
    Args:
        img1: First image tensor [B, C, H, W] or [C, H, W] in range [0, 1]
        img2: Second image tensor [B, C, H, W] or [C, H, W] in range [0, 1]
    
    Returns:
        SSIM value in range [0, 1] (higher is better)
    """
    # Convert to numpy
    if img1.dim() == 4:
        # Take first image in batch
        img1_np = img1[0].cpu().numpy().transpose(1, 2, 0)
        img2_np = img2[0].cpu().numpy().transpose(1, 2, 0)
    else:
        img1_np = img1.cpu().numpy().transpose(1, 2, 0)
        img2_np = img2.cpu().numpy().transpose(1, 2, 0)
    
    # Ensure values are in [0, 1]
    img1_np = np.clip(img1_np, 0, 1)
    img2_np = np.clip(img2_np, 0, 1)
    
    # Convert to grayscale if RGB (SSIM typically computed on grayscale)
    if img1_np.shape[2] == 3:
        # Convert to grayscale using standard weights
        img1_gray = 0.299 * img1_np[:, :, 0] + 0.587 * img1_np[:, :, 1] + 0.114 * img1_np[:, :, 2]
        img2_gray = 0.299 * img2_np[:, :, 0] + 0.587 * img2_np[:, :, 1] + 0.114 * img2_np[:, :, 2]
    else:
        img1_gray = img1_np[:, :, 0]
        img2_gray = img2_np[:, :, 0]
    
    # Calculate SSIM
    ssim_value = ssim(
        img1_gray,
        img2_gray,
        data_range=1.0,
        win_size=min(7, min(img1_gray.shape))
    )
    
    return ssim_value


def calculate_metrics_batch(sr_images: torch.Tensor, 
                            hr_images: torch.Tensor) -> dict:
    """
    Calculate PSNR and SSIM for a batch of images
    
    Args:
        sr_images: Super-resolved images [B, C, H, W]
        hr_images: Ground truth HR images [B, C, H, W]
    
    Returns:
        Dictionary with 'psnr' and 'ssim' values
    """
    batch_size = sr_images.size(0)
    psnr_values = []
    ssim_values = []
    
    for i in range(batch_size):
        psnr = calculate_psnr(sr_images[i], hr_images[i])
        ssim_val = calculate_ssim(sr_images[i], hr_images[i])
        
        psnr_values.append(psnr)
        ssim_values.append(ssim_val)
    
    return {
        'psnr': np.mean(psnr_values),
        'ssim': np.mean(ssim_values),
        'psnr_std': np.std(psnr_values),
        'ssim_std': np.std(ssim_values)
    }
