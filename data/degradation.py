"""
Degradation pipeline for simulating low-resolution satellite images
Implements: Gaussian Blur → Bicubic Downsampling → Gaussian Noise
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np


class DegradationPipeline:
    """
    Degradation pipeline to simulate low-resolution images from high-resolution ones
    Mimics real-world satellite image degradation factors:
    - Atmospheric scattering (Gaussian blur)
    - Sensor limitations (bicubic downsampling)
    - Sensor noise (additive Gaussian noise)
    """
    
    def __init__(self, 
                 blur_sigma: float = 1.2,
                 noise_std: float = 0.01,
                 scale_factor: int = 4):
        """
        Initialize degradation pipeline
        
        Args:
            blur_sigma: Standard deviation for Gaussian blur kernel
            noise_std: Standard deviation for additive Gaussian noise
            scale_factor: Downsampling scale factor (e.g., 4 for 256→64)
        """
        self.blur_sigma = blur_sigma
        self.noise_std = noise_std
        self.scale_factor = scale_factor
        
        # Create Gaussian blur kernel
        kernel_size = int(6 * blur_sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        self.blur_kernel = self._create_gaussian_kernel(kernel_size, blur_sigma)
    
    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """
        Create 2D Gaussian kernel for blurring
        
        Args:
            kernel_size: Size of the kernel (must be odd)
            sigma: Standard deviation of Gaussian
        
        Returns:
            2D Gaussian kernel tensor [1, 1, H, W]
        """
        # Create coordinate grid
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords -= kernel_size // 2
        
        # Compute Gaussian
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        
        # Create 2D kernel
        kernel = g[:, None] * g[None, :]
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        
        return kernel
    
    def apply_blur(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian blur to simulate atmospheric scattering
        
        Args:
            image: Input image tensor [B, C, H, W]
        
        Returns:
            Blurred image tensor
        """
        # Ensure kernel is on same device as image
        kernel = self.blur_kernel.to(image.device)
        
        # Expand kernel to match number of channels (1, C, H, W)
        num_channels = image.shape[1]
        kernel = kernel.repeat(num_channels, 1, 1, 1)
        
        # Apply blur to each channel separately
        # Use groups=num_channels to apply kernel independently to each channel
        blurred = F.conv2d(image, kernel, groups=num_channels, padding=self.blur_kernel.shape[-1] // 2)
        
        return blurred
    
    def apply_downsample(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply bicubic downsampling to reduce resolution
        
        Args:
            image: Input image tensor [B, C, H, W]
        
        Returns:
            Downsampled image tensor [B, C, H/scale, W/scale]
        """
        # Get target size
        _, _, h, w = image.shape
        target_h, target_w = h // self.scale_factor, w // self.scale_factor
        
        # Bicubic interpolation (downsampling)
        downsampled = F.interpolate(
            image, 
            size=(target_h, target_w),
            mode='bicubic',
            align_corners=False
        )
        
        return downsampled
    
    def apply_noise(self, image: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to simulate sensor noise
        
        Args:
            image: Input image tensor [B, C, H, W]
        
        Returns:
            Noisy image tensor
        """
        noise = torch.randn_like(image) * self.noise_std
        noisy = image + noise
        
        # Clamp to valid range [0, 1]
        noisy = torch.clamp(noisy, 0.0, 1.0)
        
        return noisy
    
    def __call__(self, hr_image: torch.Tensor) -> torch.Tensor:
        """
        Apply complete degradation pipeline
        
        Args:
            hr_image: High-resolution image tensor [B, C, H, W] in range [0, 1]
        
        Returns:
            Low-resolution image tensor [B, C, H/scale, W/scale] in range [0, 1]
        """
        # Step 1: Gaussian blur (atmospheric scattering)
        blurred = self.apply_blur(hr_image)
        
        # Step 2: Bicubic downsampling (sensor limitations)
        downsampled = self.apply_downsample(blurred)
        
        # Step 3: Additive Gaussian noise (sensor noise)
        lr_image = self.apply_noise(downsampled)
        
        return lr_image
