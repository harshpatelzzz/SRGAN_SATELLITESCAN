"""
Discriminator network for SRGAN
CNN classifier that distinguishes real HR images from generated HR images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscriminatorBlock(nn.Module):
    """
    Discriminator block with convolution, batch norm, and LeakyReLU
    """
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 stride: int = 1,
                 use_bn: bool = True):
        """
        Initialize discriminator block
        
        Args:
            in_channels: Input feature maps
            out_channels: Output feature maps
            stride: Convolution stride
            use_bn: Whether to use batch normalization
        """
        super(DiscriminatorBlock, self).__init__()
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=not use_bn
        )
        
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        out = self.conv(x)
        if self.bn is not None:
            out = self.bn(out)
        out = self.leaky_relu(out)
        return out


class Discriminator(nn.Module):
    """
    Discriminator network for SRGAN
    
    Architecture:
    - Progressive feature extraction with increasing channels (64 → 128 → 256 → 512)
    - Global average pooling
    - Fully connected layers
    - Binary classification output (real vs fake)
    """
    
    def __init__(self, 
                 num_features: int = 64,
                 num_channels: int = 3):
        """
        Initialize Discriminator
        
        Args:
            num_features: Starting number of feature maps (default: 64)
            num_channels: Number of input channels (default: 3 for RGB)
        """
        super(Discriminator, self).__init__()
        
        self.num_features = num_features
        
        # Initial convolutional layer (no batch norm)
        self.conv_input = nn.Sequential(
            nn.Conv2d(num_channels, num_features, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Progressive feature extraction blocks
        # Each block doubles the number of channels and halves spatial size
        self.conv_blocks = nn.Sequential(
            DiscriminatorBlock(num_features, num_features, stride=2, use_bn=False),
            DiscriminatorBlock(num_features, num_features * 2, stride=1, use_bn=True),
            DiscriminatorBlock(num_features * 2, num_features * 2, stride=2, use_bn=True),
            DiscriminatorBlock(num_features * 2, num_features * 4, stride=1, use_bn=True),
            DiscriminatorBlock(num_features * 4, num_features * 4, stride=2, use_bn=True),
            DiscriminatorBlock(num_features * 4, num_features * 8, stride=1, use_bn=True),
            DiscriminatorBlock(num_features * 8, num_features * 8, stride=2, use_bn=True),
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(num_features * 8, num_features * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_features * 16, 1),
            nn.Sigmoid()  # Output probability
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input image [B, C, H, W] (HR image, real or generated)
        
        Returns:
            Probability that input is real [B, 1]
        """
        # Feature extraction
        out = self.conv_input(x)
        out = self.conv_blocks(out)
        
        # Global pooling
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)  # Flatten
        
        # Classification
        out = self.fc(out)
        
        return out
