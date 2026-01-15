"""
Generator network for SRGAN (SRResNet-based architecture)
Implements deep residual network with 16 residual blocks and PixelShuffle upsampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers
    Architecture: Conv → BatchNorm → PReLU → Conv → BatchNorm
    """
    
    def __init__(self, num_features: int = 64):
        """
        Initialize residual block
        
        Args:
            num_features: Number of feature maps
        """
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.prelu = nn.PReLU()
        
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection
        
        Args:
            x: Input feature map [B, C, H, W]
        
        Returns:
            Output feature map [B, C, H, W]
        """
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Residual connection
        out = out + residual
        
        return out


class Generator(nn.Module):
    """
    SRResNet-based Generator for 4x super-resolution
    
    Architecture:
    1. Initial convolutional layer
    2. 16 Residual blocks
    3. Post-residual convolutional layer
    4. Two upsampling blocks (2x each, total 4x)
    5. Final output layer
    """
    
    def __init__(self, 
                 num_residual_blocks: int = 16,
                 num_features: int = 64,
                 scale_factor: int = 4,
                 num_channels: int = 3):
        """
        Initialize Generator
        
        Args:
            num_residual_blocks: Number of residual blocks (default: 16)
            num_features: Base number of feature maps (default: 64)
            scale_factor: Upscaling factor (default: 4)
            num_channels: Number of input/output channels (default: 3 for RGB)
        """
        super(Generator, self).__init__()
        
        self.num_residual_blocks = num_residual_blocks
        self.num_features = num_features
        self.scale_factor = scale_factor
        
        # Initial convolutional layer
        self.conv_input = nn.Conv2d(
            num_channels, 
            num_features, 
            kernel_size=9, 
            padding=4,
            bias=False
        )
        self.prelu_input = nn.PReLU()
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_residual_blocks)]
        )
        
        # Post-residual convolutional layer
        self.conv_mid = nn.Conv2d(
            num_features,
            num_features,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn_mid = nn.BatchNorm2d(num_features)
        
        # Upsampling blocks (using PixelShuffle for sub-pixel convolution)
        num_upsample_blocks = int(torch.log2(torch.tensor(scale_factor)).item())
        self.upsample_blocks = nn.ModuleList()
        
        for _ in range(num_upsample_blocks):
            self.upsample_blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        num_features,
                        num_features * 4,  # 4x for 2x upsampling
                        kernel_size=3,
                        padding=1,
                        bias=False
                    ),
                    nn.PixelShuffle(2),  # 2x upsampling
                    nn.PReLU()
                )
            )
        
        # Final output layer
        self.conv_output = nn.Conv2d(
            num_features,
            num_channels,
            kernel_size=9,
            padding=4
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Low-resolution input image [B, C, H, W] in range [0, 1]
        
        Returns:
            High-resolution output image [B, C, H*scale, W*scale] in range [0, 1]
        """
        # Initial feature extraction
        out = self.conv_input(x)
        out = self.prelu_input(out)
        
        # Store for skip connection
        residual = out
        
        # Residual blocks
        out = self.residual_blocks(out)
        
        # Post-residual processing
        out = self.conv_mid(out)
        out = self.bn_mid(out)
        
        # Skip connection (add input features)
        out = out + residual
        
        # Upsampling
        for upsample_block in self.upsample_blocks:
            out = upsample_block(out)
        
        # Final output
        out = self.conv_output(out)
        
        # Tanh activation maps to [-1, 1], then normalize to [0, 1]
        # Alternatively, we can use sigmoid for [0, 1] directly
        out = torch.sigmoid(out)
        
        return out
