"""
VGG19-based perceptual loss for SRGAN
Computes feature-level distance between generated and target images
"""

import torch
import torch.nn as nn
import torchvision.models as models


class VGGLoss(nn.Module):
    """
    VGG19 perceptual loss
    
    Extracts features from a pre-trained VGG19 network and computes
    Euclidean distance between feature maps of generated and target images.
    This encourages perceptual similarity rather than just pixel-level similarity.
    """
    
    def __init__(self, 
                 layer_name: str = 'relu5_4',
                 feature_layer: int = 34):
        """
        Initialize VGG loss
        
        Args:
            layer_name: Name of the VGG layer to use (default: 'relu5_4')
            feature_layer: Index of the feature layer in VGG19 (default: 34 for relu5_4)
        """
        super(VGGLoss, self).__init__()
        
        # Load pre-trained VGG19
        try:
            # Try new PyTorch API (torchvision >= 0.13)
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        except AttributeError:
            # Fallback to old API
            vgg = models.vgg19(pretrained=True)
        
        # Extract features up to the specified layer
        # VGG19 structure: conv layers + ReLU activations
        # relu5_4 is at index 34 (after conv5_4 + ReLU)
        self.feature_extractor = nn.Sequential(
            *list(vgg.features.children())[:feature_layer + 1]
        )
        
        # Freeze VGG parameters (we only use it for feature extraction)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # VGG expects ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # But our images are in [0, 1] range, so we need to normalize
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        self.layer_name = layer_name
        self.mse_loss = nn.MSELoss()
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize image to VGG19 input format
        
        Args:
            x: Image tensor in range [0, 1] [B, C, H, W]
        
        Returns:
            Normalized image tensor
        """
        return (x - self.mean) / self.std
    
    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute VGG perceptual loss
        
        Args:
            generated: Generated HR image [B, C, H, W] in range [0, 1]
            target: Target HR image [B, C, H, W] in range [0, 1]
        
        Returns:
            Perceptual loss value (scalar tensor)
        """
        # Normalize images for VGG
        gen_norm = self.normalize(generated)
        target_norm = self.normalize(target)
        
        # Extract features
        gen_features = self.feature_extractor(gen_norm)
        target_features = self.feature_extractor(target_norm)
        
        # Compute MSE loss between feature maps
        loss = self.mse_loss(gen_features, target_features)
        
        return loss
