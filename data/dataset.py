"""
PyTorch Dataset class for SRGAN satellite imagery
Handles loading HR images, tiling into patches, and applying degradation
"""

import os
from pathlib import Path
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from data.degradation import DegradationPipeline
from utils.config import Config


class SatelliteDataset(Dataset):
    """
    Dataset class for satellite image super-resolution
    
    If HR images are provided, tiles them into patches and applies degradation.
    If no HR images, can work with pre-generated LR-HR pairs.
    """
    
    def __init__(self, 
                 hr_image_dir: Optional[str] = None,
                 hr_patch_size: int = 256,
                 lr_patch_size: int = 64,
                 scale_factor: int = 4,
                 degradation_pipeline: Optional[DegradationPipeline] = None,
                 mode: str = 'train',
                 augment: bool = True):
        """
        Initialize dataset
        
        Args:
            hr_image_dir: Directory containing high-resolution images
            hr_patch_size: Size of HR patches to extract
            lr_patch_size: Size of LR patches (should be hr_patch_size / scale_factor)
            scale_factor: Upscaling factor
            degradation_pipeline: DegradationPipeline instance (created if None)
            mode: 'train' or 'val'
            augment: Whether to apply data augmentation
        """
        self.hr_image_dir = Path(hr_image_dir) if hr_image_dir else None
        self.hr_patch_size = hr_patch_size
        self.lr_patch_size = lr_patch_size
        self.scale_factor = scale_factor
        self.mode = mode
        self.augment = augment and (mode == 'train')
        
        # Initialize degradation pipeline
        if degradation_pipeline is None:
            self.degradation = DegradationPipeline(
                blur_sigma=1.2,
                noise_std=0.01,
                scale_factor=scale_factor
            )
        else:
            self.degradation = degradation_pipeline
        
        # Load image paths
        self.hr_images = []
        if self.hr_image_dir and self.hr_image_dir.exists():
            self.hr_images = self._load_image_paths(self.hr_image_dir)
        
        # Prepare patches
        self.patches = self._prepare_patches()
        
        # Use subset for faster training if enabled
        if mode == 'train' and hasattr(Config, 'USE_DATASET_SUBSET') and Config.USE_DATASET_SUBSET:
            subset_size = getattr(Config, 'DATASET_SUBSET_SIZE', 5000)
            if len(self.patches) > subset_size:
                import random
                random.seed(42)  # For reproducibility
                self.patches = random.sample(self.patches, subset_size)
                print(f"Using dataset subset: {len(self.patches)} patches (for faster training)")
        
        # Data augmentation transforms
        if self.augment:
            self.hr_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(90),
            ])
        else:
            self.hr_transform = transforms.Compose([])
    
    def _load_image_paths(self, image_dir: Path) -> list:
        """Load all image file paths from directory"""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.TIF', '.TIFF'}
        image_paths = []
        
        for ext in extensions:
            image_paths.extend(list(image_dir.glob(f'*{ext}')))
            image_paths.extend(list(image_dir.glob(f'*{ext.upper()}')))
            image_paths.extend(list(image_dir.glob(f'*{ext.lower()}')))
        
        return sorted(image_paths)
    
    def _prepare_patches(self) -> list:
        """
        Prepare list of (image_idx, patch_coords) tuples
        Each tuple represents one patch that can be extracted
        """
        patches = []
        
        if not self.hr_images:
            # If no HR images, return empty list (for synthetic data generation)
            return patches
        
        for img_idx, img_path in enumerate(self.hr_images):
            try:
                img = Image.open(img_path).convert('RGB')
                img_w, img_h = img.size
                
                # Calculate number of patches
                num_patches_h = img_h // self.hr_patch_size
                num_patches_w = img_w // self.hr_patch_size
                
                # Generate patch coordinates
                for i in range(num_patches_h):
                    for j in range(num_patches_w):
                        top = i * self.hr_patch_size
                        left = j * self.hr_patch_size
                        patches.append((img_idx, (top, left)))
            except Exception as e:
                print(f"Warning: Could not process {img_path}: {e}")
                continue
        
        return patches
    
    def _load_patch(self, img_idx: int, coords: Tuple[int, int]) -> Image.Image:
        """Load a specific patch from an image"""
        img_path = self.hr_images[img_idx]
        img = Image.open(img_path).convert('RGB')
        
        top, left = coords
        patch = img.crop((
            left,
            top,
            left + self.hr_patch_size,
            top + self.hr_patch_size
        ))
        
        return patch
    
    def __len__(self) -> int:
        """Return number of patches"""
        return len(self.patches)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a pair of (LR, HR) images
        
        Args:
            idx: Index of the patch
        
        Returns:
            Tuple of (LR_image, HR_image) tensors in range [0, 1]
        """
        if len(self.patches) == 0:
            # Generate synthetic patch if no real data
            hr_patch = torch.rand(3, self.hr_patch_size, self.hr_patch_size)
            lr_patch = self.degradation(hr_patch.unsqueeze(0)).squeeze(0)
            return lr_patch, hr_patch
        
        # Load HR patch
        img_idx, coords = self.patches[idx]
        hr_patch = self._load_patch(img_idx, coords)
        
        # Apply augmentation
        hr_patch = self.hr_transform(hr_patch)
        
        # Convert to tensor and normalize to [0, 1]
        to_tensor = transforms.ToTensor()
        hr_tensor = to_tensor(hr_patch)
        
        # Apply degradation to create LR image
        lr_tensor = self.degradation(hr_tensor.unsqueeze(0)).squeeze(0)
        
        return lr_tensor, hr_tensor
