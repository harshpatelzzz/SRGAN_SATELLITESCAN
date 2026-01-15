"""
Configuration file for SRGAN Satellite Imagery Super-Resolution
Contains all hyperparameters and settings for training and evaluation
"""

import os
from pathlib import Path

class Config:
    """Central configuration class for SRGAN training and inference"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
    RESULTS_DIR = PROJECT_ROOT / "results"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # Dataset configuration
    # Set this to your HR image directory, or use download_dataset.py to download DIV2K/UC Merced
    HR_IMAGE_DIR = r"Z:\AIMLLABEL\datasets\DIV2K\HR"  # DIV2K dataset downloaded successfully
    HR_PATCH_SIZE = 256  # High-resolution patch size
    LR_PATCH_SIZE = 64   # Low-resolution patch size (4x downscale)
    SCALE_FACTOR = 4      # Upscaling factor
    
    # Fast training mode - use subset of dataset
    USE_DATASET_SUBSET = True      # Use only subset for faster training
    DATASET_SUBSET_SIZE = 2000     # Number of patches to use
    
    # Degradation pipeline parameters
    GAUSSIAN_BLUR_SIGMA = 1.2  # Standard deviation for Gaussian blur
    GAUSSIAN_NOISE_STD = 0.01  # Standard deviation for additive noise
    
    # Model architecture
    GENERATOR_RESIDUAL_BLOCKS = 16  # Number of residual blocks in generator
    GENERATOR_FEATURES = 64         # Base number of feature maps
    DISCRIMINATOR_FEATURES = 64     # Starting feature maps in discriminator
    
    # Training hyperparameters
    BATCH_SIZE = 8                  # Reduced to avoid memory issues
    NUM_EPOCHS_PRETRAIN = 3         # Pre-training epochs (~20-30 minutes)
    NUM_EPOCHS_ADVERSARIAL = 3      # Adversarial training epochs (~30-40 minutes, total ~1 hour)
    
    LEARNING_RATE_G = 1e-4          # Generator learning rate
    LEARNING_RATE_D = 1e-4          # Discriminator learning rate
    BETA1 = 0.9                     # Adam optimizer beta1
    BETA2 = 0.999                   # Adam optimizer beta2
    
    # Loss function weights
    LOSS_VGG_WEIGHT = 1.0           # VGG perceptual loss weight
    LOSS_GAN_WEIGHT = 1e-3          # Adversarial loss weight
    LOSS_MSE_WEIGHT = 1.0           # MSE loss weight
    
    # Training schedule
    D_UPDATE_FREQ = 1               # Discriminator update frequency
    G_UPDATE_FREQ = 1               # Generator update frequency
    
    # Checkpointing
    SAVE_CHECKPOINT_FREQ = 10       # Save checkpoint every N epochs
    SAVE_SAMPLE_FREQ = 5            # Save sample images every N epochs
    
    # Evaluation
    EVAL_FREQ = 10                  # Evaluate every N epochs
    METRICS = ['PSNR', 'SSIM']      # Metrics to compute
    
    # Device configuration
    DEVICE = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
    NUM_WORKERS = 4                 # DataLoader workers
    PIN_MEMORY = True               # Pin memory for faster GPU transfer
    
    # VGG loss configuration
    VGG_LAYER = 'relu5_4'           # VGG19 layer for perceptual loss
    VGG_MODEL_PATH = None           # Path to pre-trained VGG19 (None = download)
    
    # Inference
    INFERENCE_OUTPUT_DIR = RESULTS_DIR / "inference"
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        cls.INFERENCE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 50)
        print("SRGAN Configuration")
        print("=" * 50)
        print(f"Device: {cls.DEVICE}")
        print(f"Scale Factor: {cls.SCALE_FACTOR}x")
        print(f"HR Patch Size: {cls.HR_PATCH_SIZE}")
        print(f"LR Patch Size: {cls.LR_PATCH_SIZE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Generator Residual Blocks: {cls.GENERATOR_RESIDUAL_BLOCKS}")
        print(f"Pre-train Epochs: {cls.NUM_EPOCHS_PRETRAIN}")
        print(f"Adversarial Epochs: {cls.NUM_EPOCHS_ADVERSARIAL}")
        print("=" * 50)
