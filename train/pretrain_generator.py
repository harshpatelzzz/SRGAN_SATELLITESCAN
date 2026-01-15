"""
Pre-training script for Generator
Trains generator with MSE loss only to stabilize weights before adversarial training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import os
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.generator import Generator
from data.dataset import SatelliteDataset
from data.degradation import DegradationPipeline
from utils.config import Config
from utils.logger import setup_logger


def train_generator_pretrain():
    """
    Pre-train generator using MSE loss only
    This stabilizes the generator before adversarial training
    """
    # Setup
    Config.create_directories()
    logger = setup_logger("SRGAN_Pretrain")
    Config.print_config()
    
    device = torch.device(Config.DEVICE)
    logger.info(f"Using device: {device}")
    
    # Initialize degradation pipeline
    degradation = DegradationPipeline(
        blur_sigma=Config.GAUSSIAN_BLUR_SIGMA,
        noise_std=Config.GAUSSIAN_NOISE_STD,
        scale_factor=Config.SCALE_FACTOR
    )
    
    # Create dataset
    logger.info("Loading dataset...")
    train_dataset = SatelliteDataset(
        hr_image_dir=Config.HR_IMAGE_DIR,
        hr_patch_size=Config.HR_PATCH_SIZE,
        lr_patch_size=Config.LR_PATCH_SIZE,
        scale_factor=Config.SCALE_FACTOR,
        degradation_pipeline=degradation,
        mode='train',
        augment=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    logger.info(f"Dataset size: {len(train_dataset)} patches")
    
    # Initialize generator
    logger.info("Initializing Generator...")
    generator = Generator(
        num_residual_blocks=Config.GENERATOR_RESIDUAL_BLOCKS,
        num_features=Config.GENERATOR_FEATURES,
        scale_factor=Config.SCALE_FACTOR,
        num_channels=3
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in generator.parameters())
    logger.info(f"Generator parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        generator.parameters(),
        lr=Config.LEARNING_RATE_G,
        betas=(Config.BETA1, Config.BETA2)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Training loop
    logger.info("Starting pre-training...")
    generator.train()
    
    for epoch in range(Config.NUM_EPOCHS_PRETRAIN):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (lr_images, hr_images) in enumerate(train_loader):
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            sr_images = generator(lr_images)
            
            # Compute MSE loss
            loss = criterion(sr_images, hr_images)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{Config.NUM_EPOCHS_PRETRAIN}], "
                    f"Batch [{batch_idx+1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.6f}"
                )
        
        avg_loss = epoch_loss / num_batches
        current_lr = scheduler.get_last_lr()[0]
        
        logger.info(
            f"Epoch [{epoch+1}/{Config.NUM_EPOCHS_PRETRAIN}] completed. "
            f"Average Loss: {avg_loss:.6f}, LR: {current_lr:.6e}"
        )
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % Config.SAVE_CHECKPOINT_FREQ == 0:
            checkpoint_path = Config.CHECKPOINT_DIR / f"generator_pretrain_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_path = Config.CHECKPOINT_DIR / "generator_pretrained_final.pth"
    torch.save({
        'epoch': Config.NUM_EPOCHS_PRETRAIN,
        'model_state_dict': generator.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    logger.info(f"Final pre-trained model saved: {final_path}")
    
    logger.info("Pre-training completed!")
    
    # Automatically start adversarial training
    logger.info("=" * 60)
    logger.info("Starting adversarial training automatically...")
    logger.info("=" * 60)
    
    from train.train_srgan import train_srgan
    train_srgan(pretrained_generator_path=str(final_path))


if __name__ == "__main__":
    train_generator_pretrain()
