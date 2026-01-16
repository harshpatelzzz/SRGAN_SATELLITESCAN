"""
Adversarial training script for SRGAN
Alternates between training discriminator and generator with full loss function
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import os
import sys
from torchvision.utils import save_image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.generator import Generator
from models.discriminator import Discriminator
from loss.vgg_loss import VGGLoss
from data.dataset import SatelliteDataset
from data.degradation import DegradationPipeline
from utils.config import Config
from utils.logger import setup_logger


def train_srgan(pretrained_generator_path: str = None):
    """
    Train SRGAN with adversarial loss
    
    Args:
        pretrained_generator_path: Path to pre-trained generator checkpoint
    """
    # Setup
    Config.create_directories()
    logger = setup_logger("SRGAN_Train")
    Config.print_config()
    
    # Force CUDA device explicitly - AGGRESSIVE FIX
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Force first GPU
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # Explicitly set to first CUDA device
        device = torch.device('cuda:0')
        # Force GPU allocation to verify it's working
        test_tensor = torch.zeros(1).cuda()
        logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        del test_tensor
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        logger.warning("CUDA not available, falling back to CPU")
    
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
    
    # Initialize models
    logger.info("Initializing models...")
    generator = Generator(
        num_residual_blocks=Config.GENERATOR_RESIDUAL_BLOCKS,
        num_features=Config.GENERATOR_FEATURES,
        scale_factor=Config.SCALE_FACTOR,
        num_channels=3
    ).to(device)
    
    discriminator = Discriminator(
        num_features=Config.DISCRIMINATOR_FEATURES,
        num_channels=3
    ).to(device)
    
    # Load pre-trained generator if provided
    if pretrained_generator_path and Path(pretrained_generator_path).exists():
        logger.info(f"Loading pre-trained generator from {pretrained_generator_path}")
        checkpoint = torch.load(pretrained_generator_path, map_location=device)
        generator.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Pre-trained generator loaded successfully")
    
    # Count parameters
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    logger.info(f"Generator parameters: {g_params:,}")
    logger.info(f"Discriminator parameters: {d_params:,}")
    
    # Loss functions
    vgg_loss_fn = VGGLoss().to(device)
    mse_loss_fn = nn.MSELoss()
    bce_loss_fn = nn.BCELoss()
    
    # Optimizers
    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=Config.LEARNING_RATE_G,
        betas=(Config.BETA1, Config.BETA2)
    )
    
    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=Config.LEARNING_RATE_D,
        betas=(Config.BETA1, Config.BETA2)
    )
    
    # Learning rate schedulers
    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=50, gamma=0.5)
    scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=50, gamma=0.5)
    
    # Training labels
    real_label = 1.0
    fake_label = 0.0
    
    # Training loop
    logger.info("Starting adversarial training...")
    
    for epoch in range(Config.NUM_EPOCHS_ADVERSARIAL):
        epoch_loss_g = 0.0
        epoch_loss_d = 0.0
        num_batches = 0
        
        generator.train()
        discriminator.train()
        
        for batch_idx, (lr_images, hr_images) in enumerate(train_loader):
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
            batch_size = lr_images.size(0)
            
            # ============================================
            # Train Discriminator
            # ============================================
            optimizer_d.zero_grad()
            
            # Real images
            real_output = discriminator(hr_images)
            real_labels = torch.full(
                (batch_size, 1), 
                real_label, 
                dtype=torch.float32, 
                device=device
            )
            loss_d_real = bce_loss_fn(real_output, real_labels)
            loss_d_real.backward()
            
            # Fake images (generated)
            with torch.no_grad():
                sr_images = generator(lr_images)
            
            fake_output = discriminator(sr_images.detach())
            fake_labels = torch.full(
                (batch_size, 1), 
                fake_label, 
                dtype=torch.float32, 
                device=device
            )
            loss_d_fake = bce_loss_fn(fake_output, fake_labels)
            loss_d_fake.backward()
            
            # Total discriminator loss
            loss_d = loss_d_real + loss_d_fake
            optimizer_d.step()
            
            epoch_loss_d += loss_d.item()
            
            # ============================================
            # Train Generator
            # ============================================
            if (batch_idx + 1) % Config.G_UPDATE_FREQ == 0:
                optimizer_g.zero_grad()
                
                # Generate SR images
                sr_images = generator(lr_images)
                
                # Adversarial loss (generator wants discriminator to classify as real)
                fake_output = discriminator(sr_images)
                real_labels_gen = torch.full(
                    (batch_size, 1), 
                    real_label, 
                    dtype=torch.float32, 
                    device=device
                )
                loss_gan = bce_loss_fn(fake_output, real_labels_gen)
                
                # VGG perceptual loss
                loss_vgg = vgg_loss_fn(sr_images, hr_images)
                
                # MSE pixel loss
                loss_mse = mse_loss_fn(sr_images, hr_images)
                
                # Total generator loss
                loss_g = (
                    Config.LOSS_VGG_WEIGHT * loss_vgg +
                    Config.LOSS_GAN_WEIGHT * loss_gan +
                    Config.LOSS_MSE_WEIGHT * loss_mse
                )
                
                loss_g.backward()
                optimizer_g.step()
                
                epoch_loss_g += loss_g.item()
            
            num_batches += 1
            
            # Logging
            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{Config.NUM_EPOCHS_ADVERSARIAL}], "
                    f"Batch [{batch_idx+1}/{len(train_loader)}], "
                    f"Loss_D: {loss_d.item():.6f}, "
                    f"Loss_G: {loss_g.item():.6f} "
                    f"(VGG: {loss_vgg.item():.6f}, "
                    f"GAN: {loss_gan.item():.6f}, "
                    f"MSE: {loss_mse.item():.6f})"
                )
        
        avg_loss_g = epoch_loss_g / max(num_batches // Config.G_UPDATE_FREQ, 1)
        avg_loss_d = epoch_loss_d / num_batches
        
        current_lr_g = scheduler_g.get_last_lr()[0]
        current_lr_d = scheduler_d.get_last_lr()[0]
        
        logger.info(
            f"Epoch [{epoch+1}/{Config.NUM_EPOCHS_ADVERSARIAL}] completed. "
            f"Avg Loss_G: {avg_loss_g:.6f}, Avg Loss_D: {avg_loss_d:.6f}, "
            f"LR_G: {current_lr_g:.6e}, LR_D: {current_lr_d:.6e}"
        )
        
        # Update learning rates
        scheduler_g.step()
        scheduler_d.step()
        
        # Save sample images
        if (epoch + 1) % Config.SAVE_SAMPLE_FREQ == 0:
            generator.eval()
            with torch.no_grad():
                sample_lr = lr_images[:4]  # First 4 images
                sample_hr = hr_images[:4]
                sample_sr = generator(sample_lr)
                
                # Concatenate for visualization: LR | SR | HR
                comparison = torch.cat([
                    F.interpolate(sample_lr, size=(Config.HR_PATCH_SIZE, Config.HR_PATCH_SIZE), mode='bicubic'),
                    sample_sr,
                    sample_hr
                ], dim=0)
                
                save_path = Config.RESULTS_DIR / f"sample_epoch_{epoch+1}.png"
                save_image(comparison, save_path, nrow=4, normalize=False)
                logger.info(f"Sample images saved: {save_path}")
        
        # Save checkpoint
        if (epoch + 1) % Config.SAVE_CHECKPOINT_FREQ == 0:
            checkpoint_path = Config.CHECKPOINT_DIR / f"srgan_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'loss_g': avg_loss_g,
                'loss_d': avg_loss_d,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final models
    final_g_path = Config.CHECKPOINT_DIR / "generator_final.pth"
    final_d_path = Config.CHECKPOINT_DIR / "discriminator_final.pth"
    
    torch.save(generator.state_dict(), final_g_path)
    torch.save(discriminator.state_dict(), final_d_path)
    
    logger.info(f"Final models saved:")
    logger.info(f"  Generator: {final_g_path}")
    logger.info(f"  Discriminator: {final_d_path}")
    logger.info("Training completed!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train SRGAN')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pre-trained generator checkpoint')
    args = parser.parse_args()
    
    train_srgan(pretrained_generator_path=args.pretrained)
