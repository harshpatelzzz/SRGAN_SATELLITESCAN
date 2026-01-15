"""
Evaluation script for SRGAN
Computes PSNR and SSIM metrics on test/validation dataset
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys
from torchvision.utils import save_image
import torch.nn.functional as F

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.generator import Generator
from data.dataset import SatelliteDataset
from data.degradation import DegradationPipeline
from evaluate.metrics import calculate_metrics_batch, calculate_psnr, calculate_ssim
from utils.config import Config
from utils.logger import setup_logger


def evaluate_model(generator_path: str, 
                   num_samples: int = None,
                   save_comparisons: bool = True):
    """
    Evaluate trained SRGAN model
    
    Args:
        generator_path: Path to trained generator checkpoint
        num_samples: Number of samples to evaluate (None = all)
        save_comparisons: Whether to save comparison images
    """
    # Setup
    Config.create_directories()
    logger = setup_logger("SRGAN_Eval")
    
    device = torch.device(Config.DEVICE)
    logger.info(f"Using device: {device}")
    
    # Load generator
    logger.info(f"Loading generator from {generator_path}")
    generator = Generator(
        num_residual_blocks=Config.GENERATOR_RESIDUAL_BLOCKS,
        num_features=Config.GENERATOR_FEATURES,
        scale_factor=Config.SCALE_FACTOR,
        num_channels=3
    ).to(device)
    
    checkpoint = torch.load(generator_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['model_state_dict'])
    elif 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
    else:
        generator.load_state_dict(checkpoint)
    
    generator.eval()
    logger.info("Generator loaded successfully")
    
    # Create dataset
    degradation = DegradationPipeline(
        blur_sigma=Config.GAUSSIAN_BLUR_SIGMA,
        noise_std=Config.GAUSSIAN_NOISE_STD,
        scale_factor=Config.SCALE_FACTOR
    )
    
    eval_dataset = SatelliteDataset(
        hr_image_dir=Config.HR_IMAGE_DIR,
        hr_patch_size=Config.HR_PATCH_SIZE,
        lr_patch_size=Config.LR_PATCH_SIZE,
        scale_factor=Config.SCALE_FACTOR,
        degradation_pipeline=degradation,
        mode='val',
        augment=False
    )
    
    if num_samples:
        eval_dataset.patches = eval_dataset.patches[:num_samples]
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,  # Evaluate one at a time for accurate metrics
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    logger.info(f"Evaluating on {len(eval_dataset)} samples")
    
    # Evaluation
    all_psnr_srgan = []
    all_psnr_bicubic = []
    all_ssim_srgan = []
    all_ssim_bicubic = []
    
    generator.eval()
    
    with torch.no_grad():
        for idx, (lr_images, hr_images) in enumerate(eval_loader):
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
            
            # SRGAN super-resolution
            sr_images = generator(lr_images)
            
            # Bicubic interpolation baseline
            bicubic_images = F.interpolate(
                lr_images,
                size=(Config.HR_PATCH_SIZE, Config.HR_PATCH_SIZE),
                mode='bicubic',
                align_corners=False
            )
            
            # Calculate metrics for SRGAN
            metrics_srgan = calculate_metrics_batch(sr_images, hr_images)
            all_psnr_srgan.append(metrics_srgan['psnr'])
            all_ssim_srgan.append(metrics_srgan['ssim'])
            
            # Calculate metrics for Bicubic
            metrics_bicubic = calculate_metrics_batch(bicubic_images, hr_images)
            all_psnr_bicubic.append(metrics_bicubic['psnr'])
            all_ssim_bicubic.append(metrics_bicubic['ssim'])
            
            # Save comparison images
            if save_comparisons and (idx < 10):  # Save first 10
                comparison = torch.cat([
                    F.interpolate(lr_images, size=(Config.HR_PATCH_SIZE, Config.HR_PATCH_SIZE), mode='bicubic'),
                    bicubic_images,
                    sr_images,
                    hr_images
                ], dim=0)
                
                save_path = Config.RESULTS_DIR / f"comparison_{idx}.png"
                save_image(comparison, save_path, nrow=1, normalize=False)
            
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(eval_loader)} samples")
    
    # Compute average metrics
    avg_psnr_srgan = sum(all_psnr_srgan) / len(all_psnr_srgan)
    avg_psnr_bicubic = sum(all_psnr_bicubic) / len(all_psnr_bicubic)
    avg_ssim_srgan = sum(all_ssim_srgan) / len(all_ssim_srgan)
    avg_ssim_bicubic = sum(all_ssim_bicubic) / len(all_ssim_bicubic)
    
    # Print results
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Number of samples: {len(eval_dataset)}")
    logger.info("")
    logger.info("SRGAN Results:")
    logger.info(f"  PSNR: {avg_psnr_srgan:.4f} dB")
    logger.info(f"  SSIM: {avg_ssim_srgan:.4f}")
    logger.info("")
    logger.info("Bicubic Interpolation Results:")
    logger.info(f"  PSNR: {avg_psnr_bicubic:.4f} dB")
    logger.info(f"  SSIM: {avg_ssim_bicubic:.4f}")
    logger.info("")
    logger.info("Improvement:")
    logger.info(f"  PSNR: {avg_psnr_srgan - avg_psnr_bicubic:+.4f} dB")
    logger.info(f"  SSIM: {avg_ssim_srgan - avg_ssim_bicubic:+.4f}")
    logger.info("=" * 60)
    
    return {
        'srgan_psnr': avg_psnr_srgan,
        'srgan_ssim': avg_ssim_srgan,
        'bicubic_psnr': avg_psnr_bicubic,
        'bicubic_ssim': avg_ssim_bicubic
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate SRGAN model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to generator checkpoint')
    parser.add_argument('--samples', type=int, default=None,
                        help='Number of samples to evaluate')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save comparison images')
    args = parser.parse_args()
    
    evaluate_model(
        generator_path=args.model,
        num_samples=args.samples,
        save_comparisons=not args.no_save
    )
