"""
Main entry point for SRGAN Satellite Imagery Super-Resolution
Provides CLI interface for training, evaluation, and inference
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.config import Config
from utils.logger import setup_logger
from train.pretrain_generator import train_generator_pretrain
from train.train_srgan import train_srgan
from evaluate.evaluate import evaluate_model
from inference.upscale_image import upscale_image


def main():
    """Main function with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description='SRGAN for Satellite Imagery Super-Resolution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download DIV2K dataset (recommended - best for super-resolution)
  python main.py download-dataset --dataset div2k --auto-config

  # Download UC Merced dataset (satellite imagery)
  python main.py download-dataset --dataset ucmerced --auto-config

  # Pre-train generator
  python main.py pretrain

  # Train SRGAN (with pre-trained generator)
  python main.py train --pretrained checkpoints/generator_pretrained_final.pth

  # Evaluate model
  python main.py evaluate --model checkpoints/generator_final.pth

  # Upscale single image
  python main.py upscale --image data/test_image.jpg --model checkpoints/generator_final.pth
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Pre-train command
    pretrain_parser = subparsers.add_parser('pretrain', help='Pre-train generator with MSE loss')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train SRGAN with adversarial loss')
    train_parser.add_argument('--pretrained', type=str, default=None,
                             help='Path to pre-trained generator checkpoint')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--model', type=str, required=True,
                            help='Path to generator checkpoint')
    eval_parser.add_argument('--samples', type=int, default=None,
                            help='Number of samples to evaluate')
    eval_parser.add_argument('--no-save', action='store_true',
                            help='Do not save comparison images')
    
    # Upscale command
    upscale_parser = subparsers.add_parser('upscale', help='Upscale single image')
    upscale_parser.add_argument('--image', type=str, required=True,
                               help='Path to input image')
    upscale_parser.add_argument('--model', type=str, required=True,
                               help='Path to generator checkpoint')
    upscale_parser.add_argument('--output', type=str, default=None,
                               help='Path to save upscaled image')
    upscale_parser.add_argument('--scale', type=int, default=4,
                               help='Upscaling factor')
    
    # Download dataset command
    download_parser = subparsers.add_parser('download-dataset', help='Download dataset (DIV2K or UC Merced)')
    download_parser.add_argument('--dataset', type=str, default='div2k',
                                choices=['div2k', 'ucmerced'],
                                help='Dataset to download (default: div2k)')
    download_parser.add_argument('--auto-config', action='store_true',
                                help='Automatically update config.py with dataset path')
    
    args = parser.parse_args()
    
    if args.mode is None:
        parser.print_help()
        return
    
    # Setup
    Config.create_directories()
    logger = setup_logger("SRGAN_Main")
    Config.print_config()
    
    # Execute command
    if args.mode == 'pretrain':
        logger.info("Starting generator pre-training...")
        train_generator_pretrain()
    
    elif args.mode == 'train':
        logger.info("Starting SRGAN training...")
        train_srgan(pretrained_generator_path=args.pretrained)
    
    elif args.mode == 'evaluate':
        logger.info("Starting model evaluation...")
        evaluate_model(
            generator_path=args.model,
            num_samples=args.samples,
            save_comparisons=not args.no_save
        )
    
    elif args.mode == 'upscale':
        logger.info("Starting image upscaling...")
        upscale_image(
            image_path=args.image,
            model_path=args.model,
            output_path=args.output,
            scale_factor=args.scale
        )
    
    elif args.mode == 'download-dataset':
        from data.download_dataset import DatasetDownloader
        logger.info("Starting dataset download...")
        downloader = DatasetDownloader(args.dataset)
        hr_dir = downloader.download()
        
        if hr_dir and args.auto_config:
            # Update config.py automatically
            config_path = Config.PROJECT_ROOT / "utils" / "config.py"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    content = f.read()
                
                import re
                # Escape backslashes in Windows paths for the replacement string
                escaped_path = str(hr_dir).replace('\\', '\\\\')
                pattern = r'HR_IMAGE_DIR\s*=\s*[^\n]*'
                replacement = f"HR_IMAGE_DIR = r'{escaped_path}'"
                content = re.sub(pattern, replacement, content)
                
                with open(config_path, 'w') as f:
                    f.write(content)
                
                logger.info(f"Updated config.py with dataset path: {hr_dir}")
        
        if hr_dir:
            logger.info("=" * 60)
            logger.info("Dataset download completed!")
            logger.info(f"HR images directory: {hr_dir}")
            if not args.auto_config:
                logger.info(f"Set HR_IMAGE_DIR in utils/config.py to: {hr_dir}")
            logger.info("=" * 60)
        else:
            logger.error("Dataset download failed!")
            sys.exit(1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
