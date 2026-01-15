"""
Inference script for upscaling single images using trained SRGAN
"""

import torch
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.generator import Generator
from utils.config import Config
from utils.logger import setup_logger


def upscale_image(image_path: str, 
                  model_path: str,
                  output_path: str = None,
                  scale_factor: int = 4):
    """
    Upscale a single image using trained SRGAN
    
    Args:
        image_path: Path to input low-resolution image
        model_path: Path to trained generator checkpoint
        output_path: Path to save upscaled image (auto-generated if None)
        scale_factor: Upscaling factor (default: 4)
    """
    logger = setup_logger("SRGAN_Inference")
    
    device = torch.device(Config.DEVICE)
    logger.info(f"Using device: {device}")
    
    # Load image
    logger.info(f"Loading image from {image_path}")
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    logger.info(f"Original size: {original_size}")
    
    # Preprocess image
    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(image).unsqueeze(0).to(device)
    
    # Load generator
    logger.info(f"Loading generator from {model_path}")
    generator = Generator(
        num_residual_blocks=Config.GENERATOR_RESIDUAL_BLOCKS,
        num_features=Config.GENERATOR_FEATURES,
        scale_factor=scale_factor,
        num_channels=3
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['model_state_dict'])
    elif 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
    else:
        generator.load_state_dict(checkpoint)
    
    generator.eval()
    logger.info("Generator loaded successfully")
    
    # Upscale
    logger.info("Upscaling image...")
    with torch.no_grad():
        upscaled_tensor = generator(image_tensor)
    
    # Post-process
    upscaled_tensor = torch.clamp(upscaled_tensor, 0.0, 1.0)
    upscaled_image = transforms.ToPILImage()(upscaled_tensor.squeeze(0).cpu())
    
    new_size = upscaled_image.size
    logger.info(f"Upscaled size: {new_size}")
    
    # Save result
    if output_path is None:
        input_path = Path(image_path)
        output_path = Config.INFERENCE_OUTPUT_DIR / f"{input_path.stem}_upscaled{input_path.suffix}"
    
    Config.INFERENCE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    upscaled_image.save(output_path)
    logger.info(f"Upscaled image saved to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Upscale image using SRGAN')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to generator checkpoint')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save upscaled image')
    parser.add_argument('--scale', type=int, default=4,
                        help='Upscaling factor')
    args = parser.parse_args()
    
    upscale_image(
        image_path=args.image,
        model_path=args.model,
        output_path=args.output,
        scale_factor=args.scale
    )
