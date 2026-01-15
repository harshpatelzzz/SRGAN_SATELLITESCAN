# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Setup

### Option 1: Download DIV2K (Recommended)

DIV2K is the standard benchmark for super-resolution:

```bash
python main.py download-dataset --dataset div2k --auto-config
```

This will:
- Download DIV2K dataset (~7GB)
- Extract and organize images
- Automatically update config.py

### Option 2: Download UC Merced (Satellite Imagery)

For satellite-specific imagery:

```bash
python main.py download-dataset --dataset ucmerced --auto-config
```

### Option 3: Use Your Own Dataset

Edit `utils/config.py`:

```python
HR_IMAGE_DIR = "/path/to/your/hr/images"
```

**Note**: If no dataset is provided, synthetic data will be generated automatically.

## Training Pipeline

### Step 1: Pre-train Generator

```bash
python main.py pretrain
```

This will:
- Train generator with MSE loss only
- Save checkpoints every 10 epochs
- Save final model as `checkpoints/generator_pretrained_final.pth`

### Step 2: Train SRGAN

```bash
python main.py train --pretrained checkpoints/generator_pretrained_final.pth
```

This will:
- Load pre-trained generator
- Train with full loss (VGG + GAN + MSE)
- Save checkpoints and sample images
- Save final models as `checkpoints/generator_final.pth` and `checkpoints/discriminator_final.pth`

## Evaluation

```bash
python main.py evaluate --model checkpoints/generator_final.pth
```

This will:
- Compute PSNR and SSIM metrics
- Compare against bicubic interpolation
- Save comparison images in `results/`

## Inference

Upscale a single image:

```bash
python main.py upscale --image path/to/image.jpg --model checkpoints/generator_final.pth
```

## Expected Outputs

- **Checkpoints**: Saved in `checkpoints/` directory
- **Sample Images**: Saved in `results/` directory during training
- **Logs**: Saved in `logs/` directory
- **Upscaled Images**: Saved in `results/inference/` directory

## Notes

- If no HR images are provided, the system automatically generates synthetic patches
- Training on GPU is recommended (automatically uses CUDA if available)
- Adjust batch size in `config.py` if you run out of memory
