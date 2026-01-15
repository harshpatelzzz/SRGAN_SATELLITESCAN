# SRGAN for Optimised Satellite Imagery Upscaling

A complete, production-ready implementation of a **Super-Resolution Generative Adversarial Network (SRGAN)** for 4× upscaling of satellite imagery. This project implements a deep learning system that converts low-resolution satellite images into high-resolution images with superior perceptual quality compared to traditional interpolation methods.

## 📋 Project Overview

This SRGAN system is designed specifically for satellite imagery super-resolution, implementing:

- **Deep Residual Generator (SRResNet-based)** with 16 residual blocks
- **CNN Discriminator** for adversarial training
- **Perceptual Loss** using pre-trained VGG19
- **Adversarial Loss** for realistic texture generation
- **Pixel MSE Loss** for pixel-level accuracy

The system achieves superior **SSIM (Structural Similarity Index)** scores while maintaining competitive **PSNR (Peak Signal-to-Noise Ratio)** compared to bicubic interpolation.

## 🏗️ Architecture

### Generator (SRResNet-based)

The generator follows a deep residual network architecture:

1. **Initial Convolutional Layer**: 9×9 kernel, extracts initial features
2. **16 Residual Blocks**: Each block contains:
   - Conv → BatchNorm → PReLU → Conv → BatchNorm
   - Skip connection for gradient flow
3. **Post-Residual Processing**: Additional convolution with batch normalization
4. **Upsampling Blocks**: Two PixelShuffle blocks (2× each, total 4×)
5. **Output Layer**: Final convolution to RGB output

**Key Features:**
- 16 residual blocks for deep feature extraction
- PixelShuffle (sub-pixel convolution) for efficient upsampling
- Skip connections to preserve image details

### Discriminator

The discriminator is a CNN classifier that distinguishes real HR images from generated ones:

1. **Progressive Feature Extraction**: Increasing channels (64 → 128 → 256 → 512)
2. **Strided Convolutions**: Halve spatial dimensions at each stage
3. **Global Average Pooling**: Reduces spatial dimensions
4. **Fully Connected Layers**: Binary classification (real vs fake)

**Key Features:**
- LeakyReLU activations (α = 0.2)
- Batch normalization for stable training
- Sigmoid output for probability estimation

## 📊 Loss Functions

The total generator loss combines three components:

```
L_total = L_VGG + 10^-3 × L_GAN + L_MSE
```

Where:

- **L_VGG (Perceptual Loss)**: Euclidean distance between VGG19 feature maps (layer: relu5_4)
  - Encourages perceptual similarity rather than pixel-level matching
  - Weight: 1.0

- **L_GAN (Adversarial Loss)**: Binary cross-entropy loss
  - Trains generator to fool discriminator
  - Weight: 1e-3

- **L_MSE (Pixel Loss)**: Mean squared error between pixels
  - Ensures pixel-level accuracy
  - Weight: 1.0

## 📁 Project Structure

```
SRGAN-Satellite/
│
├── data/
│   ├── dataset.py          # PyTorch Dataset class for satellite images
│   ├── degradation.py      # Degradation pipeline (blur → downsample → noise)
│
├── models/
│   ├── generator.py        # SRResNet-based Generator
│   ├── discriminator.py    # CNN Discriminator
│
├── loss/
│   ├── vgg_loss.py         # VGG19 perceptual loss
│
├── train/
│   ├── pretrain_generator.py  # Pre-training script (MSE only)
│   ├── train_srgan.py         # Adversarial training script
│
├── evaluate/
│   ├── metrics.py          # PSNR and SSIM calculation
│   ├── evaluate.py         # Evaluation script
│
├── inference/
│   ├── upscale_image.py    # Single image upscaling
│
├── utils/
│   ├── config.py           # Configuration and hyperparameters
│   ├── logger.py           # Logging utility
│
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── main.py                # Main entry point
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- PyTorch with CUDA support (if using GPU)

### Setup

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download dataset** (Recommended):
   
   **Option A: Download DIV2K (Best for Super-Resolution)**
   ```bash
   python main.py download-dataset --dataset div2k --auto-config
   ```
   DIV2K is the standard benchmark dataset with 800 training + 100 validation images.
   
   **Option B: Download UC Merced (Satellite Imagery)**
   ```bash
   python main.py download-dataset --dataset ucmerced --auto-config
   ```
   UC Merced contains 2100 aerial/satellite images (256×256) across 21 land-use classes.
   
   **Option C: Use your own dataset**
   Edit `utils/config.py` and set `HR_IMAGE_DIR` to your high-resolution image directory:
   ```python
   HR_IMAGE_DIR = "/path/to/your/hr/images"
   ```
   
   **Note**: The `--auto-config` flag automatically updates the config file. Without it, you'll need to manually set `HR_IMAGE_DIR`.

## 📦 Dataset & Degradation Pipeline

### Recommended Datasets

1. **DIV2K** (Recommended - Standard SR Benchmark)
   - 800 training images + 100 validation images
   - High-quality, diverse content
   - Standard benchmark for super-resolution research
   - Download: `python main.py download-dataset --dataset div2k --auto-config`

2. **UC Merced Land Use Dataset** (Satellite-Specific)
   - 2100 aerial/satellite images (256×256)
   - 21 land-use classes (100 images per class)
   - Specifically designed for remote sensing
   - Download: `python main.py download-dataset --dataset ucmerced --auto-config`

3. **Custom Dataset**
   - Place your HR images in a directory
   - Set `HR_IMAGE_DIR` in `utils/config.py`

### Dataset Preparation

The system automatically handles dataset preparation:

1. **High-Resolution Images**: Downloaded or placed in configured directory
2. **Automatic Tiling**: Images are automatically tiled into 256×256 patches
3. **Degradation Pipeline**: LR images are generated on-the-fly using:
   - **Gaussian Blur** (σ = 1.2): Simulates atmospheric scattering
   - **Bicubic Downsampling**: 256×256 → 64×64 (4× reduction)
   - **Gaussian Noise** (σ = 0.01): Simulates sensor noise

### Degradation Steps

```
HR Image (256×256)
    ↓
[Gaussian Blur] → Atmospheric scattering
    ↓
[Bicubic Downsample] → 64×64
    ↓
[Add Gaussian Noise] → Sensor noise
    ↓
LR Image (64×64)
```

**Note**: If no HR images are provided, the system generates synthetic patches for testing.

## 🎓 Training Strategy

Training is performed in two phases:

### Phase 1: Pre-training Generator

**Purpose**: Stabilize generator weights before adversarial training

- **Loss**: MSE only
- **Epochs**: 100 (configurable)
- **Learning Rate**: 1e-4
- **Optimizer**: Adam (β₁=0.9, β₂=0.999)

```bash
python main.py pretrain
```

### Phase 2: Adversarial Training

**Purpose**: Fine-tune with full loss function (VGG + GAN + MSE)

- **Loss**: Combined (VGG + GAN + MSE)
- **Epochs**: 200 (configurable)
- **Learning Rate**: 1e-4 (with step decay)
- **Training**: Alternates between discriminator and generator updates

```bash
python main.py train --pretrained checkpoints/generator_pretrained_final.pth
```

### Training Configuration

All hyperparameters can be adjusted in `utils/config.py`:

- Batch size: 16
- Learning rates: 1e-4
- Loss weights: VGG=1.0, GAN=1e-3, MSE=1.0
- Checkpoint frequency: Every 10 epochs
- Sample image frequency: Every 5 epochs

## 📈 Evaluation Metrics

The system computes two standard super-resolution metrics:

### PSNR (Peak Signal-to-Noise Ratio)

Measures pixel-level accuracy:
```
PSNR = 20 × log₁₀(MAX) - 10 × log₁₀(MSE)
```

**Higher is better** (typically 20-40 dB for super-resolution)

### SSIM (Structural Similarity Index)

Measures perceptual similarity:
- Compares luminance, contrast, and structure
- Range: [0, 1] (higher is better)
- More aligned with human perception than PSNR

### Evaluation

```bash
python main.py evaluate --model checkpoints/generator_final.pth
```

This will:
- Compute PSNR and SSIM for all test samples
- Compare against bicubic interpolation baseline
- Save comparison images (LR | Bicubic | SRGAN | HR)

## 🔍 Inference

Upscale a single image:

```bash
python main.py upscale --image path/to/image.jpg --model checkpoints/generator_final.pth --output output.jpg
```

## 📊 Results Interpretation

### Expected Performance

- **PSNR**: Should be competitive with or slightly better than bicubic interpolation
- **SSIM**: Should significantly outperform bicubic interpolation (0.05-0.15 improvement)
- **Visual Quality**: Generated images should have:
  - Sharper edges and textures
  - More realistic details
  - Better perceptual quality

### Visual Comparison

The evaluation script generates comparison images showing:
1. **LR**: Original low-resolution input
2. **Bicubic**: Traditional interpolation baseline
3. **SRGAN**: Our super-resolved output
4. **HR**: Ground truth high-resolution

## 🔧 Configuration

All settings are centralized in `utils/config.py`:

### Key Parameters

```python
# Dataset
HR_PATCH_SIZE = 256
LR_PATCH_SIZE = 64
SCALE_FACTOR = 4

# Model
GENERATOR_RESIDUAL_BLOCKS = 16
GENERATOR_FEATURES = 64

# Training
BATCH_SIZE = 16
NUM_EPOCHS_PRETRAIN = 100
NUM_EPOCHS_ADVERSARIAL = 200
LEARNING_RATE_G = 1e-4
LEARNING_RATE_D = 1e-4

# Loss Weights
LOSS_VGG_WEIGHT = 1.0
LOSS_GAN_WEIGHT = 1e-3
LOSS_MSE_WEIGHT = 1.0
```

## 🎯 Usage Examples

### Complete Training Pipeline

```bash
# Step 1: Pre-train generator
python main.py pretrain

# Step 2: Train SRGAN
python main.py train --pretrained checkpoints/generator_pretrained_final.pth

# Step 3: Evaluate
python main.py evaluate --model checkpoints/generator_final.pth
```

### Quick Start (Synthetic Data)

If you don't have real satellite images, the system will generate synthetic patches:

```bash
# Just run training - synthetic data will be generated automatically
python main.py pretrain
```

## 🔬 Technical Details

### Generator Architecture Details

- **Residual Blocks**: 16 blocks with skip connections
- **Upsampling**: PixelShuffle (sub-pixel convolution) - more efficient than transposed convolution
- **Activation**: PReLU (Parametric ReLU) for better gradient flow
- **Output Activation**: Sigmoid (maps to [0, 1])

### Discriminator Architecture Details

- **Feature Progression**: 64 → 128 → 256 → 512 channels
- **Spatial Reduction**: Strided convolutions (stride=2)
- **Global Pooling**: Adaptive average pooling before FC layers
- **Output**: Sigmoid for binary classification

### VGG Loss Details

- **Model**: VGG19 (ImageNet pre-trained)
- **Layer**: relu5_4 (34th layer)
- **Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Frozen**: VGG parameters are frozen (feature extraction only)

## 🚀 Future Scope & Enhancements

### Advanced Architectures

1. **SwinIR / Transformers**
   - Replace CNN with Vision Transformer
   - Better long-range dependencies
   - State-of-the-art performance

2. **ESRGAN Enhancements**
   - Residual-in-Residual Dense Blocks (RRDB)
   - Relativistic discriminator
   - Improved perceptual loss

### Applications

1. **Video Super-Resolution**
   - Temporal consistency
   - Frame interpolation
   - Real-time processing

2. **Multi-Scale Super-Resolution**
   - Arbitrary scale factors
   - Progressive upsampling
   - Adaptive resolution

### Deployment

1. **Edge Deployment**
   - TensorRT optimization
   - Model quantization
   - Mobile/embedded devices

2. **Production Pipeline**
   - REST API integration
   - Batch processing
   - Cloud deployment

## 📝 Code Quality

- **Modular Design**: Each component is independently testable
- **Comprehensive Comments**: Inline documentation for academic review
- **Type Hints**: Python type annotations for clarity
- **Error Handling**: Robust error handling and logging
- **Reproducibility**: Fixed random seeds and deterministic operations

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `BATCH_SIZE` in `config.py`
   - Use gradient accumulation

2. **No HR Images Found**
   - System will use synthetic data
   - Set `HR_IMAGE_DIR` in `config.py`

3. **Poor Training Results**
   - Ensure pre-training completes successfully
   - Adjust learning rates
   - Check loss weights

## 📚 References

- **SRGAN Paper**: "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network" (Ledig et al., CVPR 2017)
- **SRResNet**: "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network" (Shi et al., CVPR 2016)
- **VGG19**: "Very Deep Convolutional Networks for Large-Scale Image Recognition" (Simonyan & Zisserman, ICLR 2015)

## 📄 License

This project is provided for academic and research purposes.

## 👥 Author

Built as a complete, production-ready implementation for satellite imagery super-resolution, suitable for academic evaluation and research applications.

---

**Note**: This implementation follows academic best practices and is designed for thorough evaluation in academic settings, including viva presentations and code reviews.
