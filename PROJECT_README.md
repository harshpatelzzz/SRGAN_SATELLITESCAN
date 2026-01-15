# 🛰️ SRGAN for Optimised Satellite Imagery Upscaling

A complete, production-ready implementation of **Super-Resolution Generative Adversarial Network (SRGAN)** for 4× upscaling of satellite imagery. This project includes training pipeline, REST API, and modern web frontend.

## 🎯 Project Overview

This system converts low-resolution satellite images (64×64) into high-resolution images (256×256) using deep learning, achieving superior perceptual quality compared to traditional interpolation methods.

### Key Features

- ✅ **Deep Residual Generator** (SRResNet-based) with 16 residual blocks
- ✅ **CNN Discriminator** for adversarial training
- ✅ **VGG19 Perceptual Loss** for realistic textures
- ✅ **Adversarial + MSE Loss** for optimal training
- ✅ **FastAPI REST API** for production deployment
- ✅ **Next.js Frontend** with dark theme and animations
- ✅ **Google Colab Support** for free GPU training
- ✅ **Comprehensive Evaluation** (PSNR, SSIM, visual comparisons)

## 📁 Project Structure

```
SRGAN-Satellite/
│
├── api/                    # FastAPI backend
│   ├── main.py            # REST API server
│   └── requirements.txt   # API dependencies
│
├── frontend-next/         # Next.js frontend
│   ├── app/              # Next.js app directory
│   ├── components/       # React components
│   └── package.json      # Frontend dependencies
│
├── colab/                 # Google Colab notebooks
│   └── SRGAN_Training.ipynb
│
├── data/                  # Dataset handling
│   ├── dataset.py
│   ├── degradation.py
│   └── download_dataset.py
│
├── models/                # Neural networks
│   ├── generator.py
│   └── discriminator.py
│
├── train/                 # Training scripts
│   ├── pretrain_generator.py
│   └── train_srgan.py
│
├── evaluate/             # Evaluation
│   ├── metrics.py
│   └── evaluate.py
│
├── inference/            # Inference
│   └── upscale_image.py
│
├── loss/                 # Loss functions
│   └── vgg_loss.py
│
├── utils/                # Utilities
│   ├── config.py
│   └── logger.py
│
├── main.py               # CLI entry point
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <your-repo-url>
cd SRGAN-Satellite

# Install Python dependencies
pip install -r requirements.txt

# Install API dependencies
pip install -r api/requirements.txt

# Install frontend dependencies
cd frontend-next
npm install
cd ..
```

### 2. Download Dataset

```bash
# Download DIV2K (recommended)
python main.py download-dataset --dataset div2k --auto-config

# Or UC Merced (satellite-specific)
python main.py download-dataset --dataset ucmerced --auto-config
```

### 3. Training

#### Option A: Local Training

```bash
# Pre-train generator
python main.py pretrain

# Train SRGAN (automatic continuation)
python main.py train --pretrained checkpoints/generator_pretrained_final.pth
```

#### Option B: Google Colab (Free GPU)

1. Upload `colab/SRGAN_Training.ipynb` to Google Colab
2. Run cells sequentially
3. Training runs on free GPU (T4/V100)

### 4. Evaluation

```bash
python main.py evaluate --model checkpoints/generator_final.pth
```

### 5. Start Services

#### Backend API

```bash
cd api
python main.py
# API runs on http://localhost:8000
```

#### Frontend

```bash
cd frontend-next
npm run dev
# Frontend runs on http://localhost:3000
```

## 🏗️ Architecture

### Generator (SRResNet-based)

- **Input**: 64×64×3 (LR image)
- **Architecture**:
  - Initial Conv (9×9, 64 features)
  - 16 Residual Blocks (Conv → BN → PReLU → Conv → BN)
  - Post-residual processing
  - 2× Upsample Block (PixelShuffle)
  - 2× Upsample Block (PixelShuffle)
  - Output Conv (9×9, 3 channels)
- **Output**: 256×256×3 (HR image)
- **Parameters**: 1,546,774

### Discriminator

- **Input**: 256×256×3 (HR image)
- **Architecture**:
  - Progressive feature extraction (64 → 128 → 256 → 512)
  - Strided convolutions for downsampling
  - Global average pooling
  - Fully connected layers (512 → 1024 → 1)
- **Output**: Probability (0-1)
- **Parameters**: 5,213,505

### Loss Function

```
L_total = L_VGG + 10⁻³ × L_GAN + L_MSE
```

- **L_VGG**: VGG19 perceptual loss (weight: 1.0)
- **L_GAN**: Adversarial loss (weight: 10⁻³)
- **L_MSE**: Pixel MSE loss (weight: 1.0)

## 📊 Training Strategy

### Phase 1: Pre-training (MSE Only)

- **Purpose**: Stabilize generator weights
- **Loss**: MSE only
- **Epochs**: 1-100 (configurable)
- **Time**: ~10-60 minutes (depends on dataset size)

### Phase 2: Adversarial Training

- **Purpose**: Fine-tune with full loss
- **Loss**: VGG + GAN + MSE
- **Epochs**: 1-200 (configurable)
- **Time**: ~10-120 minutes

## 🎨 Frontend Features

- 🌙 **Dark Theme** with gradient accents
- ✨ **Framer Motion** animations
- 🖼️ **Before/After Slider** for comparison
- 📊 **Metrics Dashboard** (PSNR, SSIM)
- 🏗️ **Architecture Visualization**
- 📚 **Dataset Explanation** page
- 📤 **Drag & Drop** image upload
- ⚡ **Real-time Processing**

## 🔌 API Endpoints

### `GET /api/health`
Health check endpoint

### `GET /api/model/info`
Get model information and status

### `POST /api/upscale`
Upscale an image
- **Request**: Multipart form with image file
- **Response**: JSON with upscaled image (base64) and metrics

### `GET /api/metrics`
Get training evaluation metrics

## 📈 Evaluation Metrics

- **PSNR** (Peak Signal-to-Noise Ratio): Pixel-level accuracy
- **SSIM** (Structural Similarity Index): Perceptual quality
- **Visual Comparisons**: Side-by-side with bicubic baseline

## 🛠️ Configuration

Edit `utils/config.py` to customize:

- Dataset paths
- Model architecture (residual blocks, features)
- Training hyperparameters (batch size, epochs, learning rates)
- Loss weights
- Device (CPU/GPU)

## 📦 Datasets

### DIV2K (Recommended)
- **Size**: 900 images (800 train + 100 val)
- **Quality**: Very High
- **Download**: `python main.py download-dataset --dataset div2k`

### UC Merced
- **Size**: 2,100 satellite images
- **Quality**: High
- **Download**: `python main.py download-dataset --dataset ucmerced`

## 🎓 Academic Use

This project is designed for:
- ✅ College project submissions
- ✅ Viva presentations
- ✅ Research papers
- ✅ Portfolio showcase

### Key Highlights for Viva

1. **Complete Implementation**: End-to-end system
2. **Research-Grade Code**: Well-commented, modular
3. **Modern Stack**: PyTorch, FastAPI, Next.js
4. **Production-Ready**: REST API, web interface
5. **Comprehensive Documentation**: README, code comments

## 🚀 Deployment

### Local Development

```bash
# Terminal 1: API
cd api && python main.py

# Terminal 2: Frontend
cd frontend-next && npm run dev
```

### Production

```bash
# Build frontend
cd frontend-next
npm run build
npm start

# Run API with uvicorn
cd api
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 📝 License

This project is provided for academic and research purposes.

## 👥 Credits

Built as a complete, production-ready implementation for satellite imagery super-resolution, suitable for academic evaluation and research applications.

---

**For questions or issues, please refer to the documentation or open an issue.**
