# Test Images for SRGAN

## Location
All test images are stored in: `test_images/`

## Available Test Images

The following images have been prepared from your DIV2K dataset:

1. **dataset_sample_1.png** (4.72 MB)
2. **dataset_sample_2.png** (6.78 MB)
3. **dataset_sample_3.png** (5.03 MB)
4. **dataset_sample_4.png** (2.98 MB)
5. **dataset_sample_5.png** (5.71 MB)

These are high-quality images suitable for testing the SRGAN super-resolution model.

## How to Use Test Images

### Option 1: Via Web Interface (Easiest)

1. Open http://localhost:3000 in your browser
2. Click "Upload Image" or drag and drop
3. Select any image from `test_images/` folder
4. Click "Upscale Image"
5. View the before/after comparison with the slider

### Option 2: Via Command Line

```bash
# Upscale a single image
python main.py upscale --image test_images/dataset_sample_1.png --model checkpoints/generator_final.pth

# The upscaled image will be saved in results/inference/
```

### Option 3: Via API

```bash
# Using curl
curl -X POST "http://localhost:8000/api/upscale" \
  -F "file=@test_images/dataset_sample_1.png"

# Or use the interactive API docs
# Open http://localhost:8000/docs
```

### Option 4: Create Low-Resolution Test Images

To test the full pipeline (LR → HR), you can create low-resolution versions:

```python
from PIL import Image
from data.degradation import DegradationPipeline

# Load HR image
hr_image = Image.open("test_images/dataset_sample_1.png")

# Create degradation pipeline
degradation = DegradationPipeline(
    blur_sigma=1.2,
    noise_std=0.01,
    scale_factor=4
)

# Create LR version
lr_image = degradation(hr_image)
lr_image.save("test_images/dataset_sample_1_LR.png")

# Now upscale the LR image
python main.py upscale --image test_images/dataset_sample_1_LR.png --model checkpoints/generator_final.pth
```

## Adding More Test Images

### From Your Dataset

Run the download script again:
```bash
python download_test_images.py
```

### From Your Own Images

Simply copy any image (JPG, PNG, BMP, TIF) to the `test_images/` folder:

```bash
# Copy your own satellite/aerial images
copy "path\to\your\image.jpg" test_images\my_test_image.jpg
```

## Image Requirements

- **Format**: JPG, PNG, BMP, TIF, TIFF
- **Size**: Any size (will be processed automatically)
- **Channels**: RGB (3 channels) - grayscale will be converted
- **Recommended**: Images with clear features, textures, or structures work best

## Tips for Best Results

1. **Use images with clear features**: Buildings, roads, landscapes work well
2. **Avoid very dark images**: The model works better with well-lit scenes
3. **Test different image types**: Urban, agricultural, natural landscapes
4. **Compare with bicubic**: The model should outperform simple interpolation

## Expected Output

After upscaling, you'll get:
- **4× resolution increase**: 64×64 → 256×256, 128×128 → 512×512, etc.
- **Improved detail**: Sharper edges, better textures
- **Higher PSNR/SSIM**: Better than bicubic interpolation

## Troubleshooting

**Image too large?**
- The model can handle large images, but processing time increases
- For very large images (>2048×2048), consider cropping first

**Out of memory?**
- Use smaller images or reduce batch size in config
- Process images one at a time

**Model not found?**
- Make sure you've trained the model first:
  ```bash
  python main.py pretrain
  python main.py train
  ```
