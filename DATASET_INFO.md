# Dataset Information

## Recommended Dataset: DIV2K

**DIV2K** is the **best choice** for super-resolution research and is the standard benchmark used in academic papers.

### Why DIV2K?

1. **Standard Benchmark**: Used in all major super-resolution papers (SRGAN, ESRGAN, etc.)
2. **High Quality**: 800 training + 100 validation images, professionally curated
3. **Diverse Content**: Natural images, textures, faces, objects - perfect for generalization
4. **Large Size**: Images are high-resolution (up to 2048×2048), providing many patches
5. **Well-Established**: Results are comparable with published research

### Download DIV2K

```bash
python main.py download-dataset --dataset div2k --auto-config
```

**Size**: ~7GB (downloads both training and validation sets)

**Source**: https://data.vision.ee.ethz.ch/cvl/DIV2K/

---

## Alternative: UC Merced Land Use Dataset

**UC Merced** is specifically designed for remote sensing and satellite imagery applications.

### Why UC Merced?

1. **Satellite-Specific**: Aerial/satellite images (256×256 each)
2. **Land Use Classification**: 21 classes (agricultural, airplane, buildings, etc.)
3. **Smaller Size**: 2100 images total, faster download (~200MB)
4. **Domain-Specific**: Better for satellite imagery applications

### Download UC Merced

```bash
python main.py download-dataset --dataset ucmerced --auto-config
```

**Size**: ~200MB

**Source**: http://weecology.org/data/ucmerced/

---

## Custom Dataset

You can also use your own high-resolution satellite images:

1. Place images in a directory
2. Edit `utils/config.py`:
   ```python
   HR_IMAGE_DIR = "/path/to/your/images"
   ```

**Supported formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`

**Note**: Images will be automatically tiled into 256×256 patches during training.

---

## Dataset Statistics

### DIV2K
- **Training Images**: 800
- **Validation Images**: 100
- **Image Sizes**: Variable (typically 1024×1024 to 2048×2048)
- **Patches Generated**: ~50,000+ (depending on image sizes)
- **Best For**: General super-resolution, research, benchmarking

### UC Merced
- **Total Images**: 2,100
- **Image Size**: 256×256 (fixed)
- **Classes**: 21 land-use categories
- **Patches Generated**: ~2,100 (1 patch per image)
- **Best For**: Satellite/aerial imagery applications

---

## Recommendation

**For academic/research purposes**: Use **DIV2K** - it's the standard benchmark and will give you results comparable to published papers.

**For satellite-specific applications**: Use **UC Merced** if you need domain-specific satellite imagery.

**For production/custom use**: Use your own dataset with relevant satellite imagery.

---

## Automatic Dataset Setup

The download script automatically:
1. Downloads the dataset
2. Extracts ZIP files
3. Organizes images into HR directory
4. Updates `config.py` (if `--auto-config` flag is used)
5. Cleans up ZIP files to save space

You're ready to train immediately after download!
