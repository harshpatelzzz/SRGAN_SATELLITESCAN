# UC Merced Dataset - Alternative Download

## ⚠️ Automatic Download Failed

The automatic download for UC Merced failed because the URLs are outdated.

## 🔄 Alternative: Use DIV2K Subset (Recommended)

**Already configured!** The system is now using DIV2K subset mode:
- **5,000 patches** (vs 50,000+ full dataset)
- **5-10x faster training**
- **Still very good quality**
- **No download needed** (uses existing DIV2K)

## 📥 Manual UC Merced Download (Optional)

If you still want UC Merced, you can download manually:

### Option 1: Official Source
1. Visit: http://weecology.org/data/ucmerced/
2. Download the dataset
3. Extract to: `datasets/UCMerced/`
4. Update `utils/config.py`:
   ```python
   HR_IMAGE_DIR = r"Z:\AIMLLABEL\datasets\UCMerced\UCMerced_LandUse\Images"
   ```

### Option 2: Alternative Sources
- Search for "UC Merced Land Use Dataset" on Google
- Check academic repositories
- Look for mirror sites

### UC Merced Structure
After extraction, the dataset should have:
```
UCMerced/
└── UCMerced_LandUse/
    └── Images/
        ├── agricultural/
        ├── airplane/
        ├── buildings/
        └── ... (21 classes total)
```

## ✅ Current Setup (Recommended)

**DIV2K Subset Mode** is already active:
- Fast training (5,000 patches)
- Good quality
- No additional download needed

You can start training immediately!
