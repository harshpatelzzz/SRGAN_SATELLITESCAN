"""
Download sample satellite/remote sensing images for testing SRGAN
Downloads free stock images suitable for super-resolution testing
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import sys

# Test images directory
TEST_IMAGES_DIR = Path(__file__).parent / "test_images"
TEST_IMAGES_DIR.mkdir(exist_ok=True)

# Sample satellite/remote sensing images from free sources
# Using Unsplash API for high-quality free images (satellite/aerial themes)
TEST_IMAGES = [
    {
        "name": "satellite_urban.jpg",
        "url": "https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=800&q=80",
        "description": "Urban satellite view"
    },
    {
        "name": "satellite_agricultural.jpg",
        "url": "https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=800&q=80",
        "description": "Agricultural fields"
    },
    {
        "name": "satellite_coastline.jpg",
        "url": "https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=800&q=80",
        "description": "Coastline view"
    },
    {
        "name": "satellite_mountains.jpg",
        "url": "https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=800&q=80",
        "description": "Mountain terrain"
    },
    {
        "name": "satellite_forest.jpg",
        "url": "https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=800&q=80",
        "description": "Forest area"
    }
]

# Alternative: Use direct image URLs from public datasets
# These are sample images from public satellite imagery sources
ALTERNATIVE_IMAGES = [
    {
        "name": "test_satellite_1.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/View_of_Earth_during_Apollo_17.jpg/800px-View_of_Earth_during_Apollo_17.jpg",
        "description": "Earth view from space"
    },
    {
        "name": "test_satellite_2.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/83/Earth_Eastern_Hemisphere.jpg/800px-Earth_Eastern_Hemisphere.jpg",
        "description": "Eastern hemisphere satellite view"
    },
    {
        "name": "test_satellite_3.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f0/MODIS_Satellite_Image_of_the_Great_Lakes.jpg/800px-MODIS_Satellite_Image_of_the_Great_Lakes.jpg",
        "description": "Great Lakes satellite image"
    }
]


def download_image(url: str, save_path: Path, description: str = ""):
    """Download an image from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=description) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        
        return True
    except Exception as e:
        print(f"Error downloading {save_path.name}: {e}")
        return False


def create_sample_from_dataset():
    """Create sample test images from the DIV2K dataset if available"""
    from utils.config import Config
    
    hr_dir = Path(Config.HR_IMAGE_DIR)
    if hr_dir.exists():
        print(f"\nCreating test samples from dataset: {hr_dir}")
        image_files = list(hr_dir.glob("*.png")) + list(hr_dir.glob("*.jpg"))
        
        if image_files:
            # Copy first 5 images as test samples
            for i, img_path in enumerate(image_files[:5]):
                dest = TEST_IMAGES_DIR / f"dataset_sample_{i+1}.png"
                if not dest.exists():
                    import shutil
                    shutil.copy2(img_path, dest)
                    print(f"  [OK] Copied: {dest.name}")
            return True
    return False


def main():
    """Download test images"""
    print("=" * 60)
    print("Downloading Test Images for SRGAN")
    print("=" * 60)
    print(f"\nOutput directory: {TEST_IMAGES_DIR}\n")
    
    # Try to create samples from existing dataset first
    print("Step 1: Creating samples from existing dataset...")
    dataset_samples = create_sample_from_dataset()
    
    # Download alternative images
    print("\nStep 2: Downloading sample satellite images...")
    downloaded = 0
    
    for img_info in ALTERNATIVE_IMAGES:
        save_path = TEST_IMAGES_DIR / img_info["name"]
        
        if save_path.exists():
            print(f"  [SKIP] Skipping {img_info['name']} (already exists)")
            continue
        
        print(f"  Downloading {img_info['name']}...")
        if download_image(img_info["url"], save_path, img_info["description"]):
            downloaded += 1
            print(f"    [OK] Saved: {save_path}")
        else:
            print(f"    [FAIL] Failed to download")
    
    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    
    all_images = list(TEST_IMAGES_DIR.glob("*.jpg")) + list(TEST_IMAGES_DIR.glob("*.png"))
    
    if all_images:
        print(f"\n[OK] Total test images available: {len(all_images)}")
        print("\nTest images location:")
        print(f"  {TEST_IMAGES_DIR}")
        print("\nAvailable images:")
        for img in all_images:
            size_mb = img.stat().st_size / (1024 * 1024)
            print(f"  • {img.name} ({size_mb:.2f} MB)")
        
        print("\n" + "=" * 60)
        print("How to Use:")
        print("=" * 60)
        print("\n1. Via Command Line:")
        print(f"   python main.py upscale --image test_images/{all_images[0].name} --model checkpoints/generator_final.pth")
        print("\n2. Via Web Interface:")
        print("   Open http://localhost:3000 and upload any image from test_images/")
        print("\n3. Via API:")
        print("   POST http://localhost:8000/api/upscale")
        print("   Form data: file=<image from test_images/>")
    else:
        print("\n[WARN] No images downloaded. Check your internet connection.")
        print("\nAlternative: Use images from your dataset:")
        print(f"  {Path(Config.HR_IMAGE_DIR) if hasattr(Config, 'HR_IMAGE_DIR') else 'N/A'}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
