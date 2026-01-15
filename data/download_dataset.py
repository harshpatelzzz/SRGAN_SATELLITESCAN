"""
Dataset downloader for SRGAN
Downloads and prepares DIV2K (standard SR benchmark) or UC Merced (satellite imagery)
"""

import os
import sys
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from typing import Optional
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.config import Config
from utils.logger import setup_logger


class DatasetDownloader:
    """Download and prepare datasets for SRGAN training"""
    
    DIV2K_TRAIN_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
    DIV2K_VALID_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
    
    # UC Merced - multiple possible sources
    UC_MERCED_URLS = [
        "http://weecology.org/data/ucmerced/ucmerced.zip",
        "https://drive.google.com/uc?export=download&id=0B6l9LmQ9x5SdTjZxTkxKQzZtNkE",  # Alternative source
    ]
    
    def __init__(self, dataset_name: str = "div2k"):
        """
        Initialize dataset downloader
        
        Args:
            dataset_name: 'div2k' or 'ucmerced'
        """
        self.dataset_name = dataset_name.lower()
        self.logger = setup_logger("DatasetDownloader")
        Config.create_directories()
        
        # Create datasets directory
        self.datasets_dir = Config.PROJECT_ROOT / "datasets"
        self.datasets_dir.mkdir(exist_ok=True)
    
    def download_file(self, url: str, destination: Path, description: str = "file", urls: list = None):
        """
        Download a file with progress bar
        
        Args:
            url: Primary URL to download from
            destination: Path to save the file
            description: Description for progress bar
            urls: List of alternative URLs to try if primary fails
        """
        if destination.exists():
            self.logger.info(f"{description} already exists: {destination}")
            return True
        
        # Try primary URL first, then alternatives
        urls_to_try = [url]
        if urls:
            urls_to_try.extend(urls)
        
        for attempt_url in urls_to_try:
            self.logger.info(f"Downloading {description} from {attempt_url}...")
            self.logger.info(f"This may take several minutes. File will be saved to: {destination}")
            
            try:
                def show_progress(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    percent = min(downloaded * 100 / total_size, 100)
                    if block_num % 100 == 0:
                        print(f"\rProgress: {percent:.1f}%", end='', flush=True)
                
                urllib.request.urlretrieve(attempt_url, destination, show_progress)
                print()  # New line after progress
                self.logger.info(f"Downloaded {description} successfully!")
                return True
            except Exception as e:
                self.logger.warning(f"Failed to download from {attempt_url}: {e}")
                if destination.exists():
                    destination.unlink()  # Remove partial download
                continue
        
        self.logger.error(f"Failed to download {description} from all URLs")
        return False
    
    def extract_zip(self, zip_path: Path, extract_to: Path):
        """Extract ZIP file"""
        self.logger.info(f"Extracting {zip_path.name}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            self.logger.info("Extraction completed!")
            return True
        except Exception as e:
            self.logger.error(f"Failed to extract: {e}")
            return False
    
    def download_div2k(self) -> Optional[Path]:
        """
        Download and prepare DIV2K dataset
        
        Returns:
            Path to DIV2K HR images directory, or None if failed
        """
        self.logger.info("=" * 60)
        self.logger.info("Downloading DIV2K Dataset")
        self.logger.info("=" * 60)
        self.logger.info("DIV2K is the standard benchmark for super-resolution")
        self.logger.info("Contains 800 training images and 100 validation images")
        self.logger.info("=" * 60)
        
        div2k_dir = self.datasets_dir / "DIV2K"
        div2k_dir.mkdir(exist_ok=True)
        
        # Download training set
        train_zip = div2k_dir / "DIV2K_train_HR.zip"
        if not self.download_file(self.DIV2K_TRAIN_URL, train_zip, "DIV2K training set"):
            return None
        
        # Download validation set
        valid_zip = div2k_dir / "DIV2K_valid_HR.zip"
        if not self.download_file(self.DIV2K_VALID_URL, valid_zip, "DIV2K validation set"):
            return None
        
        # Extract training set
        train_extract = div2k_dir / "train"
        if not (train_extract / "DIV2K_train_HR").exists():
            if not self.extract_zip(train_zip, train_extract):
                return None
        
        # Extract validation set
        valid_extract = div2k_dir / "valid"
        if not (valid_extract / "DIV2K_valid_HR").exists():
            if not self.extract_zip(valid_zip, valid_extract):
                return None
        
        # Combine train and valid into single HR directory
        hr_dir = div2k_dir / "HR"
        hr_dir.mkdir(exist_ok=True)
        
        train_hr = train_extract / "DIV2K_train_HR"
        valid_hr = valid_extract / "DIV2K_valid_HR"
        
        # Copy training images
        if train_hr.exists():
            for img_file in train_hr.glob("*.png"):
                shutil.copy2(img_file, hr_dir / img_file.name)
        
        # Copy validation images
        if valid_hr.exists():
            for img_file in valid_hr.glob("*.png"):
                shutil.copy2(img_file, hr_dir / img_file.name)
        
        # Clean up ZIP files to save space
        if train_zip.exists():
            train_zip.unlink()
        if valid_zip.exists():
            valid_zip.unlink()
        
        # Clean up extracted folders (we only need HR folder)
        if train_extract.exists():
            shutil.rmtree(train_extract)
            self.logger.info("Removed train extraction folder to save space")
        if valid_extract.exists():
            shutil.rmtree(valid_extract)
            self.logger.info("Removed valid extraction folder to save space")
        
        self.logger.info(f"DIV2K dataset prepared successfully!")
        self.logger.info(f"HR images directory: {hr_dir}")
        self.logger.info(f"Total images: {len(list(hr_dir.glob('*.png')))}")
        
        return hr_dir
    
    def download_ucmerced(self) -> Optional[Path]:
        """
        Download and prepare UC Merced Land Use Dataset
        
        Returns:
            Path to UC Merced images directory, or None if failed
        """
        self.logger.info("=" * 60)
        self.logger.info("Downloading UC Merced Land Use Dataset")
        self.logger.info("=" * 60)
        self.logger.info("Contains 2100 aerial/satellite images (256x256)")
        self.logger.info("21 land-use classes, 100 images per class")
        self.logger.info("=" * 60)
        
        ucmerced_dir = self.datasets_dir / "UCMerced"
        ucmerced_dir.mkdir(exist_ok=True)
        
        # Download dataset (try multiple sources)
        dataset_zip = ucmerced_dir / "ucmerced.zip"
        if not self.download_file(self.UC_MERCED_URLS[0], dataset_zip, "UC Merced dataset", self.UC_MERCED_URLS[1:]):
            self.logger.warning("UC Merced download failed. You can manually download from:")
            self.logger.info("  https://drive.google.com/file/d/0B6l9LmQ9x5SdTjZxTkxKQzZtNkE/view")
            self.logger.info("  Or visit: http://weecology.org/data/ucmerced/")
            return None
        
        # Extract
        if not (ucmerced_dir / "UCMerced_LandUse").exists():
            if not self.extract_zip(dataset_zip, ucmerced_dir):
                return None
        
        # Collect all images into single directory
        hr_dir = ucmerced_dir / "HR"
        hr_dir.mkdir(exist_ok=True)
        
        source_dir = ucmerced_dir / "UCMerced_LandUse" / "Images"
        if source_dir.exists():
            for class_dir in source_dir.iterdir():
                if class_dir.is_dir():
                    for img_file in class_dir.glob("*.tif"):
                        # Copy with unique name
                        new_name = f"{class_dir.name}_{img_file.name}"
                        shutil.copy2(img_file, hr_dir / new_name)
        
        # Clean up ZIP file
        if dataset_zip.exists():
            dataset_zip.unlink()
        
        self.logger.info(f"UC Merced dataset prepared successfully!")
        self.logger.info(f"HR images directory: {hr_dir}")
        self.logger.info(f"Total images: {len(list(hr_dir.glob('*.tif')))}")
        
        return hr_dir
    
    def download(self) -> Optional[Path]:
        """
        Download the specified dataset
        
        Returns:
            Path to HR images directory, or None if failed
        """
        if self.dataset_name == "div2k":
            return self.download_div2k()
        elif self.dataset_name == "ucmerced":
            return self.download_ucmerced()
        else:
            self.logger.error(f"Unknown dataset: {self.dataset_name}")
            self.logger.info("Available datasets: 'div2k', 'ucmerced'")
            return None


def main():
    """Main function for dataset download"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download dataset for SRGAN')
    parser.add_argument('--dataset', type=str, default='div2k',
                       choices=['div2k', 'ucmerced'],
                       help='Dataset to download (default: div2k)')
    parser.add_argument('--auto-config', action='store_true',
                       help='Automatically update config.py with dataset path')
    
    args = parser.parse_args()
    
    # Download dataset
    downloader = DatasetDownloader(args.dataset)
    hr_dir = downloader.download()
    
    if hr_dir and args.auto_config:
        # Update config.py
        config_path = Config.PROJECT_ROOT / "utils" / "config.py"
        if config_path.exists():
            with open(config_path, 'r') as f:
                content = f.read()
            
            # Update HR_IMAGE_DIR
            import re
            pattern = r'HR_IMAGE_DIR\s*=\s*[^\n]*'
            replacement = f"HR_IMAGE_DIR = r'{hr_dir}'"
            content = re.sub(pattern, replacement, content)
            
            with open(config_path, 'w') as f:
                f.write(content)
            
            downloader.logger.info(f"Updated config.py with dataset path: {hr_dir}")
    
    if hr_dir:
        downloader.logger.info("=" * 60)
        downloader.logger.info("Dataset download completed!")
        downloader.logger.info(f"Set HR_IMAGE_DIR in utils/config.py to: {hr_dir}")
        downloader.logger.info("=" * 60)
    else:
        downloader.logger.error("Dataset download failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
