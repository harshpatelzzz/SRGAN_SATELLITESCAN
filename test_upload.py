"""
Quick test script for image upload to API
"""
import requests
import sys
from pathlib import Path

def test_upload(image_path: str):
    """Test uploading an image to the API"""
    api_url = "http://localhost:8000/api/upscale"
    
    print(f"Testing upload of: {image_path}")
    print(f"API URL: {api_url}")
    
    # Check if file exists
    if not Path(image_path).exists():
        print(f"Error: File not found: {image_path}")
        return False
    
    try:
        # Upload file
        with open(image_path, 'rb') as f:
            files = {'file': f}
            print("Uploading...")
            response = requests.post(api_url, files=files, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            print("\n[SUCCESS] Upload successful!")
            print(f"  Original size: {data['original_size']}")
            print(f"  Upscaled size: {data['upscaled_size']}")
            print(f"  Processing time: {data['processing_time']:.2f}s")
            print(f"  Image data length: {len(data['upscaled_image_base64'])} chars")
            return True
        else:
            print(f"\n[FAILED] Status code: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Cannot connect to API server")
        print("  Make sure the API is running: cd api && python main.py")
        return False
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    # Test with first available test image
    test_image = Path(__file__).parent / "test_images" / "dataset_sample_1.png"
    
    if not test_image.exists():
        print(f"Test image not found: {test_image}")
        print("Available test images:")
        test_dir = Path(__file__).parent / "test_images"
        if test_dir.exists():
            for img in test_dir.glob("*.png"):
                print(f"  - {img.name}")
        sys.exit(1)
    
    success = test_upload(str(test_image))
    sys.exit(0 if success else 1)
