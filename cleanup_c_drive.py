"""
Safe cleanup script for C drive
Removes temporary files and caches
"""

import os
import shutil
from pathlib import Path

def get_size(path):
    """Get total size of directory in GB"""
    if not Path(path).exists():
        return 0, 0
    total = 0
    count = 0
    try:
        for f in Path(path).rglob('*'):
            if f.is_file():
                total += f.stat().st_size
                count += 1
    except (PermissionError, OSError):
        pass
    return total / (1024**3), count

def cleanup_c_drive():
    """Clean up C drive safely"""
    user = os.path.expanduser('~')
    
    print("=" * 60)
    print("C DRIVE CLEANUP - Safe Cleanup Only")
    print("=" * 60)
    
    total_freed = 0
    
    # 1. Clean temp files (SAFE)
    temp_dir = os.path.join(user, "AppData", "Local", "Temp")
    if Path(temp_dir).exists():
        size_before, _ = get_size(temp_dir)
        print(f"\n1. Cleaning temp files...")
        print(f"   Before: {size_before:.2f} GB")
        
        try:
            for item in Path(temp_dir).iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                except (PermissionError, OSError):
                    pass  # Skip files in use
            
            size_after, _ = get_size(temp_dir)
            freed = size_before - size_after
            total_freed += freed
            print(f"   After: {size_after:.2f} GB")
            print(f"   [FREED] {freed:.2f} GB")
        except Exception as e:
            print(f"   Error: {e}")
    
    # 2. Clean pip cache (SAFE - will re-download if needed)
    pip_cache = os.path.join(user, ".cache", "pip")
    if Path(pip_cache).exists():
        size_before, _ = get_size(pip_cache)
        if size_before > 0:
            print(f"\n2. Cleaning pip cache...")
            print(f"   Size: {size_before:.2f} GB")
            try:
                shutil.rmtree(pip_cache)
                total_freed += size_before
                print(f"   [FREED] {size_before:.2f} GB")
            except Exception as e:
                print(f"   Error: {e}")
    
    # 3. Clean PyTorch hub cache (SAFE - will re-download models if needed)
    torch_cache = os.path.join(user, ".cache", "torch")
    if Path(torch_cache).exists():
        size_before, _ = get_size(torch_cache)
        if size_before > 0:
            print(f"\n3. Cleaning PyTorch cache...")
            print(f"   Size: {size_before:.2f} GB")
            print(f"   WARNING: This will delete cached models (VGG19, etc.)")
            print(f"   They will re-download when needed.")
            try:
                shutil.rmtree(torch_cache)
                total_freed += size_before
                print(f"   [FREED] {size_before:.2f} GB")
            except Exception as e:
                print(f"   Error: {e}")
    
    # 4. Windows temp files
    windows_temp = "C:\\Windows\\Temp"
    if Path(windows_temp).exists():
        size_before, _ = get_size(windows_temp)
        if size_before > 0.1:  # Only if > 100MB
            print(f"\n4. Windows temp files...")
            print(f"   Size: {size_before:.2f} GB")
            print(f"   (Skipping - requires admin rights)")
    
    print("\n" + "=" * 60)
    print(f"TOTAL SPACE FREED: {total_freed:.2f} GB")
    print("=" * 60)
    
    # Check C drive status
    try:
        stat = shutil.disk_usage("C:\\")
        free_gb = stat.free / (1024**3)
        print(f"\nC Drive Free Space: {free_gb:.2f} GB")
    except:
        pass

if __name__ == "__main__":
    cleanup_c_drive()
