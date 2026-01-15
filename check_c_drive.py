"""
Check C drive space usage, especially Python/PyTorch related files
"""

import os
from pathlib import Path
import shutil

def get_size(path):
    """Get total size of directory in GB"""
    if not Path(path).exists():
        return 0
    total = 0
    count = 0
    try:
        for f in Path(path).rglob('*'):
            if f.is_file():
                total += f.stat().st_size
                count += 1
    except (PermissionError, OSError):
        pass
    return total / (1024**3), count  # Convert to GB

def check_c_drive():
    """Check what's using space on C drive"""
    user = os.path.expanduser('~')
    
    print("=" * 60)
    print("CHECKING C DRIVE SPACE USAGE")
    print("=" * 60)
    
    locations = {
        "Python Packages": os.path.join(user, "AppData", "Roaming", "Python"),
        "PyTorch Cache": os.path.join(user, ".cache", "torch"),
        "PyTorch Models": os.path.join(user, ".torch"),
        "Temp Files": os.path.join(user, "AppData", "Local", "Temp"),
        "Python Cache": os.path.join(user, ".cache", "pip"),
    }
    
    total_size = 0
    for name, path in locations.items():
        result = get_size(path)
        if isinstance(result, tuple):
            size, count = result
        else:
            size = result
            count = 0
        if size > 0:
            print(f"\n{name}:")
            print(f"  Path: {path}")
            print(f"  Size: {size:.2f} GB")
            if count > 0:
                print(f"  Files: {count:,}")
            total_size += size
    
    print("\n" + "=" * 60)
    print(f"TOTAL PYTHON-RELATED: {total_size:.2f} GB")
    print("=" * 60)
    
    # Check C drive free space
    try:
        stat = shutil.disk_usage("C:\\")
        free_gb = stat.free / (1024**3)
        used_gb = stat.used / (1024**3)
        total_gb = stat.total / (1024**3)
        print(f"\nC Drive Status:")
        print(f"  Total: {total_gb:.2f} GB")
        print(f"  Used: {used_gb:.2f} GB")
        print(f"  Free: {free_gb:.2f} GB")
        print(f"  Free %: {(free_gb/total_gb)*100:.1f}%")
    except:
        pass

if __name__ == "__main__":
    check_c_drive()
