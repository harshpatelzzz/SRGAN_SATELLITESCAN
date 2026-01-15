"""
Cleanup script to free up disk space
Removes duplicate dataset folders and cache files
"""

from pathlib import Path
import shutil

def get_size(path):
    """Get total size of directory in GB"""
    total = 0
    for f in Path(path).rglob('*'):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024**3)  # Convert to GB

def cleanup():
    """Clean up unnecessary files"""
    project_root = Path(__file__).parent
    
    print("=" * 60)
    print("DISK CLEANUP - Freeing up space")
    print("=" * 60)
    
    # 1. Remove duplicate dataset folders (keep only HR)
    div2k_dir = project_root / "datasets" / "DIV2K"
    if div2k_dir.exists():
        train_dir = div2k_dir / "train"
        valid_dir = div2k_dir / "valid"
        
        if train_dir.exists():
            size = get_size(train_dir)
            print(f"\n1. Removing train folder: {size:.2f} GB")
            shutil.rmtree(train_dir)
            print("   [DELETED]")
        
        if valid_dir.exists():
            size = get_size(valid_dir)
            print(f"\n2. Removing valid folder: {size:.2f} GB")
            shutil.rmtree(valid_dir)
            print("   [DELETED]")
    
    # 2. Remove Python cache files
    print("\n3. Removing Python cache files (__pycache__)...")
    cache_dirs = list(project_root.rglob("__pycache__"))
    total_cache_size = 0
    for cache_dir in cache_dirs:
        size = get_size(cache_dir)
        total_cache_size += size
        shutil.rmtree(cache_dir)
    print(f"   [REMOVED] {len(cache_dirs)} cache directories ({total_cache_size:.2f} GB)")
    
    # 3. Clean old logs (keep only latest 3)
    print("\n4. Cleaning old log files...")
    logs_dir = project_root / "logs"
    if logs_dir.exists():
        log_files = sorted(logs_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
        if len(log_files) > 3:
            for old_log in log_files[3:]:
                size = old_log.stat().st_size / (1024**2)  # MB
                print(f"   Removing: {old_log.name} ({size:.2f} MB)")
                old_log.unlink()
            print(f"   [KEPT] Latest 3 logs")
    
    # 4. Show final disk usage
    print("\n" + "=" * 60)
    print("CLEANUP COMPLETE!")
    print("=" * 60)
    
    if div2k_dir.exists():
        hr_size = get_size(div2k_dir / "HR")
        print(f"\nRemaining dataset (HR folder): {hr_size:.2f} GB")
        print(f"Total space freed: ~{size * 2:.2f} GB (estimated)")

if __name__ == "__main__":
    cleanup()
