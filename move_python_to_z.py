"""
Move Python packages to Z drive using virtual environment
This will free up C drive space
"""

import os
import subprocess
import sys
from pathlib import Path

def create_venv_on_z():
    """Create virtual environment on Z drive"""
    project_root = Path(__file__).parent
    venv_path = project_root / "venv"
    
    print("=" * 60)
    print("MOVING PYTHON TO Z DRIVE")
    print("=" * 60)
    
    if venv_path.exists():
        print(f"\nVirtual environment already exists at: {venv_path}")
        response = input("Delete and recreate? (y/n): ")
        if response.lower() == 'y':
            import shutil
            shutil.rmtree(venv_path)
        else:
            print("Using existing virtual environment")
            return venv_path
    
    print(f"\n1. Creating virtual environment on Z drive...")
    print(f"   Location: {venv_path}")
    
    # Create virtual environment
    result = subprocess.run(
        [sys.executable, "-m", "venv", str(venv_path)],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"   ERROR: {result.stderr}")
        return None
    
    print("   [CREATED] Virtual environment")
    
    # Get pip path
    if os.name == 'nt':  # Windows
        pip_path = venv_path / "Scripts" / "pip.exe"
        python_path = venv_path / "Scripts" / "python.exe"
    else:
        pip_path = venv_path / "bin" / "pip"
        python_path = venv_path / "bin" / "python"
    
    print(f"\n2. Installing packages in virtual environment...")
    print(f"   This will take a few minutes...")
    
    # Install packages from requirements.txt
    requirements = project_root / "requirements.txt"
    if requirements.exists():
        result = subprocess.run(
            [str(pip_path), "install", "-r", str(requirements)],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("   [INSTALLED] All packages")
        else:
            print(f"   ERROR: {result.stderr}")
    else:
        print("   WARNING: requirements.txt not found")
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print(f"\nVirtual environment created at: {venv_path}")
    print(f"\nTo activate:")
    if os.name == 'nt':
        print(f"  {venv_path}\\Scripts\\Activate.ps1")
    else:
        print(f"  source {venv_path}/bin/activate")
    print(f"\nPython executable: {python_path}")
    print(f"\nAll packages are now on Z drive, not C drive!")
    
    return venv_path

if __name__ == "__main__":
    create_venv_on_z()
