#!/usr/bin/env python3
"""
Setup script for high-performance local Clash of Clans bot with GPU acceleration.
Optimized for powerful hardware like NVIDIA RTX 3090.
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def run_command(cmd, check=True):
    """Run a command and return the result."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return e

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")

def detect_gpu():
    """Detect NVIDIA GPU and CUDA availability."""
    try:
        # Check nvidia-smi
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ NVIDIA GPU detected")
            print(result.stdout.split('\n')[0:3])  # Show header lines
            return True
        else:
            print("⚠ No NVIDIA GPU detected")
            return False
    except Exception:
        print("⚠ nvidia-smi not found - GPU detection failed")
        return False

def install_requirements():
    """Install the appropriate requirements based on GPU availability."""
    has_gpu = detect_gpu()
    
    if has_gpu:
        print("\n📦 Installing GPU-accelerated requirements...")
        # Install PyTorch with CUDA first
        if platform.system() == "Windows":
            torch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        else:
            torch_cmd = "pip install torch torchvision torchaudio"
        
        run_command(torch_cmd)
        
        # Install other GPU requirements
        run_command("pip install -r requirements-gpu.txt")
    else:
        print("\n📦 Installing CPU-only requirements...")
        run_command("pip install -r requirements.txt")

def setup_config():
    """Set up configuration files."""
    print("\n⚙️ Setting up configuration...")
    
    # Copy GPU config if it doesn't exist
    if not os.path.exists("config.yaml"):
        if os.path.exists("config-gpu.yaml"):
            shutil.copy("config-gpu.yaml", "config.yaml")
            print("✓ Created config.yaml from GPU template")
        else:
            print("⚠ No config template found")
    else:
        print("✓ config.yaml already exists")

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    dirs = ["logs", "data/crops", "checkpoints", "logs/episodes"]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {dir_path}")

def check_adb():
    """Check if ADB is available."""
    print("\n🔌 Checking ADB availability...")
    try:
        result = subprocess.run("adb version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ ADB is available")
            print(result.stdout.split('\n')[0])
        else:
            print("⚠ ADB not found - install Android SDK Platform Tools")
    except Exception:
        print("⚠ ADB not found - install Android SDK Platform Tools")

def test_installation():
    """Test the installation."""
    print("\n🧪 Testing installation...")
    try:
        # Test imports
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✓ CUDA available - {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠ CUDA not available - using CPU")
            
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
        
        # Test hardware detection
        from utils.hardware import detect_hardware, log_hardware_info
        from utils.config import load_config
        
        config = load_config()
        print("✓ Configuration loaded")
        
        hw_info = detect_hardware()
        print("✓ Hardware detection working")
        
        print("\n🎯 Hardware Summary:")
        print(f"  Platform: {hw_info['platform']}")
        print(f"  CPU Cores: {hw_info['cpu_count']}")
        print(f"  CUDA Available: {hw_info['has_cuda']}")
        if hw_info['has_cuda']:
            print(f"  CUDA Devices: {hw_info['cuda_devices']}")
            for name in hw_info['cuda_device_names']:
                print(f"    {name}")
            print(f"  GPU Memory: {hw_info['gpu_memory']} GB")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

def main():
    """Main setup function."""
    print("🚀 Clash of Clans Bot - High Performance Local Setup")
    print("=" * 50)
    
    check_python_version()
    create_directories()
    install_requirements()
    setup_config()
    check_adb()
    
    if test_installation():
        print("\n✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Start BlueStacks or Android emulator")
        print("2. Enable ADB: adb connect 127.0.0.1:5555")
        print("3. Run the bot: python main.py --mode play")
        print("\nFor training: python train.py --timesteps 2000000")
    else:
        print("\n❌ Setup completed with errors - check the output above")
        sys.exit(1)

if __name__ == "__main__":
    main()