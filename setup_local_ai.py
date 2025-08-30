#!/usr/bin/env python3
"""
Local AI Setup Script for RTX 3090
Installs all dependencies needed for local LLaVA vision models
"""

import subprocess
import sys
import platform
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status"""
    try:
        logger.info("Installing %s...", description)
        result = subprocess.run(cmd.split(), check=True, capture_output=True, text=True)
        logger.info("✅ %s installed successfully", description)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("❌ Failed to install %s: %s", description, e.stderr)
        return False


def install_pytorch_cuda():
    """Install PyTorch with CUDA support for RTX 3090"""
    logger.info("🚀 Installing PyTorch with CUDA support for RTX 3090...")
    
    # CUDA 11.8 index for compatibility with RTX 3090
    pytorch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    
    if not run_command(pytorch_cmd, "PyTorch with CUDA 11.8"):
        # Fallback to CPU version
        logger.warning("CUDA install failed, falling back to CPU version")
        return run_command("pip install torch torchvision torchaudio", "PyTorch (CPU)")
    
    return True


def install_ai_dependencies():
    """Install AI vision dependencies"""
    dependencies = [
        ("pip install transformers>=4.35.0", "Transformers"),
        ("pip install accelerate>=0.24.0", "Accelerate"),
        ("pip install bitsandbytes>=0.41.0", "BitsAndBytes"),
        ("pip install scipy", "SciPy"),
        ("pip install datasets", "Datasets"),
        ("pip install sentencepiece", "SentencePiece"),
        ("pip install protobuf", "Protobuf"),
    ]
    
    success_count = 0
    for cmd, desc in dependencies:
        if run_command(cmd, desc):
            success_count += 1
    
    return success_count == len(dependencies)


def test_installation():
    """Test if the AI vision setup is working"""
    logger.info("🧪 Testing local AI installation...")
    
    try:
        import torch
        logger.info("✅ PyTorch version: %s", torch.__version__)
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info("✅ CUDA available - GPU: %s (%.1f GB)", gpu_name, gpu_memory)
            
            if "3090" in gpu_name:
                logger.info("🎯 RTX 3090 detected - perfect for LLaVA models!")
        else:
            logger.warning("⚠️  CUDA not available - will use CPU (slower)")
        
        # Test transformers
        from transformers import LlavaNextProcessor
        logger.info("✅ Transformers available")
        
        logger.info("🎉 Local AI setup complete!")
        logger.info("🚀 Ready to run: python intelligent_main.py --mode local")
        
        return True
        
    except ImportError as e:
        logger.error("❌ Installation test failed: %s", e)
        return False


def main():
    """Main setup function"""
    logger.info("🤖 Setting up Local AI for Clash Royale Bot")
    logger.info("🎯 Optimized for RTX 3090 with 24GB VRAM")
    logger.info("💡 This will enable free, offline AI vision analysis")
    print()
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8+ required. Current: %s", platform.python_version())
        return False
    
    logger.info("✅ Python version: %s", platform.python_version())
    
    # Install PyTorch with CUDA
    if not install_pytorch_cuda():
        logger.error("❌ PyTorch installation failed")
        return False
    
    # Install AI dependencies
    if not install_ai_dependencies():
        logger.error("❌ Some AI dependencies failed to install")
        return False
    
    # Test installation
    if not test_installation():
        logger.error("❌ Installation test failed")
        return False
    
    print()
    logger.info("🎉 Local AI setup completed successfully!")
    print()
    logger.info("Next steps:")
    logger.info("1. Run: python intelligent_main.py --mode local")
    logger.info("2. The first run will download LLaVA model (~13GB)")
    logger.info("3. Enjoy free, fast AI vision analysis!")
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)