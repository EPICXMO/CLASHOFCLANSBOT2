#!/usr/bin/env python3
"""
Test script to verify the local GPU setup is working correctly.
Run this after setup_local.py to ensure everything is configured properly.
"""

import sys
import numpy as np
from utils.config import load_config
from utils.hardware import detect_hardware, get_performance_settings, log_hardware_info

def test_basic_imports():
    """Test that all basic imports work."""
    print("🧪 Testing basic imports...")
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
        
        import yaml
        print("✓ PyYAML")
        
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_gpu_detection():
    """Test GPU detection and configuration."""
    print("\n🔍 Testing GPU detection...")
    
    hw_info = detect_hardware()
    print(f"Platform: {hw_info['platform']}")
    print(f"CPU Cores: {hw_info['cpu_count']}")
    print(f"CUDA Available: {hw_info['has_cuda']}")
    
    if hw_info['has_cuda']:
        print(f"CUDA Devices: {hw_info['cuda_devices']}")
        for i, name in enumerate(hw_info['cuda_device_names']):
            print(f"  GPU {i}: {name}")
        print(f"GPU Memory: {hw_info['gpu_memory']} GB")
    
    return True

def test_performance_optimization():
    """Test performance optimization settings."""
    print("\n⚡ Testing performance optimization...")
    
    config = load_config()
    settings = get_performance_settings(config)
    
    print(f"Device: {settings['device']}")
    print(f"YOLO Image Size: {settings['yolo_imgsz']}")
    print(f"YOLO Batch Size: {settings['yolo_batch_size']}")
    print(f"OCR GPU: {settings['ocr_use_gpu']}")
    print(f"Half Precision: {settings['half_precision']}")
    
    return True

def test_vision_system():
    """Test vision system with optimizations."""
    print("\n👁️ Testing vision system...")
    
    try:
        from vision.detect import UIDetector
        from vision.ocr import OCR
        
        config = load_config()
        
        # Test OCR
        ocr = OCR(enabled=True, lang="en", config=config)
        print(f"OCR Enabled: {ocr.enabled}")
        print(f"OCR GPU: {ocr.use_gpu}")
        
        # Test UIDetector
        detector = UIDetector(config)
        print(f"YOLO Enabled: {detector.enable_yolo}")
        print(f"YOLO Device: {detector.yolo_device}")
        
        # Test with dummy frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        ui = detector.detect_ui(frame)
        print(f"Detection completed: {len(ui)} UI elements")
        
        return True
        
    except Exception as e:
        print(f"❌ Vision test failed: {e}")
        return False

def test_ml_libraries():
    """Test ML libraries (PyTorch, etc.)."""
    print("\n🤖 Testing ML libraries...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available - {torch.cuda.device_count()} device(s)")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory // (1024**3)
                print(f"  GPU {i}: {name} ({memory} GB)")
        else:
            print("ℹ️ CUDA not available - using CPU")
            
        # Test basic operations
        x = torch.randn(2, 3)
        if torch.cuda.is_available():
            x_gpu = x.cuda()
            print("✓ GPU tensor operations working")
        
        return True
        
    except ImportError:
        print("⚠️ PyTorch not installed")
        return False
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False

def test_advanced_features():
    """Test advanced features like YOLO and PaddleOCR."""
    print("\n🚀 Testing advanced features...")
    
    # Test YOLO
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics YOLO available")
        
        # Try to load a model (without downloading if not present)
        try:
            model = YOLO("yolo11n.pt")
            print("✓ YOLO model loaded")
        except Exception:
            print("ℹ️ YOLO model not downloaded yet")
            
    except ImportError:
        print("⚠️ Ultralytics not installed")
    
    # Test PaddleOCR
    try:
        from paddleocr import PaddleOCR
        print("✓ PaddleOCR available")
    except ImportError:
        print("⚠️ PaddleOCR not installed")
    
    return True

def main():
    """Run all tests."""
    print("🚀 Local GPU Setup Test Suite")
    print("=" * 40)
    
    tests = [
        test_basic_imports,
        test_gpu_detection,
        test_performance_optimization,
        test_vision_system,
        test_ml_libraries,
        test_advanced_features,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 40)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ All {total} tests passed!")
        print("\n🎯 Your system is ready for high-performance bot operation!")
        if detect_hardware()['has_cuda']:
            print("🔥 GPU acceleration is configured and working!")
    else:
        print(f"⚠️ {passed}/{total} tests passed")
        print("Some features may not be available. Check the output above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)