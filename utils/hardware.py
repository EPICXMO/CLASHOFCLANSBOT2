"""Hardware detection and optimization utilities for local high-performance setup."""

import logging
import os
import platform
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def detect_hardware() -> Dict[str, Any]:
    """Detect available hardware and capabilities."""
    info = {
        "platform": platform.system(),
        "cpu_count": os.cpu_count(),
        "has_cuda": False,
        "cuda_devices": 0,
        "cuda_device_names": [],
        "recommended_device": "cpu",
        "gpu_memory": 0,
    }
    
    # Detect CUDA availability
    try:
        import torch
        info["has_cuda"] = torch.cuda.is_available()
        if info["has_cuda"]:
            info["cuda_devices"] = torch.cuda.device_count()
            info["cuda_device_names"] = [torch.cuda.get_device_name(i) for i in range(info["cuda_devices"])]
            if info["cuda_devices"] > 0:
                info["recommended_device"] = "cuda:0"
                # Get GPU memory for first device
                try:
                    info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory // (1024**3)  # GB
                except Exception:
                    info["gpu_memory"] = 0
    except ImportError:
        logger.debug("PyTorch not available for CUDA detection")
    
    # Try pynvml for more detailed GPU info
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0 and not info["cuda_device_names"]:
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                info["cuda_device_names"].append(name)
                if i == 0:  # Get memory for first device
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    info["gpu_memory"] = mem_info.total // (1024**3)  # GB
    except ImportError:
        logger.debug("pynvml not available for GPU detection")
    except Exception as e:
        logger.debug(f"Error detecting GPU with pynvml: {e}")
    
    return info


def get_optimal_device(config: Dict[str, Any]) -> str:
    """Get the optimal device based on config and hardware."""
    hw_config = config.get("hardware", {})
    use_gpu = hw_config.get("use_gpu", True)
    gpu_device = hw_config.get("gpu_device", "auto")
    
    if not use_gpu:
        return "cpu"
    
    if gpu_device != "auto":
        return gpu_device
    
    # Auto-detect best device
    hw_info = detect_hardware()
    if hw_info["has_cuda"] and hw_info["cuda_devices"] > 0:
        return "cuda:0"
    
    return "cpu"


def get_performance_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get optimized performance settings based on hardware and config."""
    hw_config = config.get("hardware", {})
    performance_mode = hw_config.get("performance_mode", "medium")
    
    hw_info = detect_hardware()
    device = get_optimal_device(config)
    is_gpu = device.startswith("cuda")
    
    settings = {
        "device": device,
        "yolo_imgsz": 320,
        "yolo_batch_size": 1,
        "ocr_use_gpu": False,
        "half_precision": False,
    }
    
    # Optimize based on performance mode and hardware
    if performance_mode == "high" and is_gpu:
        # High performance for powerful GPUs like RTX 3090
        if hw_info["gpu_memory"] >= 20:  # RTX 3090 has 24GB
            settings.update({
                "yolo_imgsz": 640,  # Higher resolution for better accuracy
                "yolo_batch_size": 4,
                "ocr_use_gpu": hw_config.get("enable_gpu_ocr", True),
                "half_precision": True,  # FP16 for speed
            })
        elif hw_info["gpu_memory"] >= 8:
            settings.update({
                "yolo_imgsz": 512,
                "yolo_batch_size": 2,
                "ocr_use_gpu": hw_config.get("enable_gpu_ocr", True),
            })
        else:
            settings.update({
                "yolo_imgsz": 416,
                "ocr_use_gpu": hw_config.get("enable_gpu_ocr", True),
            })
    elif performance_mode == "medium":
        settings.update({
            "yolo_imgsz": 416 if is_gpu else 320,
            "ocr_use_gpu": is_gpu and hw_config.get("enable_gpu_ocr", True),
        })
    # Low performance mode keeps defaults
    
    return settings


def log_hardware_info(config: Dict[str, Any]) -> None:
    """Log detected hardware information and optimization settings."""
    hw_info = detect_hardware()
    settings = get_performance_settings(config)
    
    logger.info("=== Hardware Detection ===")
    logger.info(f"Platform: {hw_info['platform']}")
    logger.info(f"CPU Cores: {hw_info['cpu_count']}")
    logger.info(f"CUDA Available: {hw_info['has_cuda']}")
    
    if hw_info["has_cuda"]:
        logger.info(f"CUDA Devices: {hw_info['cuda_devices']}")
        for i, name in enumerate(hw_info["cuda_device_names"]):
            logger.info(f"  GPU {i}: {name}")
        if hw_info["gpu_memory"] > 0:
            logger.info(f"GPU Memory: {hw_info['gpu_memory']} GB")
    
    logger.info("=== Optimization Settings ===")
    logger.info(f"Using Device: {settings['device']}")
    logger.info(f"YOLO Image Size: {settings['yolo_imgsz']}")
    logger.info(f"YOLO Batch Size: {settings['yolo_batch_size']}")
    logger.info(f"OCR GPU Enabled: {settings['ocr_use_gpu']}")
    logger.info(f"Half Precision: {settings['half_precision']}")


def create_optimized_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create an optimized config based on detected hardware."""
    optimized = base_config.copy()
    settings = get_performance_settings(base_config)
    
    # Update vision settings
    if "vision" in optimized:
        optimized["vision"]["yolo_imgsz"] = settings["yolo_imgsz"]
    
    # Add hardware-specific settings
    if "hardware" not in optimized:
        optimized["hardware"] = {}
    
    optimized["hardware"].update({
        "detected_device": settings["device"],
        "optimized_yolo_imgsz": settings["yolo_imgsz"],
        "use_half_precision": settings["half_precision"],
    })
    
    return optimized