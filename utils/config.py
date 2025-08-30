import os
from dataclasses import dataclass
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "resolution": {"width": 1920, "height": 1080},
    "adb": {"host": "127.0.0.1", "port": 5555, "connect_timeout_s": 5, "dry_run": True},
    "ocr": {
        "enabled": True,
        "lang": "en",
        "elixir_roi": [0.4, 0.92, 0.6, 0.98],
        "gold_roi": [0.85, 0.02, 0.95, 0.08],
        "daily_wins_roi": [0.65, 0.02, 0.78, 0.08],
    },
    "vision": {
        "save_unknown_crops": True,
        "crops_dir": "data/crops",
        "enable_yolo": True,
        "yolo_model": "yolo11n.pt",
        "yolo_conf_thresh": 0.35,
        "yolo_imgsz": 320,
        "battle_roi": [0.4, 0.85, 0.6, 0.92],
        "cards_centers": [[0.2, 0.92], [0.4, 0.92], [0.6, 0.92], [0.8, 0.92]],
        "safe_zone_roi": [0.1, 0.55, 0.9, 0.75],
        "rewards_roi_center": [0.3, 0.4, 0.7, 0.6],
        "rewards_roi_bottom": [0.4, 0.85, 0.6, 0.92],
        "upgrade_roi": [0.45, 0.55, 0.55, 0.65],
    },
    "rl": {
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "n_steps": 2048,
        "batch_size": 64,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
    },
    "logging": {"level": "INFO", "file": "logs/run.log", "max_bytes": 1048576, "backup_count": 3},
}


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    if not os.path.exists(path):
        return DEFAULT_CONFIG.copy()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # Merge shallowly with defaults
        cfg = DEFAULT_CONFIG.copy()
        for k, v in data.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                merged = cfg[k].copy()
                merged.update(v)
                cfg[k] = merged
            else:
                cfg[k] = v
        return cfg
    except Exception:
        return DEFAULT_CONFIG.copy()
