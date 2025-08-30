import os
import numpy as np
import cv2

from vision.detect import save_crop, UIDetector
from utils.config import load_config


def test_save_crop(tmp_path):
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    box = (10, 10, 60, 40)
    out = save_crop("test", img, box, str(tmp_path))
    assert out is not None
    assert os.path.exists(out)


def test_uidetector_basic():
    cfg = load_config()  # defaults
    d = UIDetector(cfg)
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    ui = d.detect_ui(img)
    assert "elixir" in ui
    assert "battle_button" in ui and isinstance(ui["battle_button"], tuple)
    assert "card_slots" in ui and len(ui["card_slots"]) == 4
    assert "safe_zone" in ui
    assert "upgrade_button" in ui and isinstance(ui["upgrade_button"], tuple)


def test_yolo_stub_and_rewards_ocr(monkeypatch):
    cfg = load_config()
    d = UIDetector(cfg)
    # Monkeypatch YOLO predictions
    def fake_predict(frame):
        return [
            {"label": "troop", "conf": 0.9, "box": (100, 100, 120, 140)},
            {"label": "button", "conf": 0.95, "box": (800, 900, 1000, 950)},
        ]
    monkeypatch.setattr(d, "_predict_yolo", fake_predict)
    # Monkeypatch OCR text to simulate gold/daily_wins
    def fake_read_text(_img):
        return ["50"]
    monkeypatch.setattr(d.ocr, "read_text", fake_read_text)
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    ui = d.detect_ui(img)
    assert "yolo" in ui and "troop" in ui["yolo"] and len(ui["yolo"]["troop"]) >= 1
    assert isinstance(ui.get("gold", 0), int)
    assert isinstance(ui.get("daily_wins", 0), int)
