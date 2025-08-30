import os
import time
import logging
from typing import Dict, Tuple, Optional, List

import numpy as np
import cv2

from .ocr import OCR
from . import templates

logger = logging.getLogger(__name__)


def save_crop(label: str, img: np.ndarray, box: Tuple[int, int, int, int], crops_dir: str) -> Optional[str]:
    try:
        os.makedirs(crops_dir, exist_ok=True)
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        crop = img[y1:y2, x1:x2]
        ts = int(time.time() * 1000)
        path = os.path.join(crops_dir, f"{label}_{ts}.png")
        cv2.imwrite(path, crop)
        return path
    except Exception as e:
        logger.debug("Save crop error: %s", e)
        return None


class UIDetector:
    def __init__(self, cfg):
        ocr_cfg = cfg.get("ocr", {})
        self.ocr = OCR(enabled=ocr_cfg.get("enabled", True), lang=ocr_cfg.get("lang", "en"))
        self.elixir_roi = tuple(ocr_cfg.get("elixir_roi", [0.4, 0.92, 0.6, 0.98]))  # type: ignore
        self.gold_roi = tuple(ocr_cfg.get("gold_roi", [0.85, 0.02, 0.95, 0.08]))  # type: ignore
        self.daily_wins_roi = tuple(ocr_cfg.get("daily_wins_roi", [0.65, 0.02, 0.78, 0.08]))  # type: ignore
        vis_cfg = cfg.get("vision", {})
        self.save_unknown = bool(vis_cfg.get("save_unknown_crops", True))
        self.crops_dir = str(vis_cfg.get("crops_dir", "data/crops"))
        self.cards_centers_norm: List[List[float]] = list(vis_cfg.get("cards_centers", [[0.2, 0.92], [0.4, 0.92], [0.6, 0.92], [0.8, 0.92]]))
        self.safe_zone_roi = list(vis_cfg.get("safe_zone_roi", [0.1, 0.55, 0.9, 0.75]))
        self.battle_roi = list(vis_cfg.get("battle_roi", [0.4, 0.85, 0.6, 0.92]))
        self.rewards_roi_center = list(vis_cfg.get("rewards_roi_center", [0.3, 0.4, 0.7, 0.6]))
        self.rewards_roi_bottom = list(vis_cfg.get("rewards_roi_bottom", [0.4, 0.85, 0.6, 0.92]))
        self.upgrade_roi = list(vis_cfg.get("upgrade_roi", [0.45, 0.55, 0.55, 0.65]))

        # YOLO
        self.enable_yolo = bool(vis_cfg.get("enable_yolo", True))
        self.yolo_conf = float(vis_cfg.get("yolo_conf_thresh", 0.35))
        self.yolo = None
        if self.enable_yolo:
            try:  # pragma: no cover - heavy dep
                from ultralytics import YOLO  # type: ignore
                model_path = str(vis_cfg.get("yolo_model", "yolo11n.pt"))
                self.yolo = YOLO(model_path)
            except Exception as e:
                logging.getLogger(__name__).warning("YOLO init failed: %s", e)
                self.yolo = None

    def detect_ui(self, frame: np.ndarray) -> Dict:
        ui: Dict = {}
        # Elixir
        try:
            ui["elixir"] = int(self.ocr.read_elixir(frame, self.elixir_roi))
        except Exception:
            ui["elixir"] = 0
        if self.save_unknown and ui.get("elixir", 0) == 0:
            # save elixir ROI crop for later labeling
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = self.elixir_roi
            save_crop("elixir", frame, (int(x1*w), int(y1*h), int(x2*w), int(y2*h)), self.crops_dir)

        # Gold
        try:
            ui["gold"] = int(self.ocr.read_gold(frame, self.gold_roi))
        except Exception:
            ui["gold"] = 0
        if self.save_unknown and ui.get("gold", 0) == 0:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = self.gold_roi
            save_crop("gold", frame, (int(x1*w), int(y1*h), int(x2*w), int(y2*h)), self.crops_dir)

        # Daily wins
        try:
            x1, y1, x2, y2 = self.daily_wins_roi
            roi = frame[int(y1*frame.shape[0]):int(y2*frame.shape[0]), int(x1*frame.shape[1]):int(x2*frame.shape[1])]
            lines = self.ocr.read_text(roi)
            ui["daily_wins"] = int(next((t for t in [l.replace(" ", "") for l in lines] if t.isdigit()), "0"))
        except Exception:
            ui["daily_wins"] = 0

        # Heuristic ROIs and positions
        ui["battle_button"] = templates.roi_to_pixels(frame, self.battle_roi)
        ui["rewards_button_center"] = templates.roi_to_pixels(frame, self.rewards_roi_center)
        ui["rewards_button_bottom"] = templates.roi_to_pixels(frame, self.rewards_roi_bottom)
        ui["upgrade_button"] = templates.roi_to_pixels(frame, self.upgrade_roi)
        ui["card_slots"] = templates.card_centers_from_config(frame, self.cards_centers_norm)
        ui["safe_zone"] = templates.roi_to_pixels(frame, self.safe_zone_roi)

        # YOLO detections
        ui["yolo"] = {"troop": [], "my_tower": [], "enemy_tower": [], "button": [], "card": [], "chest": [], "rewards_button": [], "upgrade_button": []}
        for det in self._predict_yolo(frame):
            label = det.get("label", "")
            conf = float(det.get("conf", 0.0))
            box = det.get("box")
            name = det.get("name", label)
            if box is None:
                continue
            use_label = label
            if label == "tower":
                # Split into my/enemy by vertical position (lower half = my)
                h = frame.shape[0]
                cy = (box[1] + box[3]) / 2
                use_label = "my_tower" if cy > h * 0.5 else "enemy_tower"
            if conf >= self.yolo_conf and use_label in ui["yolo"]:
                ui["yolo"][use_label].append(box)
            else:
                # Save low-conf/unknown crops for later labeling
                if self.save_unknown:
                    lbl = f"unknown_{label or name}".replace(" ", "_")
                    save_crop(lbl, frame, box, self.crops_dir)

        # Derive state helpers
        # Hand cards via OCR around card centers
        try:
            ui["hand_cards"] = self.ocr.read_hand_cards(frame, self.cards_centers_norm)
        except Exception:
            ui["hand_cards"] = ["", "", "", ""]

        # Tower HPs via OCR near detected towers
        my_towers = ui["yolo"].get("my_tower", [])
        enemy_towers = ui["yolo"].get("enemy_tower", [])
        ui["tower_hps"] = [self.ocr.read_number_near_box(frame, b) for b in my_towers]
        ui["enemy_tower_hps"] = [self.ocr.read_number_near_box(frame, b) for b in enemy_towers]

        # Rewards readout from center ROI
        try:
            ui["rewards"] = self.ocr.read_rewards(frame, tuple(self.rewards_roi_center))
        except Exception:
            ui["rewards"] = {"gold": 0, "cards": 0}
        return ui

    def _predict_yolo(self, frame: np.ndarray) -> List[Dict]:
        """Returns list of {label, conf, box=(x1,y1,x2,y2)}."""
        if self.yolo is None:
            return []
        dets: List[Dict] = []
        try:  # pragma: no cover
            imgsz = 320
            try:
                imgsz = int(self.yolo_conf)  # placeholder; corrected below
            except Exception:
                pass
            # read configured imgsz
            try:
                from utils.config import load_config
                imgsz = int(load_config().get("vision", {}).get("yolo_imgsz", 320))
            except Exception:
                imgsz = 320
            results = self.yolo.predict(source=frame, conf=self.yolo_conf, imgsz=imgsz, verbose=False)
            if not results:
                return []
            res = results[0]
            names = getattr(res, "names", {})
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                return []
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else []
            confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else []
            clsi = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, "cls") else []
            for (x1, y1, x2, y2), cf, ci in zip(xyxy, confs, clsi):
                name = names.get(int(ci), str(ci)) if isinstance(names, dict) else str(ci)
                label = self._map_name_to_label(str(name), frame.shape[0])
                dets.append({"name": str(name), "label": label, "conf": float(cf), "box": (int(x1), int(y1), int(x2), int(y2))})
        except Exception as e:
            logger.debug("YOLO predict error: %s", e)
        return dets

    def _map_name_to_label(self, name: str, img_h: int) -> str:
        n = name.lower()
        # Simple heuristics: map to our target label set
        if "reward" in n:
            return "rewards_button"
        if "upgrade" in n:
            return "upgrade_button"
        if "chest" in n:
            return "chest"
        if "card" in n:
            return "card"
        if "button" in n or "ok" in n:
            return "button"
        if "tower" in n:
            # Side unknown -> generic tower; caller may split by y
            return "tower"
        if any(k in n for k in ("troop", "knight", "goblin", "archer", "minion", "barbarian")):
            return "troop"
        return "troop"
