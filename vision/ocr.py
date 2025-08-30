import logging
from typing import Optional, Tuple, List, Dict

import numpy as np

logger = logging.getLogger(__name__)

try:  # pragma: no cover - heavy dep
    from paddleocr import PaddleOCR
except Exception:  # pragma: no cover
    PaddleOCR = None  # type: ignore


class OCR:
    def __init__(self, enabled: bool = True, lang: str = "en"):
        self.enabled = enabled and (PaddleOCR is not None)
        self.lang = lang
        self._ocr = None
        if self.enabled:
            try:
                self._ocr = PaddleOCR(use_angle_cls=True, lang=self.lang)
            except Exception as e:  # pragma: no cover
                logger.warning("PaddleOCR init failed: %s", e)
                self.enabled = False

    def read_text(self, image: np.ndarray) -> List[str]:
        if not self.enabled or self._ocr is None:
            return []
        try:  # pragma: no cover - depends on PaddleOCR
            result = self._ocr.ocr(image, cls=True)
            lines: List[str] = []
            for line in result:
                text = " ".join([word_info[-1][0] for word_info in line])
                if text:
                    lines.append(text)
            return lines
        except Exception as e:
            logger.debug("OCR error: %s", e)
            return []

    def read_elixir(self, image: np.ndarray, roi_norm: Tuple[float, float, float, float]) -> int:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = roi_norm
        x1i, y1i, x2i, y2i = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        x1i, y1i = max(0, x1i), max(0, y1i)
        x2i, y2i = min(w, x2i), min(h, y2i)
        roi = image[y1i:y2i, x1i:x2i]
        lines = self.read_text(roi)
        for text in lines:
            t = text.strip().replace(" ", "")
            if t.isdigit():
                try:
                    return int(t)
                except Exception:
                    pass
        return 0

    def read_gold(self, image: np.ndarray, roi_norm: Tuple[float, float, float, float]) -> int:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = roi_norm
        x1i, y1i, x2i, y2i = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        x1i, y1i = max(0, x1i), max(0, y1i)
        x2i, y2i = min(w, x2i), min(h, y2i)
        roi = image[y1i:y2i, x1i:x2i]
        lines = self.read_text(roi)
        for text in lines:
            t = text.strip().replace(",", "").replace(" ", "")
            if t.isdigit():
                try:
                    return int(t)
                except Exception:
                    pass
        return 0

    # Generic helpers
    def _roi(self, image: np.ndarray, roi_norm: Tuple[float, float, float, float]) -> np.ndarray:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = roi_norm
        x1i, y1i, x2i, y2i = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        x1i, y1i = max(0, x1i), max(0, y1i)
        x2i, y2i = min(w, x2i), min(h, y2i)
        return image[y1i:y2i, x1i:x2i]

    def _first_int(self, lines: List[str]) -> int:
        for text in lines:
            t = text.replace(",", "").replace(" ", "")
            if t.isdigit():
                try:
                    return int(t)
                except Exception:
                    pass
        return 0

    def read_rewards(self, image: np.ndarray, roi_norm: Tuple[float, float, float, float]) -> Dict[str, int]:
        """Parse rewards area for gold and cards counts. Returns {gold, cards}.
        Very heuristic: searches first ints in lines and keywords.
        """
        roi = self._roi(image, roi_norm)
        lines = self.read_text(roi)
        gold = 0
        cards = 0
        for text in lines:
            low = text.lower()
            # naive keyword matching
            if "gold" in low or "coin" in low:
                gold = max(gold, self._first_int([text]))
            if "card" in low:
                cards = max(cards, self._first_int([text]))
        # fallback: just first numbers encountered
        if gold == 0:
            gold = self._first_int(lines)
        if cards == 0 and len(lines) > 1:
            cards = self._first_int(lines[1:])
        return {"gold": gold, "cards": cards}

    def read_hand_cards(self, image: np.ndarray, centers_norm: List[Tuple[float, float]], box_norm_hw: Tuple[float, float] = (0.08, 0.08)) -> List[str]:
        """OCR card names near provided normalized centers along bottom row."""
        h, w = image.shape[:2]
        bw, bh = int(box_norm_hw[0] * w), int(box_norm_hw[1] * h)
        names: List[str] = []
        for nx, ny in centers_norm:
            cx, cy = int(nx * w), int(ny * h)
            x1 = max(0, cx - bw // 2)
            y1 = max(0, cy - bh // 2)
            x2 = min(w, x1 + bw)
            y2 = min(h, y1 + bh)
            roi = image[y1:y2, x1:x2]
            texts = self.read_text(roi)
            names.append(texts[0] if texts else "")
        return names

    def read_number_near_box(self, image: np.ndarray, box: Tuple[int, int, int, int], rel_roi: Tuple[float, float, float, float] = (0.0, -0.25, 1.0, 0.0)) -> int:
        """Read a number near the given box. rel_roi is relative to the box: (x1_rel, y1_rel, x2_rel, y2_rel), where
        (-0.25..1.25) expand around box. Default reads slightly above the box."""
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        rx1, ry1, rx2, ry2 = rel_roi
        px1 = x1 + int(rx1 * bw)
        py1 = y1 + int(ry1 * bh)
        px2 = x1 + int(rx2 * bw)
        py2 = y1 + int(ry2 * bh)
        h, w = image.shape[:2]
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(w, px2), min(h, py2)
        roi = image[py1:py2, px1:px2]
        lines = self.read_text(roi)
        return self._first_int(lines)
