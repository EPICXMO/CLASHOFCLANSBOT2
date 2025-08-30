import logging
from typing import Optional, Tuple, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def roi_to_pixels(img: np.ndarray, roi_norm: List[float]) -> Tuple[int, int, int, int]:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = roi_norm
    return (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))


def card_centers_from_config(img: np.ndarray, centers_norm: List[List[float]]) -> Dict[int, Tuple[int, int]]:
    h, w = img.shape[:2]
    res: Dict[int, Tuple[int, int]] = {}
    for i, (nx, ny) in enumerate(centers_norm):
        res[i] = (int(nx * w), int(ny * h))
    return res


def center_of_roi(img: np.ndarray, roi_norm: List[float]) -> Tuple[float, float]:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = roi_norm
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return (cx, cy)
