import logging
from typing import Optional, Tuple

import numpy as np

from actions.adb import ADBController


logger = logging.getLogger(__name__)


class Capture:
    def __init__(self, adb: ADBController):
        self.adb = adb

    def screenshot(self) -> np.ndarray:
        img = self.adb.screenshot_bgr()
        if img is None:
            # Return a blank 1080p BGR frame in dry-run
            w, h = self.adb.window_size()
            return np.zeros((h, w, 3), dtype=np.uint8)
        return img

