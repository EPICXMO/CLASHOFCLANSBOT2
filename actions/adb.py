import time
import logging
from typing import Tuple, Optional

try:
    from adbutils import adb
except Exception:  # pragma: no cover - optional in CI
    adb = None  # type: ignore


logger = logging.getLogger(__name__)


class ADBController:
    def __init__(self, host: str = "127.0.0.1", port: int = 5555, connect_timeout_s: int = 5, dry_run: bool = False, serial: Optional[str] = None):
        self.host = host
        self.port = port
        self.connect_timeout_s = connect_timeout_s
        self.dry_run = dry_run or (adb is None)
        self._dev = None
        self.serial = serial

    def connect(self) -> bool:
        if self.dry_run:
            logger.info("ADB dry-run: skipping connect to %s:%s", self.host, self.port)
            return False
        try:
            deadline = time.time() + self.connect_timeout_s
            serial = self.serial or f"{self.host}:{self.port}"
            while time.time() < deadline:
                try:
                    self._dev = adb.device(serial)
                    logger.info("Connected to ADB device %s", serial)
                    return True
                except Exception:
                    try:
                        adb.connect(serial)
                    except Exception:
                        pass
                    time.sleep(0.5)
            logger.warning("ADB connect timeout for %s", serial)
        except Exception as e:
            logger.exception("ADB connect error: %s", e)
        return False

    def window_size(self) -> Tuple[int, int]:
        if self.dry_run or self._dev is None:
            return (1920, 1080)
        try:
            return self._dev.window_size()
        except Exception:
            return (1920, 1080)

    def click(self, x: int, y: int) -> None:
        if self.dry_run or self._dev is None:
            logger.debug("ADB dry-run click at (%d, %d)", x, y)
            return
        try:
            self._dev.click(x, y)
        except Exception as e:
            logger.debug("ADB click error: %s", e)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 400) -> None:
        if self.dry_run or self._dev is None:
            logger.debug("ADB dry-run swipe (%d,%d)->(%d,%d) %dms", x1, y1, x2, y2, duration_ms)
            return
        try:
            self._dev.swipe(x1, y1, x2, y2, duration=duration_ms / 1000.0)
        except Exception as e:
            logger.debug("ADB swipe error: %s", e)

    def screenshot_bgr(self):
        """Returns BGR numpy image or None in dry-run."""
        if self.dry_run or self._dev is None:
            return None
        try:
            img = self._dev.screenshot()
            import numpy as np  # local import to keep CI light
            import cv2

            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.debug("ADB screenshot error: %s", e)
            return None

    # Convenience wrappers for normalized [0,1] coordinates
    def tap_norm(self, nx: float, ny: float) -> None:
        w, h = self.window_size()
        self.click(int(nx * w), int(ny * h))

    def swipe_norm(self, nx1: float, ny1: float, nx2: float, ny2: float, duration_ms: int = 400) -> None:
        w, h = self.window_size()
        self.swipe(int(nx1 * w), int(ny1 * h), int(nx2 * w), int(ny2 * h), duration_ms)
