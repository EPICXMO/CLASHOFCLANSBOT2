import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, Any


def ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def setup_logging(cfg: Dict[str, Any]) -> None:
    log_cfg = cfg.get("logging", {})
    level = getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO)
    log_file = log_cfg.get("file", "logs/run.log")
    ensure_dir(os.path.dirname(log_file))

    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers (useful in tests)
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    try:
        fh = RotatingFileHandler(
            log_file,
            maxBytes=int(log_cfg.get("max_bytes", 1048576)),
            backupCount=int(log_cfg.get("backup_count", 3)),
            encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root.addHandler(fh)
    except Exception:
        # File logging may fail in constrained environments; keep console logging only.
        pass

