import argparse
import os
import logging

from utils.config import load_config
from utils.logging import setup_logging


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/run1")
    parser.add_argument("--offline", action="store_true", help="Train offline from logs/episodes")
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg)

    lr = float(cfg.get("rl", {}).get("learning_rate", 3e-4))
    os.makedirs(args.checkpoint, exist_ok=True)

    if args.offline:
        from planner.ppo import build_and_train_offline
        model = build_and_train_offline(total_timesteps=args.timesteps, lr=lr, log_dir=os.path.join(args.checkpoint, "tb"))
    else:
        from planner.ppo import build_and_train_ppo
        model = build_and_train_ppo(total_timesteps=args.timesteps, lr=lr, log_dir=os.path.join(args.checkpoint, "tb"))
    path = os.path.join(args.checkpoint, "model.zip")
    model.save(path)
    logger.info("Saved PPO checkpoint to %s", path)


if __name__ == "__main__":
    main()
