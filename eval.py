import argparse
import logging

from utils.config import load_config
from utils.logging import setup_logging


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg)

    # Use mock env for evaluation scaffold
    from planner.ppo import MockClashEnv
    import numpy as np

    env = MockClashEnv()
    wins = 0
    losses = 0
    for _ in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_rew = 0.0
        while not done:
            action = env.action_space.sample()
            obs, rew, done, truncated, info = env.step(action)
            ep_rew += rew
        if ep_rew > 0:
            wins += 1
        else:
            losses += 1
    logger.info("Eval results: wins=%d losses=%d", wins, losses)


if __name__ == "__main__":
    main()

