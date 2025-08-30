import argparse
import time
import logging
import json
import os
import random
from typing import Optional, Dict, Any

import numpy as np

from utils.config import load_config
from utils.logging import setup_logging
from actions.adb import ADBController
from vision.capture import Capture
from vision.detect import UIDetector
from state.game_state import GameState
from planner.rules import RulePlanner


logger = logging.getLogger(__name__)


class EpisodeLogger:
    def __init__(self, out_dir: str = "logs/episodes"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        ts = int(time.time())
        self.path = os.path.join(self.out_dir, f"ep_{ts}.jsonl")
        self._f = open(self.path, "a", encoding="utf-8")

    def log(self, record: dict) -> None:
        try:
            self._f.write(json.dumps(record) + "\n")
            self._f.flush()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


def play_loop(cfg: Dict[str, Any], seed: Optional[int] = None):
    adb_cfg = cfg.get("adb", {})
    controller = ADBController(
        host=adb_cfg.get("host", "127.0.0.1"),
        port=int(adb_cfg.get("port", 5555)),
        connect_timeout_s=int(adb_cfg.get("connect_timeout_s", 5)),
        dry_run=bool(adb_cfg.get("dry_run", True)),
        serial=str(adb_cfg.get("serial", "")) or None,
    )
    controller.connect()

    capture = Capture(controller)
    detector = UIDetector(cfg)
    planner = RulePlanner()
    state = GameState()

    prev_gold = 0
    prev_daily_wins = 0
    prev_enemy_towers = 3
    ep = EpisodeLogger()
    pending = None
    episode_idx = 0

    def state_vec(s: GameState, enemy_troop_count: int = 0):
        elixir_norm = s.elixir / 10.0
        my = (np.array(s.tower_hps, dtype=np.float32)[:3] if len(s.tower_hps) else np.zeros(3)) / 100.0
        en = (np.array(s.enemy_tower_hps, dtype=np.float32)[:3] if len(s.enemy_tower_hps) else np.zeros(3)) / 100.0
        hand = np.ones(4, dtype=np.float32) / 4.0
        troop = np.array([min(1.0, enemy_troop_count / 10.0)], dtype=np.float32)
        return np.concatenate([[elixir_norm], my, en, hand, troop]).astype(float).tolist()

    logger.info("Starting play loop (dry_run=%s)", controller.dry_run)
    start_ts = time.time()
    last_action_ts = 0.0
    last_force_tap_ts = 0.0
    while True:
        frame = capture.screenshot()
        ui = detector.detect_ui(frame)
        state.update_from_frame(frame, ui)
        # Log detection snapshot
        troop_names = ui.get("yolo_meta", {}).get("troop_names", [])
        logger.info(
            "Detected: in_battle=%s elixir=%s hand=%s towers=%s enemy_towers=%s troops=%d names=%s",
            ui.get("in_battle", False),
            ui.get("elixir", 0),
            ui.get("hand_cards", []),
            ui.get("tower_hps", []),
            ui.get("enemy_tower_hps", []),
            len(troop_names),
            troop_names,
        )
        # Fallback notice
        if not (ui.get("in_battle") or any(ui.get("yolo", {}).get(k) for k in ["troop", "my_tower", "enemy_tower"])) and ui.get("elixir", 0) == 0:
            logger.info("Detection likely failed, using ROI fallback taps if needed")

        in_battle = bool(ui.get("in_battle", False))
        action_taken = False
        if not in_battle:
            # Try tapping a top-region YOLO button (e.g., mode switcher), else battle button ROI
            top_buttons = [b for b in ui.get("yolo", {}).get("button", []) if (b[1] + b[3]) / 2 < frame.shape[0] * 0.25]
            target_btn = top_buttons[0] if top_buttons else ui.get("battle_button")
            if target_btn:
                x1, y1, x2, y2 = target_btn
                controller.click(int((x1 + x2) / 2), int((y1 + y2) / 2))
                logger.info("Tap: not in battle -> switching/battle button")
                action_taken = True
            # Force immediate action after 1s from start
            if not action_taken and (time.time() - start_ts) > 1.0 and (time.time() - last_force_tap_ts) > 1.0:
                x1, y1, x2, y2 = ui.get("battle_button", (int(0.4*frame.shape[1]), int(0.85*frame.shape[0]), int(0.6*frame.shape[1]), int(0.92*frame.shape[0])))
                controller.click(int((x1 + x2) / 2), int((y1 + y2) / 2))
                logger.info("Force Tap: battle ROI (startup)")
                last_force_tap_ts = time.time()

        # Handle popups: YOLO 'button' with OCR text 'OK'
        try:
            for box in ui.get("yolo", {}).get("button", []):
                x1, y1, x2, y2 = box
                crop = frame[y1:y2, x1:x2]
                texts = detector.ocr.read_text(crop)
                low = " ".join(texts).lower()
                if any(k in low for k in ["ok", "close", "confirm", "collect", "continue"]):
                    controller.click(int((x1 + x2) / 2), int((y1 + y2) / 2))
                    logger.info("Tap: popup button (%s)", low[:40])
        except Exception:
            pass

        # Basic action cadence (only while in battle)
        now = time.time()
        if in_battle and (now - last_action_ts > 0.5):
            plan = planner.plan_battle(state, frame)
            if plan is not None:
                card_idx, (nx, ny) = plan
                # Select card (approximate slot centers)
                slots = ui.get("card_slots", {})
                px, py = slots.get(card_idx, (int(0.2 * frame.shape[1]), int(0.92 * frame.shape[0])))
                controller.click(px, py)
                time.sleep(0.1)
                controller.tap_norm(nx, ny)
                # Compute reward: gold delta + crown bonus + win bonus
                gold = int(ui.get("gold", 0))
                daily_wins = int(ui.get("daily_wins", 0))
                gold_delta = max(0, gold - prev_gold)
                enemy_tower_count = len(ui.get("yolo", {}).get("enemy_tower", [])) or prev_enemy_towers
                crown_bonus = 5 if enemy_tower_count < prev_enemy_towers else 0
                win_bonus = 10 if daily_wins > prev_daily_wins else 0
                reward_signal = gold_delta + crown_bonus + win_bonus
                planner.record_outcome(card_idx, float(reward_signal))
                obs = state_vec(state, enemy_troop_count=len(troop_names))
                troop_names = ui.get("yolo_meta", {}).get("troop_names", [])
                record = {
                    "t": time.time(),
                    "state": obs,
                    "action": {"card": card_idx, "nx": nx, "ny": ny},
                    "reward": reward_signal,
                    "enemy_troops": {"count": len(troop_names), "types": sorted(list(set(troop_names)))},
                }
                if pending is not None:
                    pending["next_state"] = obs
                    ep.log(pending)
                pending = record
                logger.info(
                    "ACTION card=%d target=(%.3f, %.3f) reward=%d (gold=%d, wins=%d, crowns=%d)",
                    card_idx,
                    nx,
                    ny,
                    reward_signal,
                    gold,
                    daily_wins,
                    3 - enemy_tower_count,
                )
                # Update trackers and exploration decay when a battle ends (wins increment)
                if daily_wins > prev_daily_wins:
                    episode_idx += 1
                    planner.set_exploration(episode_idx)
                prev_gold = gold
                prev_daily_wins = daily_wins
                prev_enemy_towers = enemy_tower_count
                last_action_ts = now
                action_taken = True

        # Progression actions: rewards/upgrade/chests taps
        planner.handle_progression(controller, ui)

        # If no action for 10s, force battle tap fallback
        if not action_taken and (time.time() - last_action_ts) > 10.0:
            x1, y1, x2, y2 = ui.get("battle_button", (int(0.4*frame.shape[1]), int(0.85*frame.shape[0]), int(0.6*frame.shape[1]), int(0.92*frame.shape[0])))
            controller.click(int((x1 + x2) / 2), int((y1 + y2) / 2))
            logger.info("No action - forcing battle tap")
            last_force_tap_ts = time.time()

        time.sleep(0.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["play"], default="play")
    parser.add_argument("--seed", type=int, default=123, help="Deterministic seed")
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg)

    if args.mode == "play":
        try:
            random.seed(args.seed)
            np.random.seed(args.seed)
        except Exception:
            pass
        play_loop(cfg, seed=args.seed)


if __name__ == "__main__":
    main()
