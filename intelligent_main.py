#!/usr/bin/env python3
"""
Intelligent Clash Royale Bot with Local AI Vision
Uses LLaVA models for advanced strategic analysis
Optimized for RTX 3090 with no API costs
"""

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
from vision.ai_vision import LocalVisionAI
from state.game_state import GameState
from planner.rules import RulePlanner


logger = logging.getLogger(__name__)


class AIEpisodeLogger:
    """Enhanced episode logger with AI insights"""
    
    def __init__(self, out_dir: str = "logs/ai_episodes"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        ts = int(time.time())
        self.path = os.path.join(self.out_dir, f"ai_ep_{ts}.jsonl")
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


def intelligent_play_loop(cfg: Dict[str, Any], seed: Optional[int] = None):
    """Enhanced play loop with AI vision analysis"""
    
    # Initialize components
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
    ai_vision = LocalVisionAI(cfg)  # New AI vision component
    planner = RulePlanner()
    state = GameState()

    # Episode tracking
    prev_gold = 0
    prev_daily_wins = 0
    prev_enemy_towers = 3
    ep = AIEpisodeLogger()
    pending = None
    episode_idx = 0
    
    # AI tracking
    last_ai_advice = ""
    ai_analysis_count = 0

    def state_vec(s: GameState, enemy_troop_count: int = 0):
        elixir_norm = s.elixir / 10.0
        my = (np.array(s.tower_hps, dtype=np.float32)[:3] if len(s.tower_hps) else np.zeros(3)) / 100.0
        en = (np.array(s.enemy_tower_hps, dtype=np.float32)[:3] if len(s.enemy_tower_hps) else np.zeros(3)) / 100.0
        hand = np.ones(4, dtype=np.float32) / 4.0
        troop = np.array([min(1.0, enemy_troop_count / 10.0)], dtype=np.float32)
        return np.concatenate([[elixir_norm], my, en, hand, troop]).astype(float).tolist()

    logger.info("🤖 Starting intelligent play loop with AI vision")
    logger.info("🎯 AI Model: %s", ai_vision.model_name if ai_vision.is_available() else "Not available")
    logger.info("🚀 Device: %s", ai_vision.device if ai_vision.is_available() else "N/A")
    logger.info("🎮 Dry run: %s", controller.dry_run)
    
    start_ts = time.time()
    last_action_ts = 0.0
    last_force_tap_ts = 0.0
    detection_failures = 0
    
    try:
        while True:
            frame = capture.screenshot()
            ui = detector.detect_ui(frame)
            state.update_from_frame(frame, ui)
            
            # AI Vision Analysis
            ai_analysis = None
            if ai_vision.is_available():
                ai_analysis = ai_vision.analyze_screenshot(frame)
                if ai_analysis:
                    ai_analysis_count += 1
                    strategic_advice = ai_vision.get_strategic_advice(ai_analysis)
                    if strategic_advice != last_ai_advice:
                        logger.info("🧠 AI Strategy: %s", strategic_advice)
                        last_ai_advice = strategic_advice
            
            # Enhanced logging with AI insights
            troop_names = ui.get("yolo_meta", {}).get("troop_names", [])
            ai_summary = ai_analysis.get("summary", "") if ai_analysis else ""
            
            logger.info(
                "Detection: battle=%s elixir=%s towers=%s/%s troops=%d %s",
                ui.get("in_battle", False),
                ui.get("elixir", 0),
                len(ui.get("tower_hps", [])),
                len(ui.get("enemy_tower_hps", [])),
                len(troop_names),
                f"AI: {ai_summary[:50]}..." if ai_summary else ""
            )

            # Fallback notice
            if not (ui.get("in_battle") or any(ui.get("yolo", {}).get(k) for k in ["troop", "my_tower", "enemy_tower"])) and ui.get("elixir", 0) == 0:
                logger.info("Detection likely failed, using ROI fallback taps if needed")

            in_battle = bool(ui.get("in_battle", False))
            action_taken = False
            
            # Menu navigation
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
                
                # Count detection failures and force start if needed
                has_any_signal = bool(troop_names or ui.get("elixir", 0) > 0 or any(ui.get("yolo", {}).get(k) for k in ["button", "rewards_button", "upgrade_button"]))
                if not has_any_signal:
                    detection_failures += 1
                else:
                    detection_failures = 0
                if detection_failures >= 5 and (time.time() - last_force_tap_ts) > 1.0:
                    x1, y1, x2, y2 = ui.get("battle_button", (int(0.4*frame.shape[1]), int(0.85*frame.shape[0]), int(0.6*frame.shape[1]), int(0.92*frame.shape[0])))
                    controller.click(int((x1 + x2) / 2), int((y1 + y2) / 2))
                    logger.info("Thought: Screen not recognized — assuming main, tapping center battle")
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

            # Enhanced battle action planning with AI insights
            now = time.time()
            if in_battle and (now - last_action_ts > 0.5):
                # Check if AI recommends specific action
                ai_card = None
                ai_priority = "low"
                if ai_analysis:
                    ai_action = ai_analysis.get("recommended_action", "wait")
                    ai_card = ai_analysis.get("card_to_play")
                    ai_priority = ai_analysis.get("priority", "low")
                    
                    # High priority AI recommendations override planner
                    if ai_priority == "high" and ai_action != "wait" and ai_card:
                        logger.info("🚨 High priority AI action: %s with %s", ai_action, ai_card)
                
                # Use rule-based planner (potentially enhanced by AI later)
                plan = planner.plan_battle(state, frame)
                if plan is not None:
                    card_idx, (nx, ny) = plan
                    
                    # Override with AI card recommendation if high priority
                    if ai_priority == "high" and ai_card:
                        # Try to find AI recommended card in hand (simplified)
                        logger.info("🧠 AI recommends: %s (priority: %s)", ai_card, ai_priority)
                    
                    # Select card (approximate slot centers)
                    slots = ui.get("card_slots", {})
                    px, py = slots.get(card_idx, (int(0.2 * frame.shape[1]), int(0.92 * frame.shape[0])))
                    logger.info("Action: Selecting card slot %d at (%d,%d)", card_idx, px, py)
                    controller.click(px, py)
                    time.sleep(0.1)
                    logger.info("Action: Deploying at normalized (%.2f, %.2f)", nx, ny)
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
                        "ai_analysis": ai_analysis,  # Include AI insights
                        "ai_advice": strategic_advice if ai_analysis else None
                    }
                    
                    if pending is not None:
                        pending["next_state"] = obs
                        ep.log(pending)
                    pending = record
                    
                    logger.info(
                        "ACTION card=%d target=(%.3f, %.3f) reward=%d (gold=%d, wins=%d, crowns=%d)",
                        card_idx, nx, ny, reward_signal, gold, daily_wins, 3 - enemy_tower_count
                    )
                    
                    # Update trackers and exploration decay when a battle ends (wins increment)
                    if daily_wins > prev_daily_wins:
                        episode_idx += 1
                        planner.set_exploration(episode_idx)
                        logger.info("🏆 Battle completed! Episode %d, AI analyses: %d", episode_idx, ai_analysis_count)
                    
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
            
    except KeyboardInterrupt:
        logger.info("🛑 Stopping intelligent bot...")
    finally:
        # Cleanup
        ep.close()
        ai_vision.cleanup()
        logger.info("🧹 Cleanup complete")


def main():
    parser = argparse.ArgumentParser(description="Intelligent Clash Royale Bot with Local AI")
    parser.add_argument("--mode", choices=["local", "play"], default="local", 
                       help="Bot mode: 'local' for AI-enhanced, 'play' for standard")
    parser.add_argument("--seed", type=int, default=123, help="Deterministic seed")
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg)

    logger.info("🤖 Intelligent Clash Royale Bot Starting")
    logger.info("🎯 Mode: %s", args.mode)
    
    if args.mode == "local":
        # Check AI availability
        try:
            from vision.ai_vision import LocalVisionAI
            ai = LocalVisionAI(cfg)
            if ai.is_available():
                logger.info("✅ Local AI vision ready!")
            else:
                logger.warning("⚠️  Local AI not available - run: python setup_local_ai.py")
        except Exception as e:
            logger.error("❌ AI initialization failed: %s", e)
            logger.info("💡 Run setup script: python setup_local_ai.py")
    
    # Set deterministic seed
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        import torch
        if torch.cuda.is_available():
            torch.manual_seed(args.seed)
    except Exception:
        pass

    # Run the enhanced loop
    if args.mode == "local":
        intelligent_play_loop(cfg, seed=args.seed)
    else:
        # Fallback to standard play loop
        from main import play_loop
        play_loop(cfg, seed=args.seed)


if __name__ == "__main__":
    main()