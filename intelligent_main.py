"""
Intelligent Main - AI-powered main loop for Clash Royale bot.

This replaces the scripted behavior with AI decision making using GPT-4 Vision.
The bot now sees, understands, and acts strategically like a human player.
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
from state.game_state import GameState
from ai_vision import AIVision
from intelligent_controller import IntelligentController
from ai_brain import AIBrain

logger = logging.getLogger(__name__)


class IntelligentEpisodeLogger:
    """Enhanced episode logger for AI decisions and performance."""
    
    def __init__(self, out_dir: str = "logs/ai_episodes"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        ts = int(time.time())
        self.path = os.path.join(self.out_dir, f"ai_ep_{ts}.jsonl")
        self._f = open(self.path, "a", encoding="utf-8")
        logger.info("AI Episode logging to: %s", self.path)

    def log_ai_decision(self, analysis, ui_data: Dict, action_taken: bool) -> None:
        """Log AI decision with full context."""
        try:
            record = {
                "timestamp": time.time(),
                "ai_analysis": {
                    "game_state": analysis.game_state,
                    "situation": analysis.situation,
                    "strategy": analysis.strategy,
                    "action_type": analysis.action_type,
                    "target_card": analysis.target_card,
                    "target_position": analysis.target_position,
                    "narration": analysis.narration,
                    "confidence": analysis.confidence,
                    "reasoning": analysis.reasoning,
                    "threat_level": analysis.threat_level
                },
                "ui_context": {
                    "in_battle": ui_data.get("in_battle", False),
                    "elixir": ui_data.get("elixir", 0),
                    "gold": ui_data.get("gold", 0),
                    "tower_count": len(ui_data.get("yolo", {}).get("my_tower", [])),
                    "enemy_tower_count": len(ui_data.get("yolo", {}).get("enemy_tower", [])),
                    "troop_count": len(ui_data.get("yolo", {}).get("troop", []))
                },
                "action_executed": action_taken
            }
            self._f.write(json.dumps(record) + "\n")
            self._f.flush()
        except Exception as e:
            logger.warning("Failed to log AI decision: %s", e)

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


def intelligent_play_loop(cfg: Dict[str, Any], seed: Optional[int] = None):
    """
    Main AI-powered game loop.
    
    This replaces the scripted behavior with intelligent decision making.
    """
    # Initialize components (reuse existing infrastructure)
    adb_cfg = cfg.get("adb", {})
    controller = ADBController(
        host=adb_cfg.get("host", "127.0.0.1"),
        port=int(adb_cfg.get("port", 5555)),
        connect_timeout_s=int(adb_cfg.get("connect_timeout_s", 5)),
        dry_run=bool(adb_cfg.get("dry_run", True)),
        serial=str(adb_cfg.get("serial", "")) or None,
    )
    controller.connect()

    # Vision and detection (reuse existing)
    capture = Capture(controller)
    detector = UIDetector(cfg)
    state = GameState()

    # NEW: AI-powered components
    ai_vision = AIVision(cfg)
    intelligent_controller = IntelligentController(controller)
    ai_brain = AIBrain(cfg, ai_vision, intelligent_controller)

    # Episode logging
    ep_logger = IntelligentEpisodeLogger()
    
    # Performance tracking
    last_performance_log = time.time()
    last_force_action_time = 0.0
    loop_count = 0

    logger.info("🚀 Starting INTELLIGENT play loop (AI-powered)")
    logger.info("🧠 AI Vision Enabled: %s", ai_vision.is_enabled())
    logger.info("🎮 Dry Run Mode: %s", controller.dry_run)
    
    if not ai_vision.is_enabled():
        logger.error("❌ AI Vision is not enabled! Please configure OpenAI API key.")
        return

    try:
        while True:
            loop_count += 1
            current_time = time.time()
            
            # 1. CAPTURE: Get current screenshot
            frame = capture.screenshot()
            if frame is None:
                logger.warning("Failed to capture screenshot")
                time.sleep(1.0)
                continue
            
            # 2. DETECT: Analyze UI using existing vision system
            ui = detector.detect_ui(frame)
            state.update_from_frame(frame, ui)
            
            # 3. LOG: Current game state
            if loop_count % 10 == 0:  # Log every 10 loops to avoid spam
                logger.debug("🔍 Game State: Battle=%s | Elixir=%s | Gold=%s", 
                           ui.get("in_battle", False), ui.get("elixir", 0), ui.get("gold", 0))
            
            # 4. THINK AND ACT: Let AI brain decide what to do
            action_taken = False
            try:
                action_taken = ai_brain.think_and_act(frame, ui)
            except Exception as e:
                logger.error("🚨 AI Brain error: %s", e, exc_info=True)
            
            # 5. FALLBACK: Handle edge cases and stuck situations
            if not action_taken:
                action_taken = _handle_fallback_actions(controller, ui, current_time, last_force_action_time)
                if action_taken:
                    last_force_action_time = current_time
            
            # 6. POPUP HANDLING: Handle game popups (reuse existing logic)
            _handle_popups(controller, detector, ui, frame)
            
            # 7. PERFORMANCE LOGGING: Log AI performance periodically
            if current_time - last_performance_log > 300:  # Every 5 minutes
                ai_brain.log_performance_summary()
                last_performance_log = current_time
            
            # 8. SLEEP: Control loop frequency
            time.sleep(0.3)  # Slightly slower than original for AI processing
            
    except KeyboardInterrupt:
        logger.info("🛑 Stopping intelligent play loop...")
    except Exception as e:
        logger.error("🚨 Fatal error in intelligent play loop: %s", e, exc_info=True)
    finally:
        # Cleanup
        ai_brain.log_performance_summary()
        ep_logger.close()
        logger.info("🏁 Intelligent play loop ended")


def _handle_fallback_actions(controller: ADBController, ui: Dict, current_time: float, last_force_time: float) -> bool:
    """Handle fallback actions when AI doesn't act."""
    # Don't force actions too frequently
    if current_time - last_force_time < 10.0:
        return False
    
    in_battle = ui.get("in_battle", False)
    
    if not in_battle:
        # Try to start a battle
        battle_button = ui.get("battle_button")
        if battle_button:
            x1, y1, x2, y2 = battle_button
            controller.click(int((x1 + x2) / 2), int((y1 + y2) / 2))
            logger.info("🔄 FALLBACK: Clicking battle button")
            return True
        
        # Look for any YOLO buttons in top area
        top_buttons = [b for b in ui.get("yolo", {}).get("button", []) 
                      if (b[1] + b[3]) / 2 < 300]  # Rough top area
        if top_buttons:
            x1, y1, x2, y2 = top_buttons[0]
            controller.click(int((x1 + x2) / 2), int((y1 + y2) / 2))
            logger.info("🔄 FALLBACK: Clicking top area button")
            return True
    
    return False


def _handle_popups(controller: ADBController, detector: UIDetector, ui: Dict, frame) -> None:
    """Handle game popups (reuse existing logic)."""
    try:
        for box in ui.get("yolo", {}).get("button", []):
            x1, y1, x2, y2 = box
            crop = frame[y1:y2, x1:x2]
            texts = detector.ocr.read_text(crop)
            low = " ".join(texts).lower()
            if any(k in low for k in ["ok", "close", "confirm", "collect", "continue"]):
                controller.click(int((x1 + x2) / 2), int((y1 + y2) / 2))
                logger.info("🔄 POPUP: Handled popup button (%s)", low[:30])
                break
    except Exception as e:
        logger.debug("Popup handling error: %s", e)


def main():
    """Main entry point for intelligent bot."""
    parser = argparse.ArgumentParser(description="AI-Powered Clash Royale Bot")
    parser.add_argument("--mode", choices=["play", "intelligent"], default="intelligent",
                       help="Bot mode: 'play' for original scripted, 'intelligent' for AI-powered")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")
    parser.add_argument("--config", type=str, default="config.yaml", help="Configuration file path")
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)
    setup_logging(cfg)

    # Set random seeds
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
    except Exception:
        pass

    # Check AI configuration
    ai_config = cfg.get("ai", {})
    if not ai_config.get("openai_api_key"):
        logger.error("❌ OpenAI API key not configured!")
        logger.error("Please add your API key to config.yaml under 'ai.openai_api_key'")
        return

    logger.info("🤖 Starting AI-Powered Clash Royale Bot")
    logger.info("🎯 Mode: %s", args.mode)
    logger.info("🔑 API Key configured: %s", bool(ai_config.get("openai_api_key")))

    if args.mode == "intelligent":
        intelligent_play_loop(cfg, seed=args.seed)
    elif args.mode == "play":
        # Import and run original main loop
        from main import play_loop
        logger.info("🔄 Running original scripted mode")
        play_loop(cfg, seed=args.seed)


if __name__ == "__main__":
    main()