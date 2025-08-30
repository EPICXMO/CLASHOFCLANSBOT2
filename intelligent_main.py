"""
Intelligent Main Loop - AI-powered gameplay that sees, understands, and adapts.
This replaces the old scripted main.py with true AI intelligence.
"""

import argparse
import time
import logging
import os
import random
from typing import Optional, Dict, Any

import numpy as np

from utils.config import load_config
from utils.logging import setup_logging
from actions.adb import ADBController
from vision.capture import Capture
from vision.detect import UIDetector  # Keep for fallback
from ai_brain import AIBrain

logger = logging.getLogger(__name__)


def intelligent_play_loop(cfg: Dict[str, Any], seed: Optional[int] = None):
    """
    AI-powered play loop that thinks and adapts like ChatGPT.
    This is the new intelligent gameplay system.
    """
    adb_cfg = cfg.get("adb", {})
    controller = ADBController(
        host=adb_cfg.get("host", "127.0.0.1"),
        port=int(adb_cfg.get("port", 5555)),
        connect_timeout_s=int(adb_cfg.get("connect_timeout_s", 5)),
        dry_run=bool(adb_cfg.get("dry_run", True)),
        serial=str(adb_cfg.get("serial", "")) or None,
    )
    
    # Connect to device
    if not controller.connect():
        logger.error("Failed to connect to device. Check ADB connection.")
        return
    
    # Initialize AI and capture systems
    capture = Capture(controller)
    ai_brain = AIBrain(controller, cfg)
    
    # Keep the old detector as fallback for emergencies
    fallback_detector = UIDetector(cfg)
    
    logger.info("🚀 Starting INTELLIGENT play loop (AI-powered)")
    logger.info("🧠 Bot will see, understand, and adapt like ChatGPT")
    logger.info("🎯 Dry run mode: %s", controller.dry_run)
    
    # Performance tracking
    start_time = time.time()
    loop_count = 0
    ai_action_count = 0
    fallback_action_count = 0
    
    # Main AI gameplay loop
    while True:
        try:
            loop_count += 1
            
            # Capture current screen
            frame = capture.screenshot()
            if frame is None or frame.size == 0:
                logger.warning("Invalid frame captured, retrying...")
                time.sleep(1.0)
                continue
            
            # AI THINKING AND ACTING
            # This is where the magic happens - AI sees and understands the game
            ai_action_taken = ai_brain.think_and_act(frame)
            
            if ai_action_taken:
                ai_action_count += 1
            
            # FALLBACK HANDLING
            # Use traditional detection only for emergencies/popups
            if not ai_action_taken:
                fallback_action_taken = _handle_fallback_situations(
                    frame, controller, fallback_detector, ai_brain
                )
                if fallback_action_taken:
                    fallback_action_count += 1
            
            # PERFORMANCE LOGGING
            if loop_count % 20 == 0:  # Every 20 loops
                elapsed = time.time() - start_time
                stats = ai_brain.get_performance_stats()
                logger.info(
                    "📊 Stats: loops=%d ai_actions=%d fallback_actions=%d time=%.1fs state=%s",
                    loop_count, ai_action_count, fallback_action_count, elapsed, stats["current_state"]
                )
            
            # Loop timing
            time.sleep(0.2)  # 5 FPS is sufficient for strategic gameplay
            
        except KeyboardInterrupt:
            logger.info("🛑 Stopping intelligent bot (user interrupt)")
            break
        except Exception as e:
            logger.error("Error in intelligent play loop: %s", e)
            logger.info("🔄 Continuing with AI brain reset...")
            ai_brain.reset_state()
            time.sleep(2.0)


def _handle_fallback_situations(frame: np.ndarray, controller: ADBController, 
                               detector: UIDetector, ai_brain: AIBrain) -> bool:
    """
    Handle situations where AI needs fallback assistance.
    This bridges the old system with the new AI system.
    """
    try:
        # Use AI first to try to handle the situation
        ai_handled = ai_brain.handle_emergency_situation(frame)
        if ai_handled:
            return True
        
        # If AI can't handle it, use traditional detection for basic popups
        ui = detector.detect_ui(frame)
        
        # Handle obvious popups with OCR
        try:
            for box in ui.get("yolo", {}).get("button", []):
                x1, y1, x2, y2 = box
                crop = frame[y1:y2, x1:x2]
                texts = detector.ocr.read_text(crop)
                low = " ".join(texts).lower()
                
                if any(k in low for k in ["ok", "close", "confirm", "collect", "continue"]):
                    controller.click(int((x1 + x2) / 2), int((y1 + y2) / 2))
                    logger.info("🔧 Fallback: popup button clicked (%s)", low[:40])
                    return True
        except Exception:
            pass
        
        # Check if we need to get back to game (basic state recovery)
        game_state = ai_brain.quick_state_check(frame)
        if game_state == "menu":
            # Try to start a battle
            controller.tap_norm(0.5, 0.88)  # Battle button area
            logger.info("🔧 Fallback: battle button clicked (menu state)")
            return True
        
        return False
        
    except Exception as e:
        logger.debug("Fallback handling error: %s", e)
        return False


def traditional_play_loop(cfg: Dict[str, Any], seed: Optional[int] = None):
    """
    Traditional rule-based play loop (the old system).
    This is kept for comparison but should not be the default.
    """
    logger.warning("🚨 Running in TRADITIONAL mode (rule-based, not AI)")
    logger.warning("🚨 This is the OLD system that uses scripts and coordinate clicking")
    
    # Import the old main loop
    from main import play_loop
    play_loop(cfg, seed)


def main():
    parser = argparse.ArgumentParser(description="Clash Royale AI Bot")
    parser.add_argument(
        "--mode", 
        choices=["intelligent", "traditional"], 
        default="intelligent",
        help="intelligent = AI-powered (default), traditional = old rule-based system"
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg)
    
    # Check AI configuration
    ai_config = cfg.get("ai", {})
    if args.mode == "intelligent" and not ai_config.get("openai_api_key"):
        logger.error("🚨 OpenAI API key not configured!")
        logger.error("🚨 Please set your API key in config.yaml under ai.openai_api_key")
        logger.error("🚨 Falling back to traditional mode...")
        args.mode = "traditional"
    
    # Set random seeds
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
    except Exception:
        pass
    
    # Run the selected mode
    if args.mode == "intelligent":
        logger.info("🤖 Starting INTELLIGENT AI-powered bot")
        intelligent_play_loop(cfg, seed=args.seed)
    else:
        logger.info("🕹️  Starting TRADITIONAL rule-based bot")
        traditional_play_loop(cfg, seed=args.seed)


if __name__ == "__main__":
    main()