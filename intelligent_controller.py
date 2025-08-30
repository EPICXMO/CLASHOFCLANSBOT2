"""
Intelligent Controller - Executes actions with understanding rather than blind clicking.
This replaces coordinate-based actions with context-aware intelligent execution.
"""

import logging
import time
from typing import Dict, Any, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


class IntelligentController:
    """
    Intelligent action controller that understands context and executes actions intelligently.
    Unlike the old system that just clicked coordinates, this understands WHY it's acting.
    """
    
    def __init__(self, adb_controller, vision_ai, config: Dict[str, Any]):
        self.adb = adb_controller
        self.vision_ai = vision_ai
        self.config = config
        self.last_action_time = 0
        self.action_history = []
        
        # Action cooldowns to prevent spam
        self.min_action_interval = 0.5
        self.last_card_deploy_time = 0
        self.card_deploy_cooldown = 1.0
    
    def execute_ai_decision(self, analysis: Dict[str, Any], frame: np.ndarray) -> bool:
        """
        Execute an action based on AI analysis.
        Returns True if action was taken, False otherwise.
        """
        current_time = time.time()
        
        # Respect minimum action interval
        if current_time - self.last_action_time < self.min_action_interval:
            return False
        
        action_type = analysis.get("action_type", "wait")
        action_details = analysis.get("action_details", {})
        
        # Log the AI's reasoning
        reasoning = action_details.get("reasoning", "No reasoning provided")
        logger.info("🧠 AI Decision: %s - %s", action_type, reasoning)
        
        action_taken = False
        
        try:
            if action_type == "click_battle":
                action_taken = self._intelligent_battle_click(frame, analysis)
            elif action_type == "deploy_card":
                action_taken = self._intelligent_card_deploy(frame, analysis)
            elif action_type == "defend":
                action_taken = self._intelligent_defend(frame, analysis)
            elif action_type == "attack":
                action_taken = self._intelligent_attack(frame, analysis)
            elif action_type == "click_button":
                action_taken = self._intelligent_button_click(frame, analysis)
            elif action_type == "wait":
                logger.info("🤖 AI says wait: %s", reasoning)
                action_taken = False
            else:
                logger.warning("Unknown action type: %s", action_type)
                action_taken = False
        
        except Exception as e:
            logger.error("Error executing AI action %s: %s", action_type, e)
            action_taken = False
        
        if action_taken:
            self.last_action_time = current_time
            self._record_action(action_type, action_details, analysis)
        
        return action_taken
    
    def _intelligent_battle_click(self, frame: np.ndarray, analysis: Dict[str, Any]) -> bool:
        """Click battle button intelligently based on screen understanding"""
        # Use AI to find the battle button rather than hard-coded coordinates
        action_details = analysis.get("action_details", {})
        
        # Try to click at AI-suggested coordinates if available
        if "target_x" in action_details and "target_y" in action_details:
            x = action_details["target_x"]
            y = action_details["target_y"]
            self.adb.tap_norm(x, y)
            logger.info("🎯 AI-guided battle click at (%.2f, %.2f)", x, y)
            return True
        
        # Fallback to center battle area
        self.adb.tap_norm(0.5, 0.88)  # Typical battle button location
        logger.info("🎯 Fallback battle click")
        return True
    
    def _intelligent_card_deploy(self, frame: np.ndarray, analysis: Dict[str, Any]) -> bool:
        """Deploy a card intelligently based on strategic analysis"""
        current_time = time.time()
        
        # Respect card deployment cooldown
        if current_time - self.last_card_deploy_time < self.card_deploy_cooldown:
            logger.info("⏰ Card deploy on cooldown")
            return False
        
        action_details = analysis.get("action_details", {})
        
        # Get card index from AI
        card_index = action_details.get("card_index", 0)
        target_x = action_details.get("target_x", 0.5)
        target_y = action_details.get("target_y", 0.6)
        
        # Validate card index
        if not (0 <= card_index <= 3):
            logger.warning("Invalid card index from AI: %s", card_index)
            card_index = 0
        
        # Calculate card slot position
        card_x_positions = [0.2, 0.4, 0.6, 0.8]  # Typical card positions
        card_y = 0.92
        
        # Select the card first
        self.adb.tap_norm(card_x_positions[card_index], card_y)
        logger.info("🃏 Selected card %d", card_index)
        
        # Small delay to ensure card is selected
        time.sleep(0.1)
        
        # Deploy at target location
        self.adb.tap_norm(target_x, target_y)
        logger.info("🚀 Deployed card %d at (%.2f, %.2f)", card_index, target_x, target_y)
        
        self.last_card_deploy_time = current_time
        return True
    
    def _intelligent_defend(self, frame: np.ndarray, analysis: Dict[str, Any]) -> bool:
        """Execute defensive strategy based on AI analysis"""
        # This is essentially a defensive card deployment
        return self._intelligent_card_deploy(frame, analysis)
    
    def _intelligent_attack(self, frame: np.ndarray, analysis: Dict[str, Any]) -> bool:
        """Execute offensive strategy based on AI analysis"""
        # This is essentially an offensive card deployment
        return self._intelligent_card_deploy(frame, analysis)
    
    def _intelligent_button_click(self, frame: np.ndarray, analysis: Dict[str, Any]) -> bool:
        """Click a button intelligently (popups, OK buttons, etc.)"""
        action_details = analysis.get("action_details", {})
        
        # Use AI-provided coordinates
        target_x = action_details.get("target_x", 0.5)
        target_y = action_details.get("target_y", 0.5)
        
        self.adb.tap_norm(target_x, target_y)
        logger.info("🔘 AI button click at (%.2f, %.2f)", target_x, target_y)
        return True
    
    def handle_popup_intelligently(self, frame: np.ndarray) -> bool:
        """
        Use AI to detect and handle popups intelligently.
        This replaces the old OCR-based popup detection.
        """
        try:
            # Quick analysis to detect popups
            popup_prompt = """
Look at this screenshot. Is there a popup, dialog, or button that needs to be clicked?
If yes, respond with: YES,x,y,description
If no, respond with: NO

Where x,y are normalized coordinates (0.0-1.0) of where to click.
Example: YES,0.5,0.6,OK button to continue
"""
            
            # This would be a lightweight popup detection
            # For now, return False to use existing popup logic
            return False
            
        except Exception as e:
            logger.debug("AI popup detection failed: %s", e)
            return False
    
    def _record_action(self, action_type: str, action_details: Dict[str, Any], analysis: Dict[str, Any]):
        """Record action for learning and debugging"""
        action_record = {
            "timestamp": time.time(),
            "action_type": action_type,
            "action_details": action_details,
            "game_state": analysis.get("game_state", "unknown"),
            "confidence": analysis.get("confidence", 0.0),
            "narration": analysis.get("narration", "")
        }
        
        self.action_history.append(action_record)
        
        # Keep only last 50 actions to prevent memory bloat
        if len(self.action_history) > 50:
            self.action_history = self.action_history[-50:]
    
    def get_recent_actions(self, count: int = 5) -> list:
        """Get recent actions for context"""
        return self.action_history[-count:] if self.action_history else []
    
    def reset_cooldowns(self):
        """Reset all action cooldowns (useful for testing)"""
        self.last_action_time = 0
        self.last_card_deploy_time = 0