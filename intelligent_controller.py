"""
Intelligent Controller - Wraps ADB controller with AI-powered decision execution.

This module provides a smart layer over the existing ADB controller that can execute
complex actions based on AI analysis results.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any

from actions.adb import ADBController
from ai_vision import AIAnalysis

logger = logging.getLogger(__name__)


class IntelligentController:
    """Intelligent wrapper around ADB controller for AI-driven actions."""
    
    def __init__(self, adb_controller: ADBController):
        """Initialize with existing ADB controller."""
        self.adb = adb_controller
        self.last_action_time = 0.0
        self.action_cooldown = 0.5  # Minimum time between actions
        logger.info("Intelligent controller initialized")
    
    def execute_intelligent_action(self, analysis: AIAnalysis, ui_data: Dict) -> bool:
        """
        Execute an action based on AI analysis.
        
        Args:
            analysis: AI analysis result with action recommendation
            ui_data: Current UI detection data
            
        Returns:
            True if action was executed, False if skipped
        """
        current_time = time.time()
        
        # Respect action cooldown
        if current_time - self.last_action_time < self.action_cooldown:
            logger.debug("Action on cooldown, skipping")
            return False
        
        try:
            executed = False
            
            if analysis.action_type == "deploy_card":
                executed = self._deploy_card(analysis, ui_data)
            elif analysis.action_type == "click_button":
                executed = self._click_button(analysis, ui_data)
            elif analysis.action_type == "wait":
                logger.info("🤖 AI Decision: Waiting - %s", analysis.reasoning)
                executed = True  # Waiting is a valid action
            elif analysis.action_type == "no_action":
                logger.debug("AI recommends no action")
                executed = True
            else:
                logger.warning("Unknown action type: %s", analysis.action_type)
                
            if executed:
                self.last_action_time = current_time
                
            return executed
            
        except Exception as e:
            logger.error("Failed to execute intelligent action: %s", e, exc_info=True)
            return False
    
    def _deploy_card(self, analysis: AIAnalysis, ui_data: Dict) -> bool:
        """Deploy a card based on AI analysis."""
        if analysis.target_card is None or analysis.target_position is None:
            logger.warning("Deploy card action missing target card or position")
            return False
        
        # Get card slots from UI data
        card_slots = ui_data.get("card_slots", {})
        if not card_slots:
            logger.warning("No card slots detected")
            return False
        
        # Validate card index
        if analysis.target_card not in card_slots:
            logger.warning("Target card slot %d not available", analysis.target_card)
            # Try to use first available slot
            available_slots = list(card_slots.keys())
            if available_slots:
                analysis.target_card = available_slots[0]
                logger.info("Using fallback card slot %d", analysis.target_card)
            else:
                return False
        
        # Get card slot position
        card_x, card_y = card_slots[analysis.target_card]
        
        # Calculate deployment position
        deploy_x, deploy_y = analysis.target_position
        
        logger.info("🎯 AI Decision: deploy_card - %s", analysis.reasoning)
        logger.info("🃏 Selecting card slot %d at (%d, %d)", analysis.target_card, card_x, card_y)
        
        # Select card
        self.adb.click(card_x, card_y)
        time.sleep(0.1)  # Brief pause between selection and deployment
        
        logger.info("🎲 Deploying at normalized (%.2f, %.2f)", deploy_x, deploy_y)
        # Deploy card
        self.adb.tap_norm(deploy_x, deploy_y)
        
        return True
    
    def _click_button(self, analysis: AIAnalysis, ui_data: Dict) -> bool:
        """Click a button based on AI analysis."""
        if not analysis.target_button:
            logger.warning("Click button action missing target button")
            return False
        
        button_pos = None
        
        # Map button names to UI elements
        if analysis.target_button == "battle":
            button_pos = ui_data.get("battle_button")
        elif analysis.target_button == "ok":
            # Look for OK button in YOLO detections
            for button_box in ui_data.get("yolo", {}).get("button", []):
                button_pos = button_box
                break  # Use first button found
        elif analysis.target_button == "collect":
            button_pos = ui_data.get("rewards_button_center") or ui_data.get("rewards_button_bottom")
        elif analysis.target_button == "upgrade":
            button_pos = ui_data.get("upgrade_button")
        else:
            logger.warning("Unknown button target: %s", analysis.target_button)
            return False
        
        if not button_pos:
            logger.warning("Button %s not found in UI", analysis.target_button)
            return False
        
        # Calculate click position
        if isinstance(button_pos, (list, tuple)) and len(button_pos) == 4:
            x1, y1, x2, y2 = button_pos
            click_x, click_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
        else:
            logger.warning("Invalid button position format: %s", button_pos)
            return False
        
        logger.info("🖱️ AI Decision: click_button(%s) - %s", analysis.target_button, analysis.reasoning)
        logger.info("📍 Clicking at (%d, %d)", click_x, click_y)
        
        self.adb.click(click_x, click_y)
        return True
    
    def calculate_strategic_position(self, strategy: str, ui_data: Dict) -> Tuple[float, float]:
        """
        Calculate strategic deployment position based on strategy description.
        
        Args:
            strategy: Strategy description from AI
            ui_data: Current UI detection data
            
        Returns:
            Normalized (x, y) position for deployment
        """
        # Get safe zone for fallback
        safe_zone = ui_data.get("safe_zone", [0.1, 0.55, 0.9, 0.75])
        safe_x = (safe_zone[0] + safe_zone[2]) / 2
        safe_y = (safe_zone[1] + safe_zone[3]) / 2
        
        strategy_lower = strategy.lower()
        
        # Strategic position mapping
        if "left" in strategy_lower or "counter-push" in strategy_lower:
            return (0.2, 0.65)  # Left lane
        elif "right" in strategy_lower:
            return (0.8, 0.65)  # Right lane
        elif "center" in strategy_lower or "defend" in strategy_lower:
            return (0.5, 0.6)   # Center defense
        elif "bridge" in strategy_lower:
            return (0.5, 0.7)   # Bridge area
        elif "back" in strategy_lower or "support" in strategy_lower:
            return (0.5, 0.8)   # Back support
        else:
            # Default to safe zone center
            return (safe_x, safe_y)
    
    def get_action_cooldown(self) -> float:
        """Get current action cooldown."""
        return self.action_cooldown
    
    def set_action_cooldown(self, cooldown: float) -> None:
        """Set action cooldown."""
        self.action_cooldown = max(0.1, cooldown)
        logger.debug("Action cooldown set to %.1fs", self.action_cooldown)
    
    def time_since_last_action(self) -> float:
        """Get time since last action was executed."""
        return time.time() - self.last_action_time
    
    def can_act(self) -> bool:
        """Check if controller can perform an action (not on cooldown)."""
        return self.time_since_last_action() >= self.action_cooldown