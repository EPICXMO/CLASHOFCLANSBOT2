"""
AI Brain - Central reasoning engine that coordinates seeing, thinking, and acting.
This is the "ChatGPT-like" intelligence that replaces the rule-based planner.
"""

import logging
import time
from typing import Dict, Any, Optional

import numpy as np

from ai_vision import VisionAI
from intelligent_controller import IntelligentController

logger = logging.getLogger(__name__)


class AIBrain:
    """
    The central AI brain that coordinates vision, reasoning, and action.
    This replaces the simple rule-based planner with true AI intelligence.
    """
    
    def __init__(self, adb_controller, config: Dict[str, Any]):
        self.config = config
        ai_config = config.get("ai", {})
        
        # Initialize AI components
        self.vision = VisionAI(config)
        self.controller = IntelligentController(adb_controller, self.vision, config)
        
        # AI behavior settings
        self.analysis_interval = ai_config.get("analysis_interval", 2.0)
        self.enable_narration = ai_config.get("enable_narration", True)
        
        # State tracking
        self.last_analysis_time = 0
        self.current_analysis = None
        self.game_memory = []
        
        # Performance tracking
        self.analysis_count = 0
        self.action_count = 0
        
        logger.info("🧠 AI Brain initialized - Ready for intelligent gameplay")
    
    def think_and_act(self, frame: np.ndarray) -> bool:
        """
        Main thinking loop - analyzes the current situation and takes intelligent action.
        This is the core method that replaces the old rule-based decision making.
        
        Returns True if action was taken, False otherwise.
        """
        current_time = time.time()
        
        # Perform AI analysis at configured intervals
        should_analyze = (
            self.current_analysis is None or 
            (current_time - self.last_analysis_time) >= self.analysis_interval
        )
        
        if should_analyze:
            self.current_analysis = self._analyze_situation(frame)
            self.last_analysis_time = current_time
            self.analysis_count += 1
        
        # Execute action based on analysis
        action_taken = False
        if self.current_analysis:
            action_taken = self._execute_intelligent_action(frame)
            if action_taken:
                self.action_count += 1
        
        return action_taken
    
    def _analyze_situation(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Analyze the current game situation using AI vision.
        This is where the bot "sees and understands" like ChatGPT would.
        """
        try:
            logger.info("🔍 AI is analyzing the screen...")
            
            # Get AI vision analysis
            analysis = self.vision.analyze_screenshot(frame)
            
            # Add context from recent game memory
            analysis = self._add_memory_context(analysis)
            
            # Log what the AI understands
            self._log_ai_understanding(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error("AI analysis failed: %s", e)
            return None
    
    def _add_memory_context(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Add context from recent game history to improve decision making"""
        try:
            # Add recent actions as context
            recent_actions = self.controller.get_recent_actions(3)
            analysis["recent_actions"] = recent_actions
            
            # Store current analysis in memory
            memory_entry = {
                "timestamp": time.time(),
                "game_state": analysis.get("game_state", "unknown"),
                "understanding": analysis.get("understanding", ""),
                "decision": analysis.get("decision", "")
            }
            
            self.game_memory.append(memory_entry)
            
            # Keep only recent memory to prevent bloat
            if len(self.game_memory) > 20:
                self.game_memory = self.game_memory[-20:]
            
            return analysis
            
        except Exception as e:
            logger.debug("Failed to add memory context: %s", e)
            return analysis
    
    def _log_ai_understanding(self, analysis: Dict[str, Any]):
        """Log what the AI sees and understands"""
        try:
            understanding = analysis.get("understanding", "No understanding available")
            game_state = analysis.get("game_state", "unknown")
            decision = analysis.get("decision", "No decision")
            confidence = analysis.get("confidence", 0.0)
            
            # Log AI's understanding
            logger.info("🧠 AI UNDERSTANDS: %s", understanding[:100] + "..." if len(understanding) > 100 else understanding)
            logger.info("🎮 Game State: %s (confidence: %.1f)", game_state, confidence)
            logger.info("🎯 AI Decision: %s", decision[:80] + "..." if len(decision) > 80 else decision)
            
            # Log narration if enabled
            if self.enable_narration and "narration" in analysis:
                narration = analysis["narration"]
                if narration and narration.strip():
                    logger.info("🎤 AI SAYS: \"%s\"", narration)
            
        except Exception as e:
            logger.debug("Failed to log AI understanding: %s", e)
    
    def _execute_intelligent_action(self, frame: np.ndarray) -> bool:
        """
        Execute the action decided by AI analysis.
        This replaces the old coordinate-clicking with intelligent action execution.
        """
        if not self.current_analysis:
            return False
        
        try:
            action_taken = self.controller.execute_ai_decision(self.current_analysis, frame)
            return action_taken
            
        except Exception as e:
            logger.error("Failed to execute AI action: %s", e)
            return False
    
    def quick_state_check(self, frame: np.ndarray) -> str:
        """Quick check of game state for main loop logic"""
        try:
            return self.vision.quick_state_check(frame)
        except Exception as e:
            logger.debug("Quick state check failed: %s", e)
            return "unknown"
    
    def handle_emergency_situation(self, frame: np.ndarray) -> bool:
        """
        Handle emergency situations (popups, crashes, etc.) with AI intelligence.
        This replaces the old OCR-based popup handling.
        """
        try:
            # Use AI to detect and handle emergencies
            return self.controller.handle_popup_intelligently(frame)
            
        except Exception as e:
            logger.debug("AI emergency handling failed: %s", e)
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring"""
        return {
            "analysis_count": self.analysis_count,
            "action_count": self.action_count,
            "memory_entries": len(self.game_memory),
            "last_analysis_time": self.last_analysis_time,
            "current_state": self.current_analysis.get("game_state", "unknown") if self.current_analysis else "none"
        }
    
    def reset_state(self):
        """Reset AI state (useful for testing or error recovery)"""
        self.current_analysis = None
        self.game_memory.clear()
        self.controller.reset_cooldowns()
        self.last_analysis_time = 0
        logger.info("🔄 AI Brain state reset")