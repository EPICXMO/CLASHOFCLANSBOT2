"""
AI Brain - Main thinking loop that coordinates vision and action.

This module implements the core AI logic that sees, thinks, and acts like a human player.
It includes memory, narration, and strategic decision-making capabilities.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Deque
from collections import deque
from dataclasses import dataclass

import numpy as np

from ai_vision import AIVision, AIAnalysis
from intelligent_controller import IntelligentController

logger = logging.getLogger(__name__)


@dataclass
class GameMemory:
    """Memory of past game states and actions for learning."""
    timestamp: float
    analysis: AIAnalysis
    ui_data: Dict
    action_taken: bool
    reward_signal: float = 0.0


class AIBrain:
    """Main AI brain that coordinates vision, thinking, and action."""
    
    def __init__(self, config: Dict[str, Any], vision: AIVision, controller: IntelligentController):
        """Initialize AI brain with components."""
        self.config = config
        self.vision = vision
        self.controller = controller
        
        # AI configuration
        ai_config = config.get("ai", {})
        self.analysis_interval = ai_config.get("analysis_interval", 2.0)
        self.enable_narration = ai_config.get("enable_narration", True)
        
        # Memory and learning
        self.memory: Deque[GameMemory] = deque(maxlen=100)  # Keep last 100 memories
        self.last_analysis_time = 0.0
        self.consecutive_waits = 0
        self.game_session_start = time.time()
        
        # Performance tracking
        self.total_analyses = 0
        self.successful_actions = 0
        self.failed_actions = 0
        
        logger.info("🧠 AI Brain initialized - Ready to think and act!")
    
    def think_and_act(self, screenshot: np.ndarray, ui_data: Dict) -> bool:
        """
        Main thinking loop: SEE -> THINK -> NARRATE -> ACT
        
        Args:
            screenshot: Current game screenshot
            ui_data: UI detection data from existing vision system
            
        Returns:
            True if an action was taken, False otherwise
        """
        current_time = time.time()
        
        # Check if it's time for a new analysis
        if current_time - self.last_analysis_time < self.analysis_interval:
            return False
        
        try:
            # 1. SEE: Analyze with GPT-4 Vision
            analysis = self.vision.analyze_screenshot(screenshot, ui_data)
            if not analysis:
                logger.warning("🚨 AI vision analysis failed")
                return False
            
            self.total_analyses += 1
            self.last_analysis_time = current_time
            
            # 2. THINK: Log understanding and integrate memory
            self._log_ai_thoughts(analysis, ui_data)
            self._update_strategy_based_on_memory(analysis)
            
            # 3. NARRATE: Explain decisions like a streamer
            if self.enable_narration:
                self._narrate(analysis)
            
            # 4. ACT: Execute intelligent actions
            action_taken = self._execute_action(analysis, ui_data)
            
            # 5. REMEMBER: Store experience for learning
            self._store_memory(analysis, ui_data, action_taken)
            
            return action_taken
            
        except Exception as e:
            logger.error("🚨 AI Brain error in think_and_act: %s", e, exc_info=True)
            return False
    
    def _log_ai_thoughts(self, analysis: AIAnalysis, ui_data: Dict) -> None:
        """Log AI's understanding and thought process."""
        logger.info("🧠 AI UNDERSTANDS: %s - %s", analysis.game_state, analysis.situation)
        logger.info("🎯 STRATEGY: %s", analysis.strategy)
        logger.info("⚡ THREAT LEVEL: %s | CONFIDENCE: %.1f%%", 
                   analysis.threat_level, analysis.confidence * 100)
        
        # Log detected game elements for context
        elixir = ui_data.get("elixir", 0)
        in_battle = ui_data.get("in_battle", False)
        towers = len(ui_data.get("yolo", {}).get("my_tower", []))
        enemy_towers = len(ui_data.get("yolo", {}).get("enemy_tower", []))
        troops = len(ui_data.get("yolo", {}).get("troop", []))
        
        logger.info("📊 GAME STATE: Battle=%s | Elixir=%d | Towers=%d/%d | Troops=%d",
                   in_battle, elixir, towers, enemy_towers, troops)
    
    def _narrate(self, analysis: AIAnalysis) -> None:
        """Narrate AI decisions like a human streamer."""
        if analysis.narration:
            logger.info("🎤 AI SAYS: \"%s\"", analysis.narration)
        
        # Add contextual commentary based on game state
        if analysis.game_state == "in_battle":
            if analysis.threat_level == "high":
                logger.info("🚨 ALERT: High threat situation detected!")
            elif analysis.action_type == "deploy_card":
                logger.info("🎮 PLAY: Making strategic deployment")
            elif analysis.action_type == "wait":
                logger.info("⏳ PATIENCE: Waiting for better opportunity")
        elif analysis.game_state == "menu":
            logger.info("🏠 LOBBY: Navigating menu interface")
    
    def _execute_action(self, analysis: AIAnalysis, ui_data: Dict) -> bool:
        """Execute the recommended action."""
        # Handle consecutive waits to prevent getting stuck
        if analysis.action_type == "wait":
            self.consecutive_waits += 1
            if self.consecutive_waits > 5:
                logger.warning("🔄 Too many consecutive waits, forcing action")
                # Override with a fallback action
                analysis.action_type = "click_button"
                analysis.target_button = "battle"
                analysis.reasoning = "Breaking wait loop with fallback action"
        else:
            self.consecutive_waits = 0
        
        # Execute the action
        success = self.controller.execute_intelligent_action(analysis, ui_data)
        
        if success:
            self.successful_actions += 1
            logger.info("✅ Action executed successfully")
        else:
            self.failed_actions += 1
            logger.warning("❌ Action execution failed")
        
        return success
    
    def _update_strategy_based_on_memory(self, analysis: AIAnalysis) -> None:
        """Update strategy based on recent memory and experience."""
        if len(self.memory) < 5:
            return  # Need some history to learn from
        
        # Analyze recent patterns
        recent_memories = list(self.memory)[-5:]
        recent_actions = [m.analysis.action_type for m in recent_memories]
        recent_threats = [m.analysis.threat_level for m in recent_memories]
        
        # Adjust strategy based on patterns
        if recent_actions.count("wait") >= 3:
            logger.info("🔍 AI LEARNING: Too much waiting recently, being more aggressive")
        
        if recent_threats.count("high") >= 2:
            logger.info("🛡️ AI LEARNING: High threat pattern detected, focusing on defense")
        
        # Adjust analysis interval based on game state
        if analysis.game_state == "in_battle" and analysis.threat_level == "high":
            self.analysis_interval = max(1.0, self.analysis_interval * 0.9)  # Faster analysis
        else:
            self.analysis_interval = min(3.0, self.analysis_interval * 1.05)  # Slower analysis
    
    def _store_memory(self, analysis: AIAnalysis, ui_data: Dict, action_taken: bool) -> None:
        """Store current experience in memory for learning."""
        memory = GameMemory(
            timestamp=time.time(),
            analysis=analysis,
            ui_data=ui_data.copy(),
            action_taken=action_taken,
            reward_signal=self._calculate_reward_signal(ui_data)
        )
        self.memory.append(memory)
        
        logger.debug("💾 Memory stored: %s | Action: %s | Reward: %.1f", 
                    analysis.situation, action_taken, memory.reward_signal)
    
    def _calculate_reward_signal(self, ui_data: Dict) -> float:
        """Calculate a simple reward signal based on game state."""
        reward = 0.0
        
        # Positive rewards
        gold = ui_data.get("gold", 0)
        if gold > 0:
            reward += gold * 0.01  # Small reward for gold
        
        # Battle performance indicators
        my_towers = len(ui_data.get("yolo", {}).get("my_tower", []))
        enemy_towers = len(ui_data.get("yolo", {}).get("enemy_tower", []))
        
        # Reward for having more towers than enemy
        if my_towers > enemy_towers:
            reward += (my_towers - enemy_towers) * 10
        
        # Small reward for being in battle (engagement)
        if ui_data.get("in_battle", False):
            reward += 1.0
        
        return reward
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get AI performance statistics."""
        session_time = time.time() - self.game_session_start
        success_rate = 0.0
        if self.total_analyses > 0:
            success_rate = self.successful_actions / max(1, self.successful_actions + self.failed_actions)
        
        return {
            "session_time_minutes": session_time / 60.0,
            "total_analyses": self.total_analyses,
            "successful_actions": self.successful_actions,
            "failed_actions": self.failed_actions,
            "success_rate": success_rate,
            "memory_size": len(self.memory),
            "current_interval": self.analysis_interval,
            "ai_enabled": self.vision.is_enabled()
        }
    
    def log_performance_summary(self) -> None:
        """Log a summary of AI performance."""
        stats = self.get_performance_stats()
        logger.info("📈 AI PERFORMANCE SUMMARY:")
        logger.info("   Session Time: %.1f minutes", stats["session_time_minutes"])
        logger.info("   Total Analyses: %d", stats["total_analyses"])
        logger.info("   Action Success Rate: %.1f%%", stats["success_rate"] * 100)
        logger.info("   Memory Experiences: %d", stats["memory_size"])
        logger.info("   Current Analysis Interval: %.1fs", stats["current_interval"])
    
    def is_ready(self) -> bool:
        """Check if AI brain is ready to operate."""
        return self.vision.is_enabled() and self.controller.can_act()
    
    def reset_session(self) -> None:
        """Reset AI session (clear memory, reset counters)."""
        self.memory.clear()
        self.game_session_start = time.time()
        self.total_analyses = 0
        self.successful_actions = 0
        self.failed_actions = 0
        self.consecutive_waits = 0
        self.analysis_interval = self.config.get("ai", {}).get("analysis_interval", 2.0)
        logger.info("🔄 AI Brain session reset")