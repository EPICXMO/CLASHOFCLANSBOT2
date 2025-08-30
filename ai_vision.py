"""
AI Vision Module - GPT-4 Vision integration for Clash Royale screenshot analysis.

This module provides intelligent visual analysis of game screenshots using OpenAI's GPT-4 Vision API.
It returns structured game state analysis and strategic recommendations.
"""

import base64
import io
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import json

import numpy as np
from PIL import Image
import openai

logger = logging.getLogger(__name__)


@dataclass
class AIAnalysis:
    """Structured result from AI vision analysis."""
    # Game state understanding
    game_state: str  # "menu", "in_battle", "end_screen", "loading"
    elixir_level: int  # 0-10
    situation: str  # Brief description of current situation
    
    # Strategic analysis
    threat_level: str  # "low", "medium", "high"
    opportunity: str  # Description of current opportunity
    strategy: str  # Recommended strategy
    
    # Action recommendation
    action_type: str  # "deploy_card", "wait", "click_button", "no_action"
    target_card: Optional[int] = None  # 0-3 for card slots
    target_position: Optional[Tuple[float, float]] = None  # Normalized (x, y)
    target_button: Optional[str] = None  # Button to click if action_type is click_button
    
    # AI personality and narration
    narration: str = ""  # What the AI would say as a streamer
    confidence: float = 0.0  # 0.0-1.0 confidence in analysis
    reasoning: str = ""  # Why this decision was made


class AIVision:
    """GPT-4 Vision API integration for game analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize AI vision with configuration."""
        ai_config = config.get("ai", {})
        self.api_key = ai_config.get("openai_api_key", "")
        self.model = ai_config.get("model", "gpt-4o")
        self.max_tokens = ai_config.get("max_tokens", 500)
        self.temperature = ai_config.get("temperature", 0.7)
        self.enable_narration = ai_config.get("enable_narration", True)
        
        if not self.api_key:
            logger.warning("OpenAI API key not configured. AI vision will be disabled.")
            self.enabled = False
        else:
            self.enabled = True
            openai.api_key = self.api_key
            logger.info("AI Vision initialized with model %s", self.model)
    
    def _encode_image(self, frame: np.ndarray) -> str:
        """Convert BGR numpy array to base64 encoded image."""
        # Convert BGR to RGB
        rgb_frame = frame[:, :, ::-1]
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)
        
        # Resize if too large (GPT-4V works better with smaller images)
        if pil_image.width > 1024 or pil_image.height > 1024:
            pil_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _create_analysis_prompt(self) -> str:
        """Create the system prompt for game analysis."""
        return """You are an expert Clash Royale player and AI analyst. Analyze the screenshot and provide strategic guidance.

RESPOND IN VALID JSON FORMAT with these fields:
{
    "game_state": "menu|in_battle|end_screen|loading",
    "elixir_level": 0-10,
    "situation": "brief description of what's happening",
    "threat_level": "low|medium|high", 
    "opportunity": "current opportunity description",
    "strategy": "recommended strategy",
    "action_type": "deploy_card|wait|click_button|no_action",
    "target_card": 0-3 or null,
    "target_position": [x, y] normalized 0-1 or null,
    "target_button": "battle|ok|collect|upgrade" or null,
    "narration": "what you'd say as an engaging streamer",
    "confidence": 0.0-1.0,
    "reasoning": "why you made this decision"
}

ANALYSIS GUIDELINES:
- If you see towers, elixir bar, and cards at bottom: game_state = "in_battle"
- If you see menu buttons, chests, or main interface: game_state = "menu"
- Look for enemy troops and assess threats
- Consider elixir cost vs available elixir
- Target positions: (0.5, 0.6) = center defense, (0.2, 0.7) = left lane, (0.8, 0.7) = right lane
- Be engaging and strategic in narration like a pro player would commentate
- Explain your reasoning clearly"""

    def analyze_screenshot(self, frame: np.ndarray, ui_data: Optional[Dict] = None) -> Optional[AIAnalysis]:
        """
        Analyze a game screenshot using GPT-4 Vision.
        
        Args:
            frame: BGR numpy array of the screenshot
            ui_data: Optional UI detection data from existing vision system
            
        Returns:
            AIAnalysis object with strategic recommendations, or None if analysis fails
        """
        if not self.enabled:
            logger.debug("AI vision disabled, skipping analysis")
            return None
            
        try:
            # Encode image for API
            base64_image = self._encode_image(frame)
            
            # Create enhanced prompt with UI context if available
            prompt = self._create_analysis_prompt()
            if ui_data:
                prompt += f"\n\nADDITIONAL CONTEXT: {json.dumps(ui_data, default=str)}"
            
            # Call GPT-4 Vision API
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this Clash Royale screenshot and provide strategic guidance in JSON format."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Parse response
            content = response.choices[0].message.content
            logger.debug("Raw AI response: %s", content)
            
            # Extract JSON from response (handle cases where AI adds extra text)
            try:
                # Try to find JSON in the response
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    analysis_data = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
                    
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("Failed to parse AI response as JSON: %s", e)
                # Return fallback analysis
                return AIAnalysis(
                    game_state="unknown",
                    elixir_level=0,
                    situation="Failed to analyze",
                    threat_level="low",
                    opportunity="Unknown",
                    strategy="Wait and observe",
                    action_type="wait",
                    narration="🤖 Having trouble reading the screen right now!",
                    confidence=0.1,
                    reasoning="JSON parsing failed"
                )
            
            # Create AIAnalysis object
            analysis = AIAnalysis(
                game_state=analysis_data.get("game_state", "unknown"),
                elixir_level=int(analysis_data.get("elixir_level", 0)),
                situation=analysis_data.get("situation", ""),
                threat_level=analysis_data.get("threat_level", "low"),
                opportunity=analysis_data.get("opportunity", ""),
                strategy=analysis_data.get("strategy", ""),
                action_type=analysis_data.get("action_type", "wait"),
                target_card=analysis_data.get("target_card"),
                target_position=tuple(analysis_data["target_position"]) if analysis_data.get("target_position") else None,
                target_button=analysis_data.get("target_button"),
                narration=analysis_data.get("narration", ""),
                confidence=float(analysis_data.get("confidence", 0.5)),
                reasoning=analysis_data.get("reasoning", "")
            )
            
            logger.info("AI Analysis: %s | %s | Action: %s", 
                       analysis.situation, analysis.strategy, analysis.action_type)
            
            return analysis
            
        except Exception as e:
            logger.error("AI vision analysis failed: %s", e, exc_info=True)
            return None

    def is_enabled(self) -> bool:
        """Check if AI vision is properly configured and enabled."""
        return self.enabled