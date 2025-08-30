"""
AI Vision Module - GPT-4 Vision integration for intelligent screenshot analysis.
This replaces the traditional rule-based detection with true AI understanding.
"""

import base64
import io
import logging
import time
from typing import Dict, Any, Tuple, Optional

import numpy as np
from PIL import Image
import openai

logger = logging.getLogger(__name__)


class VisionAI:
    """
    AI Vision system that uses GPT-4 Vision to understand screenshots like ChatGPT would.
    This is the core of the intelligent bot - it sees and understands what's happening.
    """
    
    def __init__(self, config: Dict[str, Any]):
        ai_config = config.get("ai", {})
        self.client = openai.OpenAI(api_key=ai_config.get("openai_api_key", ""))
        self.model = ai_config.get("model", "gpt-4o")
        self.max_tokens = ai_config.get("max_tokens", 1000)
        self.temperature = ai_config.get("temperature", 0.3)
        self.vision_detail = ai_config.get("vision_detail", "low")
        
        if not ai_config.get("openai_api_key"):
            logger.warning("OpenAI API key not configured. AI vision will not work.")
    
    def analyze_screenshot(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyze a screenshot using GPT-4 Vision to understand what's happening.
        Returns structured analysis with understanding, strategy, and actions.
        """
        try:
            # Convert frame to base64 for API
            image_base64 = self._frame_to_base64(frame)
            
            # Create the analysis prompt
            prompt = self._create_analysis_prompt()
            
            # Call GPT-4 Vision
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}",
                                    "detail": self.vision_detail
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Parse the response
            analysis_text = response.choices[0].message.content
            return self._parse_analysis(analysis_text)
            
        except Exception as e:
            logger.error("AI vision analysis failed: %s", e)
            return self._fallback_analysis()
    
    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert numpy frame to base64 string for API"""
        # Convert BGR to RGB
        rgb_frame = frame[:, :, ::-1] if len(frame.shape) == 3 else frame
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)
        
        # Resize if too large (cost optimization)
        max_size = 1024
        if max(pil_image.size) > max_size:
            pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode()
    
    def _create_analysis_prompt(self) -> str:
        """Create the analysis prompt for GPT-4 Vision"""
        return """
You are an expert Clash Royale player and AI vision system. Analyze this screenshot and provide a detailed response in the following JSON format:

{
    "game_state": "battle|main_menu|loading|victory|defeat|chest_screen|unknown",
    "understanding": "Detailed description of what you see and understand about the current situation",
    "elixir_estimate": 5,
    "my_towers": 3,
    "enemy_towers": 3,
    "troops_visible": ["hog_rider", "skeleton_army"],
    "strategic_situation": "Description of the current strategic state",
    "decision": "What action you would take and why",
    "action_type": "click_battle|deploy_card|defend|attack|wait|click_button",
    "action_details": {
        "card_index": 2,
        "target_x": 0.5,
        "target_y": 0.6,
        "reasoning": "Why this action makes sense"
    },
    "narration": "What you would say if streaming this game (excited, strategic commentary)",
    "confidence": 0.9
}

Be strategic, adaptive, and explain your reasoning like a pro player would. Focus on:
1. What's actually happening in the game
2. The strategic implications 
3. What the best response would be
4. Why that response makes sense

Respond ONLY with valid JSON.
"""
    
    def _parse_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """Parse the GPT-4 Vision response into structured data"""
        try:
            # Extract JSON from the response
            import json
            
            # Find JSON in the response (handle markdown code blocks)
            json_start = analysis_text.find('{')
            json_end = analysis_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = analysis_text[json_start:json_end]
                parsed = json.loads(json_str)
                
                # Validate required fields
                required_fields = ["game_state", "understanding", "decision", "action_type"]
                for field in required_fields:
                    if field not in parsed:
                        parsed[field] = "unknown"
                
                # Ensure action_details exists
                if "action_details" not in parsed:
                    parsed["action_details"] = {}
                
                return parsed
            else:
                logger.warning("No valid JSON found in AI response")
                return self._fallback_analysis()
                
        except json.JSONDecodeError as e:
            logger.error("Failed to parse AI response as JSON: %s", e)
            logger.debug("Raw response: %s", analysis_text[:500])
            return self._fallback_analysis()
        except Exception as e:
            logger.error("Error parsing AI analysis: %s", e)
            return self._fallback_analysis()
    
    def _fallback_analysis(self) -> Dict[str, Any]:
        """Fallback analysis when AI vision fails"""
        return {
            "game_state": "unknown",
            "understanding": "AI vision analysis failed, using fallback mode",
            "elixir_estimate": 5,
            "my_towers": 3,
            "enemy_towers": 3,
            "troops_visible": [],
            "strategic_situation": "Unable to analyze - AI vision offline",
            "decision": "Wait and retry analysis",
            "action_type": "wait",
            "action_details": {
                "reasoning": "AI vision failed, waiting for next cycle"
            },
            "narration": "Having some technical difficulties with my vision system, one moment...",
            "confidence": 0.1
        }
    
    def quick_state_check(self, frame: np.ndarray) -> str:
        """Quick analysis to determine basic game state"""
        try:
            image_base64 = self._frame_to_base64(frame)
            
            prompt = """
Look at this Clash Royale screenshot and tell me the current game state in ONE WORD:
- "battle" if in a battle/match
- "menu" if at main menu
- "loading" if loading screen
- "victory" if victory screen
- "defeat" if defeat screen  
- "chest" if opening chests
- "unknown" if unclear

Respond with ONLY the word, nothing else.
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}",
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip().lower()
            
        except Exception as e:
            logger.debug("Quick state check failed: %s", e)
            return "unknown"