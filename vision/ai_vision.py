"""
Local AI Vision Module for Clash Royale Bot
Uses LLaVA models for advanced game analysis without API costs
Optimized for RTX 3090 GPU acceleration
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

# Handle optional dependencies
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available. Install with: pip install pillow")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Install with: pip install torch")

try:
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Install with: pip install transformers accelerate")


class LocalVisionAI:
    """Local AI vision using LLaVA models for Clash Royale game analysis"""
    
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.ai_cfg = cfg.get("ai", {})
        
        # Model configuration
        self.model_type = self.ai_cfg.get("model_type", "local")
        self.model_name = self.ai_cfg.get("model_name", "llava-hf/llava-v1.6-mistral-7b-hf")
        self.device = self.ai_cfg.get("device", "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
        self.gpu_memory_fraction = self.ai_cfg.get("gpu_memory_fraction", 0.8)
        self.analysis_interval = self.ai_cfg.get("analysis_interval", 1.5)
        self.enable_narration = self.ai_cfg.get("enable_narration", True)
        
        # Model and processor
        self.model = None
        self.processor = None
        self.last_analysis_time = 0.0
        
        # Initialize model if all dependencies available
        if (TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE and PIL_AVAILABLE and 
            self.model_type == "local"):
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize LLaVA model and processor"""
        try:
            logger.info("Initializing local AI vision model: %s", self.model_name)
            
            # Set GPU memory fraction if using CUDA
            if self.device == "cuda" and TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(self.gpu_memory_fraction)
                logger.info("Set GPU memory fraction to %.1f", self.gpu_memory_fraction)
            
            # Load processor
            self.processor = LlavaNextProcessor.from_pretrained(self.model_name)
            
            # Load model with optimizations
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,  # Use FP16 for 2x speed on RTX 3090
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            # Compile model for faster inference on PyTorch 2.0+
            if TORCH_AVAILABLE and hasattr(torch, 'compile'):
                try:
                    self.model = torch.compile(self.model)
                    logger.info("Model compiled for faster inference")
                except Exception as e:
                    logger.warning("Could not compile model: %s", e)
            
            logger.info("Local AI vision model initialized successfully on %s", self.device)
            
        except Exception as e:
            logger.error("Failed to initialize AI vision model: %s", e)
            self.model = None
            self.processor = None
    
    def is_available(self) -> bool:
        """Check if AI vision is available and ready"""
        return (TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE and PIL_AVAILABLE and
                self.model is not None and 
                self.processor is not None and
                self.model_type == "local")
    
    def analyze_screenshot(self, image_array: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Analyze game screenshot using local AI vision
        
        Args:
            image_array: Screenshot as numpy array (H, W, 3)
            
        Returns:
            Dict with analysis results or None if not available
        """
        if not self.is_available():
            return None
        
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_analysis_time < self.analysis_interval:
            return None
        
        try:
            # Convert numpy array to PIL Image
            if not PIL_AVAILABLE:
                return None
            image = Image.fromarray(image_array)
            
            # Create analysis prompt
            prompt = self._create_analysis_prompt()
            
            # Process with local model
            inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)
            
            # Generate response with optimizations
            with torch.no_grad():
                if TORCH_AVAILABLE and hasattr(torch.backends.cuda, 'sdp_flash_attention'):
                    # Use flash attention if available
                    with torch.backends.cuda.sdp_flash_attention():
                        output = self.model.generate(
                            **inputs,
                            max_new_tokens=1000,
                            do_sample=True,
                            temperature=0.1,
                            top_p=0.9,
                            pad_token_id=self.processor.tokenizer.eos_token_id
                        )
                else:
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=1000,
                        do_sample=True,
                        temperature=0.1,
                        top_p=0.9,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
            
            # Decode response
            response = self.processor.decode(output[0], skip_special_tokens=True)
            
            # Parse structured response
            analysis = self._parse_response(response)
            
            self.last_analysis_time = current_time
            
            if self.enable_narration:
                logger.info("AI Analysis: %s", analysis.get("summary", "No summary"))
            
            return analysis
            
        except Exception as e:
            logger.error("AI vision analysis failed: %s", e)
            return None
    
    def _create_analysis_prompt(self) -> str:
        """Create expert prompt for Clash Royale analysis"""
        return """<image>
You are an expert Clash Royale strategist analyzing a game screenshot. Provide a structured analysis in JSON format with the following information:

{
  "game_state": "menu|battle|victory|defeat",
  "battle_phase": "early|mid|late" (if in battle),
  "my_elixir": estimated_elixir_count,
  "enemy_pressure": "low|medium|high",
  "my_towers": {
    "king": "healthy|damaged|destroyed",
    "left": "healthy|damaged|destroyed", 
    "right": "healthy|damaged|destroyed"
  },
  "enemy_towers": {
    "king": "healthy|damaged|destroyed",
    "left": "healthy|damaged|destroyed",
    "right": "healthy|damaged|destroyed"
  },
  "troops_on_field": {
    "my_troops": ["list", "of", "visible", "troops"],
    "enemy_troops": ["list", "of", "visible", "enemy", "troops"]
  },
  "recommended_action": "wait|deploy_defense|push_left|push_right|spell_tower|counter_push",
  "card_to_play": "card_name_or_null",
  "deployment_zone": "back|mid|bridge|tower" (if card recommended),
  "priority": "high|medium|low",
  "reasoning": "brief explanation of recommendation",
  "summary": "one sentence battle summary"
}

Focus on strategic analysis for optimal card deployment and timing."""
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response into structured data"""
        try:
            # Extract JSON from response (handle cases where model adds extra text)
            import json
            
            # Find JSON block in response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Fallback: create basic structure from text
                return {
                    "game_state": "unknown",
                    "summary": response[:200],
                    "recommended_action": "wait",
                    "priority": "low",
                    "reasoning": "Failed to parse structured response"
                }
                
        except Exception as e:
            logger.warning("Failed to parse AI response: %s", e)
            return {
                "game_state": "unknown", 
                "summary": "Analysis failed",
                "recommended_action": "wait",
                "priority": "low",
                "reasoning": f"Parse error: {str(e)}"
            }
    
    def get_strategic_advice(self, analysis: Dict[str, Any]) -> str:
        """Convert analysis into actionable strategic advice"""
        if not analysis:
            return "No AI analysis available"
        
        action = analysis.get("recommended_action", "wait")
        card = analysis.get("card_to_play", "")
        zone = analysis.get("deployment_zone", "")
        priority = analysis.get("priority", "low")
        reasoning = analysis.get("reasoning", "")
        
        advice_parts = []
        
        if action != "wait":
            if card and zone:
                advice_parts.append(f"Deploy {card} in {zone}")
            else:
                advice_parts.append(f"Action: {action}")
        
        if priority == "high":
            advice_parts.append("(URGENT)")
        
        if reasoning:
            advice_parts.append(f"- {reasoning}")
        
        return " ".join(advice_parts) if advice_parts else "Wait and observe"
    
    def cleanup(self):
        """Clean up GPU memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cleared")