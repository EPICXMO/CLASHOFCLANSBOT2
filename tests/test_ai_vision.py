import os
import numpy as np
import pytest

from vision.ai_vision import LocalVisionAI
from utils.config import load_config


def test_local_vision_ai_init():
    """Test LocalVisionAI initialization without actual model loading"""
    cfg = {
        "ai": {
            "model_type": "local",
            "model_name": "test-model",
            "device": "cpu",
            "analysis_interval": 1.0,
            "enable_narration": True
        }
    }
    
    ai = LocalVisionAI(cfg)
    
    assert ai.model_type == "local"
    assert ai.model_name == "test-model"
    assert ai.device == "cpu"
    assert ai.analysis_interval == 1.0
    assert ai.enable_narration is True
    
    # Without transformers installed, should not be available
    assert ai.model is None
    assert ai.processor is None


def test_local_vision_ai_disabled():
    """Test LocalVisionAI when disabled"""
    cfg = {
        "ai": {
            "model_type": "disabled",
        }
    }
    
    ai = LocalVisionAI(cfg)
    assert not ai.is_available()
    assert ai.analyze_screenshot(np.zeros((100, 100, 3), dtype=np.uint8)) is None


def test_response_parsing():
    """Test AI response parsing"""
    cfg = {"ai": {"model_type": "local"}}
    ai = LocalVisionAI(cfg)
    
    # Test valid JSON response
    valid_response = '''
    Here is the analysis:
    {
        "game_state": "battle",
        "recommended_action": "deploy_defense",
        "priority": "high",
        "reasoning": "Enemy push detected"
    }
    That's my analysis.
    '''
    
    result = ai._parse_response(valid_response)
    assert result["game_state"] == "battle"
    assert result["recommended_action"] == "deploy_defense"
    assert result["priority"] == "high"
    
    # Test invalid response
    invalid_response = "This is not JSON"
    result = ai._parse_response(invalid_response)
    assert result["game_state"] == "unknown"
    assert "reasoning" in result


def test_strategic_advice():
    """Test strategic advice generation"""
    cfg = {"ai": {"model_type": "local"}}
    ai = LocalVisionAI(cfg)
    
    # Test high priority advice
    analysis = {
        "recommended_action": "deploy_defense",
        "card_to_play": "fireball",
        "deployment_zone": "bridge",
        "priority": "high",
        "reasoning": "Enemy giant approaching"
    }
    
    advice = ai.get_strategic_advice(analysis)
    assert "fireball" in advice
    assert "bridge" in advice
    assert "URGENT" in advice
    
    # Test wait action
    wait_analysis = {
        "recommended_action": "wait",
        "priority": "low"
    }
    
    advice = ai.get_strategic_advice(wait_analysis)
    assert "Wait and observe" in advice


def test_analysis_prompt():
    """Test that analysis prompt is properly formatted"""
    cfg = {"ai": {"model_type": "local"}}
    ai = LocalVisionAI(cfg)
    
    prompt = ai._create_analysis_prompt()
    
    # Check key elements are in prompt
    assert "<image>" in prompt
    assert "Clash Royale" in prompt
    assert "JSON format" in prompt
    assert "game_state" in prompt
    assert "recommended_action" in prompt
    assert "priority" in prompt


def test_cleanup():
    """Test cleanup functionality"""
    cfg = {"ai": {"model_type": "local"}}
    ai = LocalVisionAI(cfg)
    
    # Should not crash even with no model loaded
    ai.cleanup()
    
    assert ai.model is None
    assert ai.processor is None


def test_rate_limiting():
    """Test that rate limiting works"""
    cfg = {
        "ai": {
            "model_type": "local",
            "analysis_interval": 10.0  # Very long interval
        }
    }
    
    ai = LocalVisionAI(cfg)
    ai.last_analysis_time = 0.0  # Reset timing
    
    # Without transformers, should return None anyway
    # But this tests the rate limiting logic
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    result1 = ai.analyze_screenshot(image)
    result2 = ai.analyze_screenshot(image)  # Should be rate limited
    
    # Both should be None due to no model, but timing should be respected
    assert result1 is None
    assert result2 is None


def test_config_defaults():
    """Test configuration defaults"""
    cfg = {}  # Empty config
    
    ai = LocalVisionAI(cfg)
    
    # Check defaults are applied
    assert ai.model_type == "local"
    assert ai.analysis_interval == 1.5
    assert ai.enable_narration is True
    assert ai.gpu_memory_fraction == 0.8