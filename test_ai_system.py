"""
Tests for the AI-powered intelligent bot system.
"""

import pytest
import time
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from ai_vision import VisionAI
from ai_brain import AIBrain
from intelligent_controller import IntelligentController


class TestAIVision:
    @patch('openai.OpenAI')
    def test_vision_ai_initialization(self, mock_openai):
        """Test VisionAI initializes correctly"""
        config = {
            "ai": {
                "openai_api_key": "test-key",
                "model": "gpt-4o",
                "max_tokens": 1000,
                "temperature": 0.3
            }
        }
        vision = VisionAI(config)
        assert vision.model == "gpt-4o"
        assert vision.max_tokens == 1000
        assert vision.temperature == 0.3
    
    @patch('openai.OpenAI')
    def test_frame_to_base64(self, mock_openai):
        """Test frame conversion to base64"""
        config = {"ai": {"openai_api_key": "test-key"}}
        vision = VisionAI(config)
        
        # Create a test frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # Red channel
        
        # Convert to base64
        base64_str = vision._frame_to_base64(frame)
        
        # Should be a valid base64 string
        assert isinstance(base64_str, str)
        assert len(base64_str) > 0
        # Basic base64 validation
        import base64
        try:
            base64.b64decode(base64_str)
            valid_base64 = True
        except Exception:
            valid_base64 = False
        assert valid_base64
    
    @patch('openai.OpenAI')
    def test_fallback_analysis(self, mock_openai):
        """Test fallback analysis when AI fails"""
        config = {"ai": {"openai_api_key": "test-key"}}
        vision = VisionAI(config)
        
        fallback = vision._fallback_analysis()
        
        # Check required fields exist
        assert "game_state" in fallback
        assert "understanding" in fallback
        assert "decision" in fallback
        assert "action_type" in fallback
        assert "action_details" in fallback
        assert fallback["confidence"] == 0.1
    
    @patch('openai.OpenAI')
    def test_parse_analysis_valid_json(self, mock_openai):
        """Test parsing valid JSON response"""
        config = {"ai": {"openai_api_key": "test-key"}}
        vision = VisionAI(config)
        
        json_response = """{
            "game_state": "battle",
            "understanding": "I see a battle in progress",
            "decision": "Deploy hog rider",
            "action_type": "deploy_card",
            "action_details": {"card_index": 1}
        }"""
        
        result = vision._parse_analysis(json_response)
        
        assert result["game_state"] == "battle"
        assert result["understanding"] == "I see a battle in progress"
        assert result["decision"] == "Deploy hog rider"
        assert result["action_type"] == "deploy_card"
    
    @patch('openai.OpenAI')
    def test_parse_analysis_invalid_json(self, mock_openai):
        """Test parsing invalid JSON falls back gracefully"""
        config = {"ai": {"openai_api_key": "test-key"}}
        vision = VisionAI(config)
        
        invalid_response = "This is not JSON at all!"
        
        result = vision._parse_analysis(invalid_response)
        
        # Should return fallback analysis
        assert result["game_state"] == "unknown"
        assert result["confidence"] == 0.1


class TestIntelligentController:
    def test_controller_initialization(self):
        """Test IntelligentController initializes correctly"""
        mock_adb = Mock()
        mock_vision = Mock()
        config = {}
        
        controller = IntelligentController(mock_adb, mock_vision, config)
        
        assert controller.adb == mock_adb
        assert controller.vision_ai == mock_vision
        assert controller.min_action_interval == 0.5
        assert len(controller.action_history) == 0
    
    def test_execute_wait_action(self):
        """Test wait action doesn't execute anything"""
        mock_adb = Mock()
        mock_vision = Mock()
        config = {}
        
        controller = IntelligentController(mock_adb, mock_vision, config)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        analysis = {
            "action_type": "wait",
            "action_details": {"reasoning": "Waiting for better timing"}
        }
        
        result = controller.execute_ai_decision(analysis, frame)
        
        assert result == False  # No action taken
        assert not mock_adb.called  # ADB should not be called for wait
    
    def test_action_cooldown_respected(self):
        """Test that action cooldowns are respected"""
        mock_adb = Mock()
        mock_vision = Mock()
        config = {}
        
        controller = IntelligentController(mock_adb, mock_vision, config)
        controller.last_action_time = time.time()  # Set recent action time
        
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        analysis = {
            "action_type": "click_battle",
            "action_details": {"target_x": 0.5, "target_y": 0.5}
        }
        
        result = controller.execute_ai_decision(analysis, frame)
        
        assert result == False  # Action blocked by cooldown


class TestAIBrain:
    def test_ai_brain_initialization(self):
        """Test AIBrain initializes correctly"""
        mock_adb = Mock()
        config = {
            "ai": {
                "openai_api_key": "test-key",
                "analysis_interval": 2.0,
                "enable_narration": True
            }
        }
        
        with patch('ai_brain.VisionAI') as mock_vision_class, \
             patch('ai_brain.IntelligentController') as mock_controller_class:
            
            brain = AIBrain(mock_adb, config)
            
            assert brain.analysis_interval == 2.0
            assert brain.enable_narration == True
            assert brain.analysis_count == 0
            assert brain.action_count == 0
    
    def test_performance_stats(self):
        """Test performance statistics tracking"""
        mock_adb = Mock()
        config = {"ai": {"openai_api_key": "test-key"}}
        
        with patch('ai_brain.VisionAI'), patch('ai_brain.IntelligentController'):
            brain = AIBrain(mock_adb, config)
            brain.analysis_count = 5
            brain.action_count = 3
            
            stats = brain.get_performance_stats()
            
            assert stats["analysis_count"] == 5
            assert stats["action_count"] == 3
            assert "memory_entries" in stats
            assert "current_state" in stats


# Integration test for the complete system
class TestAISystem:
    @patch('openai.OpenAI')
    def test_system_without_api_key(self, mock_openai):
        """Test system behavior without API key"""
        config = {"ai": {"openai_api_key": ""}}
        
        vision = VisionAI(config)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Should fall back gracefully when no API key
        result = vision.analyze_screenshot(frame)
        
        assert result["game_state"] == "unknown"
        assert result["confidence"] == 0.1
    
    def test_memory_management(self):
        """Test that memory doesn't grow unbounded"""
        mock_adb = Mock()
        config = {"ai": {"openai_api_key": "test-key"}}
        
        with patch('ai_brain.VisionAI'), patch('ai_brain.IntelligentController'):
            brain = AIBrain(mock_adb, config)
            
            # Add lots of memory entries
            for i in range(50):
                brain.game_memory.append({"test": i})
            
            # Add one more - should trigger cleanup
            analysis = {"game_state": "test", "understanding": "test", "decision": "test"}
            brain._add_memory_context(analysis)
            
            # Should be limited to 20 entries
            assert len(brain.game_memory) <= 20


if __name__ == "__main__":
    import time
    pytest.main([__file__, "-v"])