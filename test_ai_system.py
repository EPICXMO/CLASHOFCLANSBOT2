#!/usr/bin/env python3
"""
Test script for AI-powered bot components.
Tests core functionality without requiring OpenAI API key.
"""

import sys
import numpy as np
from unittest.mock import Mock, patch

def test_ai_vision_without_api():
    """Test AI vision initialization without API key."""
    print("🧪 Testing AI Vision without API key...")
    
    from ai_vision import AIVision
    
    # Test with empty config (no API key)
    config = {"ai": {"openai_api_key": ""}}
    vision = AIVision(config)
    
    assert not vision.is_enabled(), "Vision should be disabled without API key"
    print("✅ AI Vision correctly disabled without API key")
    
    # Test analysis returns None when disabled
    fake_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    result = vision.analyze_screenshot(fake_frame)
    assert result is None, "Analysis should return None when disabled"
    print("✅ AI Vision returns None when disabled")

def test_intelligent_controller():
    """Test intelligent controller with mock components."""
    print("🧪 Testing Intelligent Controller...")
    
    from intelligent_controller import IntelligentController
    from ai_vision import AIAnalysis
    
    # Mock ADB controller
    mock_adb = Mock()
    mock_adb.click = Mock()
    mock_adb.tap_norm = Mock()
    
    controller = IntelligentController(mock_adb)
    
    # Test wait action
    analysis = AIAnalysis(
        game_state="in_battle",
        elixir_level=5,
        situation="test situation",
        threat_level="low",
        opportunity="test",
        strategy="test strategy",
        action_type="wait",
        reasoning="test wait"
    )
    
    ui_data = {}
    result = controller.execute_intelligent_action(analysis, ui_data)
    assert result, "Wait action should succeed"
    print("✅ Wait action works correctly")
    
    # Test deploy card action (wait for cooldown first)
    import time
    time.sleep(0.6)  # Wait for cooldown
    
    analysis.action_type = "deploy_card"
    analysis.target_card = 0
    analysis.target_position = (0.5, 0.6)
    
    ui_data = {"card_slots": {0: (100, 200), 1: (300, 200)}}
    result = controller.execute_intelligent_action(analysis, ui_data)
    assert result, "Deploy card action should succeed"
    mock_adb.click.assert_called()
    mock_adb.tap_norm.assert_called()
    print("✅ Deploy card action works correctly")

def test_ai_brain():
    """Test AI brain coordination."""
    print("🧪 Testing AI Brain...")
    
    from ai_brain import AIBrain
    from ai_vision import AIVision
    from intelligent_controller import IntelligentController
    
    # Mock components
    config = {"ai": {"analysis_interval": 1.0, "enable_narration": True}}
    mock_vision = Mock()
    mock_controller = Mock()
    
    brain = AIBrain(config, mock_vision, mock_controller)
    
    # Test initialization
    assert brain.analysis_interval == 1.0
    assert brain.enable_narration == True
    print("✅ AI Brain initialized correctly")
    
    # Test memory storage
    from ai_vision import AIAnalysis
    test_analysis = AIAnalysis(
        game_state="test",
        elixir_level=0,
        situation="test",
        threat_level="low",
        opportunity="test",
        strategy="test",
        action_type="wait"
    )
    
    brain._store_memory(test_analysis, {}, True)
    assert len(brain.memory) == 1
    print("✅ Memory storage works correctly")

def test_integration():
    """Test basic integration between components."""
    print("🧪 Testing Component Integration...")
    
    from utils.config import load_config
    from ai_vision import AIVision
    from actions.adb import ADBController
    from intelligent_controller import IntelligentController
    from ai_brain import AIBrain
    
    # Load real config
    config = load_config()
    
    # Initialize components (without API key)
    config["ai"]["openai_api_key"] = ""  # Ensure no API key
    
    adb = ADBController(dry_run=True)
    vision = AIVision(config)
    intelligent_controller = IntelligentController(adb)
    brain = AIBrain(config, vision, intelligent_controller)
    
    # Test that brain recognizes vision is disabled
    assert not brain.is_ready(), "Brain should not be ready without vision"
    print("✅ Integration test passed - Brain correctly detects disabled vision")

def main():
    """Run all tests."""
    print("🤖 Testing AI-Powered Bot Components")
    print("=" * 50)
    
    try:
        test_ai_vision_without_api()
        test_intelligent_controller()
        test_ai_brain()
        test_integration()
        
        print("=" * 50)
        print("🎉 All tests passed! AI system is working correctly.")
        print("💡 To enable full AI functionality, add your OpenAI API key to config.yaml")
        return 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())