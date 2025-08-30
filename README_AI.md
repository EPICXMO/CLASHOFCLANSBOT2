# AI-Powered Clash Royale Bot Setup Guide

## Overview

This bot has been transformed from a simple scripted bot into an intelligent AI system that uses **GPT-4 Vision** to analyze screenshots and make strategic decisions like a human player.

## ✨ Key Features

- 🧠 **AI Vision Analysis**: Uses GPT-4 Vision to understand game state
- 🎤 **Streamer-like Narration**: AI explains its decisions and strategy  
- 🎯 **Strategic Decision Making**: Adapts to different game situations
- 📊 **Performance Tracking**: Monitors AI effectiveness and learning
- 🔄 **Backwards Compatible**: Original scripted mode still available

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure OpenAI API Key

Edit `config.yaml` and add your OpenAI API key:

```yaml
ai:
  openai_api_key: "sk-your-openai-api-key-here"
  analysis_interval: 2.0
  enable_narration: true
  model: "gpt-4o"
  max_tokens: 500
  temperature: 0.7
```

### 3. Run the Intelligent Bot

```bash
# AI-powered mode (recommended)
python intelligent_main.py --mode intelligent

# Original scripted mode (fallback)
python intelligent_main.py --mode play
```

## 🎮 How It Works

### AI Decision Flow

```
📸 SCREENSHOT → 🧠 GPT-4 VISION → 🎯 STRATEGIC ANALYSIS → 🎮 ACTION
```

### Example AI Behavior

**Before (Scripted):**
```
Clicking at (540, 1650)...
Deploying card at position 1...
Following script step 5...
```

**After (AI-Powered):**
```
🎤 AI SAYS: "Enemy deployed Giant at back - perfect time to pressure opposite lane!"
🧠 AI UNDERSTANDS: in_battle - Giant push incoming left
🎯 STRATEGY: Counter-push right lane with Hog Rider
🤖 AI Decision: deploy_card - Punish their elixir investment
✅ Action executed successfully
```

## ⚙️ Configuration Options

### AI Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `openai_api_key` | "" | Your OpenAI API key |
| `analysis_interval` | 2.0 | Seconds between AI analyses |
| `enable_narration` | true | Enable AI commentary |
| `model` | "gpt-4o" | OpenAI model to use |
| `max_tokens` | 500 | Max response length |
| `temperature` | 0.7 | AI creativity (0.0-1.0) |

### Performance Tuning

- **Fast Analysis**: Set `analysis_interval: 1.0` for quicker decisions
- **Conservative**: Set `temperature: 0.3` for more predictable play
- **Aggressive**: Set `temperature: 0.9` for more creative strategies

## 📊 Monitoring AI Performance

The AI tracks its own performance and logs detailed analytics:

```
📈 AI PERFORMANCE SUMMARY:
   Session Time: 15.3 minutes
   Total Analyses: 127
   Action Success Rate: 89.2%
   Memory Experiences: 45
   Current Analysis Interval: 2.1s
```

## 🔧 Troubleshooting

### Common Issues

**1. "AI Vision is not enabled"**
- Check that your OpenAI API key is correctly set in `config.yaml`
- Verify the API key has GPT-4 Vision access

**2. "JSON parsing failed"**
- This is usually temporary - the AI will recover automatically
- Check your internet connection to OpenAI

**3. Slow AI responses**
- Increase `analysis_interval` to reduce API calls
- Consider using a smaller model if available

**4. High API costs**
- Increase `analysis_interval` to 3.0+ seconds
- Set `enable_narration: false` to reduce token usage
- Use `max_tokens: 300` for shorter responses

### Debug Mode

Add debug logging to see AI thought process:

```yaml
logging:
  level: DEBUG
```

## 💡 Advanced Usage

### Custom Strategies

You can influence AI behavior by modifying the prompts in `ai_vision.py`:

```python
# Example: Make AI more aggressive
strategy_context = "Focus on aggressive plays and quick pushes"
```

### Memory and Learning

The AI remembers recent decisions and adapts:
- Learns from successful strategies
- Avoids repeated mistakes
- Adjusts timing based on game flow

## 📁 File Structure

```
├── ai_vision.py           # GPT-4 Vision integration
├── intelligent_controller.py  # Smart action execution  
├── ai_brain.py           # Main AI coordination
├── intelligent_main.py   # AI-powered main loop
├── config.yaml          # Configuration (add API key here)
├── main.py              # Original scripted bot
└── logs/ai_episodes/    # AI decision logs
```

## 🛡️ Safety Features

- **Action Cooldowns**: Prevents spam clicking
- **Fallback Logic**: Falls back to basic actions if AI fails
- **Rate Limiting**: Respects API limits
- **Error Recovery**: Continues running even if AI fails

## 🎯 Expected Results

With proper configuration, you should see:

- Strategic card deployment based on game situation
- Intelligent lane selection and timing
- Adaptive behavior that improves over time
- Human-like decision explanations
- Better win rates compared to scripted bot

## 💰 Cost Estimation

**Typical Usage:**
- ~200-500 API calls per hour
- ~$0.10-0.50 per hour (varies by usage)
- Adjust `analysis_interval` to control costs

## 🤝 Support

- Check logs in `logs/` directory for errors
- Review AI decision logs in `logs/ai_episodes/`
- Ensure ADB connection is working with original bot first
- Test with `dry_run: true` before live gameplay

---

**Ready to unleash your AI-powered Clash Royale bot!** 🚀🤖