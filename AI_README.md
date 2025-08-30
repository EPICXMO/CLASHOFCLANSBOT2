# AI-Powered Clash Royale Bot

## 🚀 The Intelligent Upgrade

This repository now includes a revolutionary AI-powered bot that plays Clash Royale with **true intelligence** rather than scripted rules. The AI bot uses GPT-4 Vision to see, understand, and make strategic decisions just like a human player would.

## 🧠 What Makes This Different

### ❌ OLD SYSTEM (Traditional Bot)
- **Blind coordinate clicking** - `click(540, 1650)` without understanding
- **Scripted behavior** - "if elixir >= 4, deploy random card"
- **No adaptability** - same strategy regardless of opponent
- **Template matching** - breaks when UI changes
- **No strategic understanding** - just follows rules

### ✅ NEW SYSTEM (AI-Powered Bot)
- **True vision** - Uses GPT-4V to actually SEE and UNDERSTAND the game
- **Strategic thinking** - Analyzes opponent patterns and adapts strategy
- **Conversational intelligence** - Narrates decisions like a human streamer
- **Adaptive behavior** - Learns and responds to different situations
- **Error recovery** - Handles unexpected scenarios intelligently

## 🎯 AI Bot Features

### 1. **ChatGPT-Like Vision**
```python
# The AI actually SEES and UNDERSTANDS like this:
"I can see the enemy just deployed a Giant at the back. They're investing 5 elixir, 
so I should pressure the opposite lane while they can't defend well."
```

### 2. **Strategic Narration**
```
🎤 AI SAYS: "They keep rushing with Hog Rider! I'll save my Cannon for the next push."
🧠 AI UNDERSTANDS: Enemy strategy detected - Hog cycle deck
🎯 AI Decision: Deploy Skeleton Army to defend, then counter-push opposite lane
```

### 3. **Intelligent Actions**
- **Context-aware card deployment** - understands WHY to play cards
- **Strategic positioning** - places troops based on battlefield analysis
- **Adaptive timing** - learns optimal timing for different situations
- **Opponent analysis** - recognizes and counters enemy patterns

## 🛠️ Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure OpenAI API
Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys) and add it to `config.yaml`:

```yaml
ai:
  openai_api_key: "sk-your-actual-api-key-here"
  model: "gpt-4o"  # GPT-4 Vision model
  max_tokens: 1000
  temperature: 0.3
  analysis_interval: 2.0  # Seconds between AI analysis
  enable_narration: true  # Enable streamer-like commentary
```

### 3. Connect Your Device
```bash
# Connect to Android device/emulator
adb connect 127.0.0.1:5555
```

### 4. Run the AI Bot
```bash
# Run the intelligent AI-powered bot
python intelligent_main.py --mode intelligent

# Or run the old traditional bot for comparison
python intelligent_main.py --mode traditional
```

## 🎮 Usage Examples

### Intelligent Mode (AI-Powered)
```bash
python intelligent_main.py --mode intelligent
```
**Output:**
```
🚀 Starting INTELLIGENT play loop (AI-powered)
🧠 Bot will see, understand, and adapt like ChatGPT
🔍 AI is analyzing the screen...
🧠 AI UNDERSTANDS: I can see we're at the main menu. Trophy count shows 4,532. I should start a battle.
🎤 AI SAYS: "Let's get into a match and show some strategic gameplay!"
🤖 AI Decision: click_battle - I need to start a battle to begin playing
🎯 AI-guided battle click at (0.50, 0.88)
```

### Traditional Mode (Old System)
```bash
python intelligent_main.py --mode traditional
```
**Output:**
```
🕹️ Starting TRADITIONAL rule-based bot
🚨 This is the OLD system that uses scripts and coordinate clicking
Clicking at (540, 1650)...
Deploying card at position 1...
Following script step 5...
```

## 🧪 Testing

### Run AI System Tests
```bash
python -m pytest test_ai_system.py -v
```

### Run All Tests
```bash
python -m pytest tests/ -v
```

## 📊 Performance Comparison

| Feature | Traditional Bot | AI-Powered Bot |
|---------|----------------|-----------------|
| **Understanding** | None - blind clicking | Full game state comprehension |
| **Strategy** | Fixed rules | Adaptive strategic thinking |
| **Narration** | None | Human-like commentary |
| **Adaptability** | None | Learns opponent patterns |
| **Error Handling** | Basic | Intelligent recovery |
| **Decision Making** | Scripted | Context-aware reasoning |

## 🎯 Example AI Behavior

### Battle Scenario
```
🔍 AI is analyzing the screen...
🧠 AI UNDERSTANDS: Enemy just deployed Golem (8 elixir) at back-left. I have 7 elixir. 
My hand has Inferno Tower, Skeleton Army, Fireball, Zap. This is a heavy push building up.

🎤 AI SAYS: "Big Golem push incoming! I'll pressure opposite lane while they're low on elixir, 
then defend with Inferno Tower when it crosses the bridge."

🤖 AI Decision: deploy_card - Counter-pressure right lane with Hog Rider
🚀 Deployed card 2 at (0.75, 0.65)
```

### Menu Navigation
```
🔍 AI is analyzing the screen...
🧠 AI UNDERSTANDS: At main menu, can see trophy count 5,234. No chests available. 
Should start battle for progression.

🎤 AI SAYS: "Time for another battle! Let's climb those trophies!"

🤖 AI Decision: click_battle - Start battle for trophy progression
🎯 AI-guided battle click at (0.50, 0.88)
```

## 🔧 Configuration Options

### AI Behavior Settings
```yaml
ai:
  # Core AI settings
  openai_api_key: "your-key-here"
  model: "gpt-4o"  # or "gpt-4o-mini" for cost savings
  temperature: 0.3  # Lower = more consistent, Higher = more creative
  
  # Performance settings
  analysis_interval: 2.0  # How often AI analyzes (seconds)
  max_tokens: 1000  # Response length limit
  vision_detail: "low"  # "low" or "high" - affects API cost
  
  # Behavior settings
  enable_narration: true  # Streamer-like commentary
```

### Cost Optimization
- Use `gpt-4o-mini` for lower costs
- Set `vision_detail: "low"` to reduce image processing costs
- Increase `analysis_interval` to reduce API calls
- Set `max_tokens` lower to reduce response costs

## 🚨 Important Notes

### API Costs
- GPT-4 Vision API calls cost money
- Typical session: $0.10-0.50 depending on settings
- Use cost controls in your OpenAI account

### Fallback Behavior
- If no API key is configured, automatically falls back to traditional mode
- If API fails, uses intelligent fallback strategies
- Maintains robustness even with AI service issues

### Game Terms of Service
- This is for educational purposes only
- Do not use to violate game Terms of Service
- Understand your responsibilities when using automation tools

## 🤝 Contributing

This AI system is highly extensible:

1. **Improve AI Prompts** - Enhance the vision analysis prompts
2. **Add Memory Systems** - Implement long-term game memory
3. **Strategy Modules** - Add specific deck strategies
4. **Performance Metrics** - Track AI decision effectiveness

## 📈 Future Enhancements

- **Multi-game AI** - Extend to other mobile games
- **Local AI Models** - Support for local LLaVA/similar models
- **Advanced Memory** - Long-term opponent pattern recognition
- **Strategy Templates** - Deck-specific AI strategies
- **Performance Analytics** - AI decision effectiveness tracking

---

**The future of game automation is here - intelligent, adaptive, and truly understanding what it sees!**