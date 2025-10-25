# Gemini AI Agent Integration Guide

## Overview

The Aegis visual guidance system now includes an intelligent conversational AI agent powered by **Google Gemini 2.5 Flash**. This agent provides multimodal understanding by combining real-time camera vision, object detection, and natural language interaction.

## Features

### 5 Continuous Loop Patterns

The agent supports 5 different operational modes, each designed for specific use cases:

#### 1. **Q&A Loop** (`agent_qa`)
**Use Case:** User asks questions about their surroundings
**How it works:**
- User presses button and asks question (voice or text)
- Agent receives current camera frame + detection context
- Gemini processes with conversation history
- Response sent back to user (TTS + text)
- User can ask follow-up questions

**Example interactions:**
- "What objects are in front of me?"
- "How far is the person?"
- "Is it safe to walk forward?"
- "Describe what you see"

#### 2. **Monitoring Loop** (`agent_monitor`)
**Use Case:** Proactive environmental awareness
**How it works:**
- System continuously analyzes frames every 2 seconds
- Agent compares with previous scene state
- Significant changes trigger proactive alerts
- User can interrupt with voice command
- Loop continues monitoring

**Example alerts:**
- "New person approaching from left"
- "Car entering your path ahead"
- "Path is now clear to proceed"

#### 3. **Navigation Loop** (`agent_nav`)
**Use Case:** Turn-by-turn navigation assistance
**How it works:**
- User: "Guide me to the door"
- Agent activates navigation mode
- Continuous loop analyzing current frame
- Generates turn-by-turn instructions
- Updates as user moves
- Confirms when goal reached

**Example guidance:**
- "Walk straight for 3 meters"
- "Turn slightly left to avoid person"
- "Door is 2 meters ahead on your right"

#### 4. **Contextual Loop** (`agent_contextual`)
**Use Case:** Detailed spatial understanding
**How it works:**
- User asks complex question about environment
- Agent gathers recent frames (multi-frame context)
- Builds spatial understanding with 1M token context window
- Detailed response with spatial relationships
- Stores in conversation history

**Example queries:**
- "Describe the room layout"
- "What's to my left and right?"
- "Where is the nearest chair?"

#### 5. **Hybrid Loop** (`agent_hybrid`) ⭐ **Recommended**
**Use Case:** Balanced automatic + on-demand assistance
**How it works:**
- Background: Monitor detections from AMD GPU
- Detect significant changes → Agent decides priority
- Critical situations: Immediate TTS alert
- Info updates: Queued for next interaction
- User can trigger agent anytime with button
- Seamless mode switching

**Example flow:**
- [Auto] "Person approaching, 2 meters"
- [User] "How many people are around me?"
- [Agent] "I see 3 people: one ahead at 2m, one left at 4m, one right at 6m"
- [Auto] "Path clear on left side"

---

## Installation

### 1. Install Dependencies

```bash
# Ensure you're in the rocmserve conda environment
conda activate rocmserve

# Install Gemini and audio dependencies
pip install google-generativeai>=0.8.3
pip install websockets>=13.0
pip install python-dotenv==1.0.1
pip install pyaudio>=0.2.14
pip install soundfile>=0.12.1
pip install webrtcvad>=2.0.10  # Optional: Voice Activity Detection

# Or install all at once
pip install -r requirements.txt
```

### 2. Get Google API Key

1. Visit https://aistudio.google.com/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the API key

### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your API key
nano .env
```

**Required `.env` configuration:**

```bash
# Your Google API key
GOOGLE_API_KEY=your_actual_api_key_here

# Recommended model (fast and cost-efficient)
GEMINI_MODEL=gemini-2.5-flash

# Default agent mode (hybrid recommended)
AGENT_DEFAULT_MODE=agent_hybrid

# Rate limiting (queries per minute)
AGENT_MAX_QUERIES_PER_MINUTE=10

# Enable Voice Activity Detection (reduces API calls)
AGENT_ENABLE_VAD=true
```

---

## Usage

### Starting the Server with Agent Support

```bash
# Start the server (agent auto-initializes if API key is set)
python server.py
```

**Expected output:**
```
🌟 AI VISUAL GUIDANCE SYSTEM
   AMD MI300X Backend + Google Gemini Multi-Agent System
   NOW WITH TTS + HAPTIC FEEDBACK + AI AGENT!
===================================================================

✅ Audio processor initialized
✅ Gemini Agent initialized
   Model: gemini-2.5-flash
   Mode: agent_hybrid

📍 Access the system:
   → Dashboard: http://0.0.0.0:5000
   → Mobile Camera: http://0.0.0.0:5000/camera

⚡ Starting server...
```

### Using the Agent via API

#### Initialize Agent (if not auto-initialized)

```bash
curl -X POST http://localhost:5000/api/agent/init \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "your_api_key",
    "mode": "agent_hybrid"
  }'
```

**Response:**
```json
{
  "success": true,
  "mode": "agent_hybrid",
  "model": "gemini-2.5-flash"
}
```

#### Send Text Query

```bash
curl -X POST http://localhost:5000/api/agent/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What objects are in front of me?"
  }'
```

**Response:**
```json
{
  "success": true,
  "response": "I see 2 objects: a person at 2.5 meters ahead, and a car on the right at 5 meters.",
  "mode": "agent_hybrid",
  "timestamp": 1735123456.78
}
```

#### Change Agent Mode

```bash
curl -X POST http://localhost:5000/api/agent/mode \
  -H "Content-Type": application/json" \
  -d '{
    "mode": "agent_qa"
  }'
```

#### Get Agent Statistics

```bash
curl http://localhost:5000/api/agent/stats
```

**Response:**
```json
{
  "mode": "agent_hybrid",
  "conversation_turns": 5,
  "visual_context_frames": 142,
  "is_running": true,
  "last_query_time": 1735123456.78
}
```

---

## Mobile Interface Integration

### Socket.IO Events

The mobile camera interface communicates with the agent via Socket.IO:

#### **Client → Server Events:**

**1. agent_query_text**
```javascript
socket.emit('agent_query_text', {
  query: "What's in front of me?"
});
```

**2. agent_query_audio**
```javascript
socket.emit('agent_query_audio', {
  audio_data: base64AudioString,
  sample_rate: 16000
});
```

**3. agent_mode_change**
```javascript
socket.emit('agent_mode_change', {
  mode: "agent_hybrid"
});
```

**4. agent_get_stats**
```javascript
socket.emit('agent_get_stats');
```

#### **Server → Client Events:**

**1. agent_response**
```javascript
socket.on('agent_response', (data) => {
  console.log('Agent:', data.text);
  console.log('Mode:', data.mode);

  // Play audio if available
  if (data.audio_data) {
    playAudio(data.audio_data);
  }
});
```

**2. agent_error**
```javascript
socket.on('agent_error', (data) => {
  console.error('Agent error:', data.message);
});
```

**3. agent_stats**
```javascript
socket.on('agent_stats', (stats) => {
  console.log('Conversation turns:', stats.conversation_turns);
  console.log('Current mode:', stats.mode);
});
```

---

## Example: Adding Agent Button to Mobile UI

```html
<!-- Add to camera_mobile.html -->
<div id="agent-controls">
  <button id="agent-ask-btn" class="agent-button">
    🎤 Ask AI Agent
  </button>

  <select id="agent-mode-select">
    <option value="agent_hybrid">Hybrid Mode</option>
    <option value="agent_qa">Q&A Mode</option>
    <option value="agent_monitor">Monitoring Mode</option>
    <option value="agent_nav">Navigation Mode</option>
    <option value="agent_contextual">Contextual Mode</option>
  </select>
</div>

<script>
// Initialize Socket.IO
const socket = io();

// Agent ask button
document.getElementById('agent-ask-btn').addEventListener('click', async () => {
  // Option 1: Text input
  const question = prompt("What would you like to ask?");
  if (question) {
    socket.emit('agent_query_text', { query: question });
  }

  // Option 2: Voice input (requires Web Speech API or audio recording)
  // See full implementation in examples/voice_agent.html
});

// Listen for responses
socket.on('agent_response', (data) => {
  // Display response
  showNotification(data.text);

  // Speak response using TTS
  speakText(data.text);
});

// Mode change
document.getElementById('agent-mode-select').addEventListener('change', (e) => {
  const newMode = e.target.value;
  socket.emit('agent_mode_change', { mode: newMode });
});
</script>
```

---

## Architecture

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     MOBILE BROWSER                               │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────────────┐   │
│  │ Camera Feed  │→ │  Socket.IO  │ →│ Voice/Text Input     │   │
│  └──────────────┘  └─────────────┘  └──────────────────────┘   │
└────────────────────────────┬─────────────────────────────────────┘
                             │ WebSocket
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                   FLASK SERVER (server.py)                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Socket.IO Event Handlers                     │  │
│  │  • frame → process_frame()                                │  │
│  │  • agent_query_text → handle_agent_query_text()          │  │
│  │  • agent_query_audio → handle_agent_query_audio()        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                             │                                    │
│                             ↓                                    │
│  ┌────────────────┐  ┌──────────────────┐                      │
│  │ AMD Backend    │  │ Gemini Agent     │                      │
│  │ (GPU Vision)   │→ │ (AI Reasoning)   │                      │
│  └────────────────┘  └──────────────────┘                      │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                      AMD MI300X GPU                              │
│  ┌──────────────────┐     ┌────────────────────────────┐        │
│  │ YOLOv8x          │     │ Depth-Anything V2          │        │
│  │ Object Detection │     │ Distance Estimation        │        │
│  │ ~80ms            │     │ ~40ms                      │        │
│  └──────────────────┘     └────────────────────────────┘        │
│              ↓ Visual Context (objects + distances)             │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                 GOOGLE GEMINI 2.5 FLASH                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Multimodal Understanding:                                │   │
│  │ • Visual context (detections)                            │   │
│  │ • User question (text/audio)                             │   │
│  │ • Conversation history (1M tokens)                       │   │
│  │ • Spatial reasoning                                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│              ↓ Natural language response (~300-500ms)           │
└─────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Latency |
|-----------|---------------|---------|
| **AMD Backend** | Real-time object detection + depth estimation | 60-65ms |
| **Gemini Agent** | Natural language understanding + reasoning | 300-500ms |
| **Audio Processor** | Format conversion (browser ↔ Gemini) | ~5ms |
| **Server** | Orchestration + Socket.IO messaging | ~10ms |
| **Total (with agent)** | **Full conversational interaction** | **400-600ms** |

---

## Performance & Cost Optimization

### Latency Optimization

1. **Parallel Processing**: AMD GPU continues detection while Gemini processes queries
2. **Rate Limiting**: Max 10 queries/minute prevents runaway costs
3. **Context Pruning**: Keep last 50 conversation turns (fits in context window)
4. **VAD (Voice Activity Detection)**: Only process audio with speech (saves API calls)

### Cost Management

**Gemini 2.5 Flash Pricing** (as of January 2025):
- Input: $0.075 / 1M tokens
- Output: $0.30 / 1M tokens

**Estimated costs:**
- Average query: ~500 input tokens + ~100 output tokens
- Cost per query: ~$0.00007 ($0.07 per 1000 queries)
- 10 queries/minute × 60 min = 600 queries/hour = **$0.04/hour**
- Daily usage (8 hours): **~$0.30/day**
- Monthly (30 days × 8 hours): **~$9/month**

**Cost reduction strategies:**
1. Use VAD to filter silence (reduces unnecessary API calls by ~60%)
2. Set conversation history limit (we use 50 turns)
3. Enable rate limiting (10 queries/minute default)
4. Use Gemini 2.5 Flash instead of Pro (20x cheaper, minimal quality loss)

---

## Troubleshooting

### Agent Not Initializing

**Error:** `GOOGLE_API_KEY not found`

**Solution:**
```bash
# Check if .env file exists
ls -la .env

# If not, create it
cp .env.example .env

# Edit and add your API key
nano .env

# Verify environment variable is loaded
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('GOOGLE_API_KEY'))"
```

### Rate Limit Exceeded

**Error:** `Rate limit exceeded. Please wait X seconds.`

**Solution:**
```bash
# Increase rate limit in .env
AGENT_MAX_QUERIES_PER_MINUTE=20

# Or wait for the specified time before next query
```

### Slow Response Times

**Issue:** Agent responses taking >1 second

**Solutions:**
1. **Check network:** `ping -c 5 generativelanguage.googleapis.com`
2. **Reduce context:** Lower `AGENT_MAX_HISTORY_TURNS` in .env
3. **Use Flash model:** Ensure `GEMINI_MODEL=gemini-2.5-flash` (not Pro)
4. **Check GPU:** Agent uses AMD GPU for vision, ensure it's not overloaded

### No Audio Output

**Issue:** Agent responds with text but no TTS

**Current Status:** Audio TTS is handled client-side via browser's Web Speech API

**Solution:**
```javascript
// In your mobile client
function speakText(text) {
  if ('speechSynthesis' in window) {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
  }
}
```

---

## Advanced Usage

### Custom System Instructions

```python
from gemini_agent import GeminiAgent, AgentMode

# Create agent with custom instructions
agent = GeminiAgent(mode=AgentMode.QA)

custom_instruction = """You are a cheerful and encouraging guide for a visually impaired user.
Always emphasize safety first, but maintain a warm, friendly tone.
Use spatial language (left, right, ahead, behind) clearly."""

agent.start_conversation(system_instruction=custom_instruction)
```

### Running Agent in Background Loop

```python
import asyncio
import threading
from gemini_agent import GeminiAgent, AgentMode

agent = GeminiAgent(mode=AgentMode.MONITOR)
stop_event = threading.Event()

async def alert_callback(alert):
    print(f"🚨 Alert: {alert.text}")
    # Send to mobile client via Socket.IO
    socketio.emit('agent_alert', {'text': alert.text})

# Start monitoring loop in background
asyncio.run(agent.monitoring_loop(
    send_alert=alert_callback,
    stop_event=stop_event,
    alert_interval=5.0  # Alert every 5 seconds max
))
```

---

## Examples

See the `examples/` directory for full working examples:

1. **`examples/simple_qa.py`** - Basic Q&A with agent
2. **`examples/monitoring_loop.py`** - Continuous monitoring
3. **`examples/navigation_guide.py`** - Turn-by-turn navigation
4. **`examples/hybrid_mode.py`** - Combined navigation + Q&A
5. **`examples/voice_agent.html`** - Full web interface with voice input

---

## API Reference

### GeminiAgent Class

```python
class GeminiAgent:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash-exp",
        mode: AgentMode = AgentMode.HYBRID
    )

    def start_conversation(self, system_instruction: Optional[str] = None)

    async def query(
        self,
        user_input: str,
        include_visual_context: bool = True
    ) -> AgentResponse

    def update_visual_context(self, context: VisualContext)

    def get_stats(self) -> Dict[str, Any]
```

### Data Classes

```python
@dataclass
class VisualContext:
    detections: List[Dict[str, Any]]
    count: int
    guidance: str
    timestamp: float
    frame_id: int = 0

@dataclass
class AgentResponse:
    text: str
    audio_data: Optional[bytes] = None
    timestamp: float = 0.0
    mode: AgentMode = AgentMode.QA
```

---

## Security Considerations

1. **API Key Protection:**
   - Never commit `.env` to version control
   - `.env` is in `.gitignore`
   - Rotate keys periodically

2. **Rate Limiting:**
   - Prevents DoS on your API key
   - Default: 10 queries/minute
   - Adjust based on your usage

3. **Input Validation:**
   - All user inputs are sanitized
   - No code execution in prompts
   - Safe against injection attacks

4. **HTTPS Required:**
   - Use ngrok or proper SSL cert for production
   - Protects API key in transit

---

## Future Enhancements

- [ ] Speech-to-text integration (live audio → text)
- [ ] Text-to-speech via Gemini Audio (instead of browser TTS)
- [ ] Multi-language support
- [ ] Conversation export/import
- [ ] Agent personality customization
- [ ] Integration with Gemini Live API (ultra-low latency)
- [ ] Offline mode with cached responses
- [ ] Voice activity detection improvements

---

## Support

For issues or questions:

1. **Check logs:** `tail -f server.log`
2. **Test agent standalone:**
   ```bash
   python gemini_agent.py
   ```
3. **Verify API key:**
   ```bash
   curl -X POST "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"contents":[{"parts":[{"text":"Hello"}]}]}'
   ```
4. **Report issues:** Include agent mode, query, and error message

---

**Built with ❤️ for accessibility and inclusion**

**Powered by:**
- AMD MI300X + ROCm 7.0
- Google Gemini 2.5 Flash
- YOLOv8x + Depth-Anything V2
