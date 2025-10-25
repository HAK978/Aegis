# Quick Start Guide

## Installation Complete ✅

All dependencies have been installed successfully!

## Next Steps

### 1. Configure API Key

```bash
# Copy the environment template
cp .env.example .env

# Edit and add your Google API key
nano .env  # or use your preferred editor
```

Add your API key:
```
GOOGLE_API_KEY=your_google_api_key_here
```

Get your API key from: https://aistudio.google.com/apikey

### 2. Start the Server

```bash
python server.py
```

### 3. Access the System

- **Dashboard:** http://localhost:5000
- **Mobile Camera:** http://localhost:5000/camera

### 4. Initialize the AI Agent

1. Open the camera interface
2. Click Settings (⚙️)
3. Scroll to "🤖 AI Agent (Gemini)" section
4. Click "🚀 Initialize Agent"
5. Enter your API key or press Cancel to use .env

### 5. Test the Agent

Try these modes:
- **Hybrid (default):** Auto-navigation + ask questions anytime
- **Q&A:** Ask specific questions
- **Monitor:** Get automatic alerts
- **Navigation:** Continuous guidance
- **Contextual:** Smart scene descriptions

## Usage

### Ask Questions
Type in the "Ask Agent" field:
- "What's in front of me?"
- "Is it safe to continue?"
- "Describe what you see"

### Switch Modes
Use the "Agent Mode" dropdown to try different patterns:
1. Q&A - Interactive questions
2. Monitor - Proactive alerts
3. Navigation - Turn-by-turn
4. Contextual - Scene awareness
5. Hybrid - Best of both worlds

## Optional: Audio Processing

For full audio support (Voice Activity Detection), install these:

```bash
# Install system dependencies first
sudo dnf install python3-devel portaudio-devel  # Fedora
# or
sudo apt install python3-dev portaudio19-dev    # Ubuntu

# Then install Python packages
pip install webrtcvad pyaudio
```

## Documentation

- **Full Integration Guide:** [AGENT_INTEGRATION.md](AGENT_INTEGRATION.md)
- **Test Suite:** Run `python test_agent_loops.py`

## Troubleshooting

### GPU Not Detected
Your PyTorch is using CUDA (v2.9.0+cu128). For AMD ROCm support:
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.0
```

### Agent Won't Initialize
- Check that `.env` contains `GOOGLE_API_KEY`
- Verify API key at https://aistudio.google.com/apikey
- Check browser console for errors

## Ready to Go! 🚀

```bash
python server.py
```

Then visit http://localhost:5000/camera on your phone!
