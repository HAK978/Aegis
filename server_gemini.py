"""
Flask + Socket.IO Server - GEMINI NAVIGATION VERSION
Full AI-powered navigation using Gemini Agent
Compare with server.py (rule-based version)
"""

from flask import Flask, send_from_directory, request, jsonify
from flask_socketio import SocketIO, emit
import asyncio
import base64
import numpy as np
import cv2
import time
from threading import Thread
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import our backend (use the no-vllm version)
try:
    from amd_backend_no_vllm import AMDOptimizedBackend
except:
    from amd_backend import AMDOptimizedBackend

# Import Gemini Agent
try:
    from gemini_agent import GeminiAgent, AgentMode, VisualContext, UserQuery, AgentResponse
    AGENT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Gemini Agent not available: {e}")
    AGENT_AVAILABLE = False

from audio_processor import AudioProcessor

# ============================================================================
# FLASK APP SETUP
# ============================================================================

app = Flask(__name__, static_folder='.')
app.config['SECRET_KEY'] = 'amd-mi300x-knights-gemini'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global backend instance
backend = None
is_processing = False

# Global agent instance
agent = None
agent_mode = AgentMode.HYBRID  # Use hybrid mode for navigation + Q&A
audio_processor = None
frame_counter = 0

# Agent loop management
agent_loop_task = None
agent_loop_running = False

# Wake word state - store latest frame for Gemini queries
latest_frame = None
wake_word_active = False


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def base64_to_image(base64_string: str) -> np.ndarray:
    """Convert base64 string to numpy image"""
    try:
        # Remove data URL prefix if present
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]
        
        # Decode base64
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"❌ Error decoding image: {e}")
        return None


def run_async(coro):
    """Run async function in sync context"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ============================================================================
# REST API ENDPOINTS
# ============================================================================

@app.route('/')
def index():
    """Serve main dashboard page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Visual Guidance System - GEMINI VERSION</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: #1a1a1a;
                color: #fff;
            }
            .status {
                padding: 20px;
                border-radius: 10px;
                background: #2a2a2a;
                margin: 20px 0;
            }
            .status.connected { border-left: 5px solid #00ff00; }
            .status.disconnected { border-left: 5px solid #ff0000; }
            h1 { color: #00aaff; }
            .stat { 
                display: flex; 
                justify-content: space-between;
                padding: 10px 0;
                border-bottom: 1px solid #444;
            }
            .stat:last-child { border-bottom: none; }
            .label { color: #888; }
            .value { color: #0f0; font-weight: bold; }
            button {
                background: #00aaff;
                color: white;
                border: none;
                padding: 15px 30px;
                font-size: 16px;
                border-radius: 5px;
                cursor: pointer;
                margin: 10px 5px;
            }
            button:hover { background: #0088cc; }
            button:disabled { background: #555; cursor: not-allowed; }
            .highlight {
                background: #ff6600;
                color: #fff;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                font-size: 24px;
                font-weight: bold;
                margin: 30px 0;
            }
            .version-badge {
                background: #00ff00;
                color: #000;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                display: inline-block;
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <h1>🤖 AI Visual Guidance System</h1>
        <div class="version-badge">GEMINI NAVIGATION VERSION</div>
        <p>Powered by AMD MI300X + Google Gemini 2.0 Flash</p>
        
        <div class="highlight">
            📱 On your phone, visit:<br>
            <code id="camera-url">Loading...</code>
        </div>
        
        <div id="status" class="status disconnected">
            <div class="stat">
                <span class="label">Backend Status:</span>
                <span class="value" id="backend-status">Not Started</span>
            </div>
            <div class="stat">
                <span class="label">GPU:</span>
                <span class="value" id="gpu-name">-</span>
            </div>
            <div class="stat">
                <span class="label">Navigation Mode:</span>
                <span class="value" style="color: #ff6600;">Gemini AI (Hybrid)</span>
            </div>
            <div class="stat">
                <span class="label">Agent Status:</span>
                <span class="value" id="agent-status">Not Initialized</span>
            </div>
        </div>
        
        <button onclick="initBackend()" id="init-btn">Initialize Backend</button>
        <button onclick="openCamera()" id="camera-btn" disabled>Open Camera (Mobile)</button>
        
        <div style="margin-top: 30px; padding: 20px; background: #2a2a2a; border-radius: 10px;">
            <h3>🆚 Version Comparison</h3>
            <p><strong>Rule-Based (server.py):</strong></p>
            <ul>
                <li>✅ Fast (<5ms latency)</li>
                <li>✅ Free (no API costs)</li>
                <li>✅ Deterministic warnings</li>
                <li>❌ Simple, repetitive language</li>
            </ul>
            
            <p><strong>Gemini AI (server_gemini.py):</strong></p>
            <ul>
                <li>✅ Natural language navigation</li>
                <li>✅ Intelligent Q&A</li>
                <li>✅ Context-aware guidance</li>
                <li>⚠️ Slower (~500ms latency)</li>
                <li>⚠️ API costs ($0.05-0.15/hour)</li>
            </ul>
        </div>
        
        <script>
            // Show camera URL
            const hostname = window.location.hostname;
            const port = window.location.port;
            const cameraUrl = `http://${hostname}:${port}/camera_gemini`;
            document.getElementById('camera-url').textContent = cameraUrl;
            
            let backendReady = false;
            
            function initBackend() {
                document.getElementById('backend-status').textContent = 'Initializing...';
                document.getElementById('init-btn').disabled = true;
                
                fetch('/api/init', {method: 'POST'})
                    .then(r => r.json())
                    .then(data => {
                        if (data.success) {
                            document.getElementById('status').className = 'status connected';
                            document.getElementById('backend-status').textContent = 'Ready';
                            document.getElementById('gpu-name').textContent = data.gpu_name;
                            document.getElementById('agent-status').textContent = data.agent_status || 'Ready';
                            document.getElementById('camera-btn').disabled = false;
                            backendReady = true;
                        } else {
                            alert('Failed to initialize: ' + data.error);
                            document.getElementById('init-btn').disabled = false;
                        }
                    })
                    .catch(err => {
                        alert('Error: ' + err);
                        document.getElementById('init-btn').disabled = false;
                    });
            }
            
            function openCamera() {
                window.location.href = '/camera_gemini';
            }
        </script>
    </body>
    </html>
    """


@app.route('/camera_gemini')
def camera():
    """Serve Gemini-powered mobile camera interface"""
    return send_from_directory('.', 'camera_gemini.html')


@app.route('/api/init', methods=['POST'])
def init_backend():
    """Initialize the backend AND Gemini agent"""
    global backend, agent, audio_processor
    
    try:
        # Initialize YOLO backend
        if backend is None:
            print("🚀 Initializing AMD backend...")
            backend = AMDOptimizedBackend()
        
        # Initialize Gemini Agent
        if not AGENT_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'Gemini Agent not available. Install: pip install google-generativeai>=0.8.3'
            })
        
        # Use the original Gemini API key (most stable)
        api_key = os.getenv('GOOGLE_API_KEY') or "AIzaSyDHR_t5gHbL_heDE9z3eYQBoua7dtXeWYo"
        
        if agent is None:
            print("🤖 Initializing Gemini 2.0 Flash Experimental Agent (Hybrid Mode)...")
            agent = GeminiAgent(
                model="gemini-2.0-flash-exp",  # Original stable model
                mode=AgentMode.HYBRID,
                api_key=api_key
            )
            agent.start_conversation()
            print("✅ Gemini 2.0 Flash Agent ready for navigation + Q&A")
        
        # Initialize audio processor
        if audio_processor is None:
            audio_processor = AudioProcessor(enable_vad=True)
        
        # Get GPU info
        import torch
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "N/A"
        
        return jsonify({
            'success': True,
            'gpu_name': gpu_name,
            'agent_status': 'Gemini 2.5 Flash Active (Hybrid Mode)',
            'mode': 'gemini_navigation'
        })
    
    except Exception as e:
        print(f"❌ Init error: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/agent/init', methods=['POST'])
def init_agent():
    """Alternative endpoint for agent initialization (compatibility)"""
    return init_backend()


# ============================================================================
# SOCKET.IO EVENTS (Real-time streaming with Gemini)
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Client connected"""
    print(f"📱 Client connected: {request.sid}")
    emit('status', {'message': 'Connected to Gemini-powered backend'})


@socketio.on('disconnect')
def handle_disconnect():
    """Client disconnected"""
    print(f"📱 Client disconnected: {request.sid}")


@socketio.on('start_stream')
def handle_start_stream():
    """Start video streaming with Gemini navigation"""
    global is_processing, agent_loop_running
    
    is_processing = True
    print(f"📹 Stream started for {request.sid}")
    
    # Start Gemini navigation loop if not already running
    if agent and not agent_loop_running:
        print("🚀 Starting Gemini Hybrid Loop (Navigation + Q&A)")
        # The loop will be driven by visual context updates


@socketio.on('frame')
def handle_frame(data):
    """Process incoming frame - store for Gemini queries, use rule-based guidance"""
    global backend, is_processing, agent, frame_counter, latest_frame

    if not is_processing or backend is None:
        return

    try:
        # Decode image
        image = base64_to_image(data['image'])
        if image is None:
            return

        # Store latest frame for wake word queries
        latest_frame = image

        # Process frame with YOLO (rule-based, fast, free)
        result = run_async(backend.process_frame(image))

        # Update agent visual context if agent is active (for context only, no API calls)
        if agent is not None:
            frame_counter += 1
            visual_context = VisualContext(
                detections=result.get('detections', []),
                count=result.get('count', 0),
                guidance=result.get('guidance', ''),
                timestamp=result.get('timestamp', time.time()),
                frame_id=frame_counter
            )
            agent.update_visual_context(visual_context)
            
            # AEGIS MODE: No automatic API calls
            # Gemini only responds when wake word "Hello Aegis" is triggered
            # Then user asks question, Gemini answers based on latest_frame
            result['gemini_enhanced'] = False

        # Send result back (rule-based guidance only)
        emit('result', result)

    except Exception as e:
        print(f"❌ Frame processing error: {e}")
        emit('error', {'message': str(e)})


@socketio.on('wake_word_activated')
def handle_wake_word_activated():
    """Handle wake word 'Hello Aegis' activation"""
    global wake_word_active, latest_frame, agent
    
    print("👂 Wake word 'Hello Aegis' detected!")
    
    if agent is None:
        emit('agent_error', {'message': 'Gemini Agent not initialized'})
        return
    
    if latest_frame is None:
        emit('agent_error', {'message': 'No camera frame available'})
        return
    
    # Activate wake word mode - next query will use this frame
    wake_word_active = True
    
    # Send confirmation (plays "Yes?" sound)
    emit('wake_word_confirmed', {'message': 'Yes? I\'m listening.'})
    print("✅ Aegis activated - ready for question")


@socketio.on('agent_query_text')
def handle_agent_query_text(data):
    """Handle text query to Gemini agent (from wake word)"""
    global agent, latest_frame, wake_word_active

    if agent is None:
        emit('agent_error', {'message': 'Gemini Agent not initialized'})
        return

    try:
        query_text = data.get('query', '')
        if not query_text:
            emit('agent_error', {'message': 'No query provided'})
            return

        print(f"💬 User query: {query_text}")
        
        # If wake word was activated and we have a frame, send it to Gemini
        if wake_word_active and latest_frame is not None:
            print(f"📸 Sending screenshot to Gemini with query")
            
            # Convert frame to base64 for Gemini
            import cv2
            _, buffer = cv2.imencode('.jpg', latest_frame)
            import base64
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Create a multimodal query with the image
            # Note: Gemini agent needs to support image input
            # For now, we'll just include visual context from detections
            response = run_async(agent.query(query_text, include_visual_context=True))
            
            # Reset wake word state
            wake_word_active = False
            
        else:
            # Regular text query without image
            response = run_async(agent.query(query_text, include_visual_context=True))

        print(f"🤖 Gemini response: {response.text}")

        # Send response
        emit('agent_response', {
            'text': response.text,
            'mode': response.mode.value,
            'timestamp': response.timestamp
        })

    except Exception as e:
        print(f"❌ Agent query error: {e}")
        emit('agent_error', {'message': str(e)})
        wake_word_active = False  # Reset on error


@socketio.on('agent_query_audio')
def handle_agent_query_audio(data):
    """Handle audio query to agent (for wake word functionality)"""
    global agent, audio_processor

    if agent is None:
        emit('agent_error', {'message': 'Gemini Agent not initialized'})
        return

    if audio_processor is None:
        emit('agent_error', {'message': 'Audio processor not initialized'})
        return

    try:
        # Get audio data (base64 encoded)
        audio_base64 = data.get('audio', '')
        if not audio_base64:
            emit('agent_error', {'message': 'No audio provided'})
            return

        # Decode audio
        audio_bytes = audio_processor.base64_to_pcm(audio_base64)

        # Convert to Gemini format
        gemini_audio = audio_processor.browser_to_gemini(
            audio_bytes,
            source_sample_rate=data.get('sample_rate', 48000)
        )

        # Check for voice activity
        has_speech = audio_processor.is_speech(gemini_audio)

        if not has_speech:
            emit('agent_info', {'message': 'No speech detected'})
            return

        # TODO: Implement speech-to-text or direct audio query to Gemini
        # For now, send a placeholder response
        emit('agent_response', {
            'text': 'Audio query received but speech-to-text not yet implemented. Please use text queries.',
            'mode': agent.mode.value,
            'timestamp': time.time()
        })

    except Exception as e:
        print(f"❌ Agent audio query error: {e}")
        emit('agent_error', {'message': str(e)})


@socketio.on('agent_mode_change')
def handle_agent_mode_change(data):
    """Handle agent mode change request"""
    global agent, agent_mode

    if agent is None:
        emit('agent_error', {'message': 'Agent not initialized'})
        return

    try:
        new_mode_str = data.get('mode', 'agent_hybrid')
        new_mode = AgentMode(new_mode_str)

        # Update mode
        agent.mode = new_mode
        agent_mode = new_mode

        # Restart conversation with new mode
        agent.start_conversation()

        print(f"✅ Agent mode changed to: {new_mode.value}")

        # Confirm mode change
        emit('agent_mode_changed', {
            'mode': new_mode.value,
            'timestamp': time.time()
        })

    except Exception as e:
        print(f"❌ Agent mode change error: {e}")
        emit('agent_error', {'message': str(e)})


@socketio.on('agent_get_stats')
def handle_agent_get_stats():
    """Get agent statistics"""
    global agent

    if agent is None:
        emit('agent_error', {'message': 'Agent not initialized'})
        return

    try:
        stats = agent.get_stats()
        emit('agent_stats', stats)

    except Exception as e:
        print(f"❌ Agent stats error: {e}")
        emit('agent_error', {'message': str(e)})


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("🤖 AI VISUAL GUIDANCE SYSTEM - GEMINI NAVIGATION VERSION")
    print("   AMD MI300X Backend + Google Gemini 2.0 Flash AI")
    print("=" * 70)
    print()
    print("📍 Access the system:")
    print("   → Dashboard: http://0.0.0.0:5000")
    print("   → Mobile Camera: http://0.0.0.0:5000/camera_gemini")
    print()
    print("⚡ Starting Gemini-powered server on PORT 5000...")
    print("   (Using existing ngrok tunnel)")
    print("=" * 70)
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
