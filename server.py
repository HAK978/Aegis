"""
Flask + Socket.IO Server
Real-time communication between phone and AMD backend
UPDATED: Now with TTS and Haptic feedback support
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

# Import our backend (use the no-vllm version)
try:
    from amd_backend_no_vllm import AMDOptimizedBackend
except:
    from amd_backend import AMDOptimizedBackend

# ============================================================================
# FLASK APP SETUP
# ============================================================================

app = Flask(__name__, static_folder='.')
app.config['SECRET_KEY'] = 'amd-mi300x-knights'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global backend instance
backend = None
is_processing = False


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
        <title>AI Visual Guidance System</title>
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
                background: #00ff00;
                color: #000;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                font-size: 24px;
                font-weight: bold;
                margin: 30px 0;
            }
        </style>
    </head>
    <body>
        <h1>🦮 AI Visual Guidance System</h1>
        <p>Powered by AMD MI300X + Google ADK Multi-Agent System</p>
        
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
                <span class="label">VRAM Usage:</span>
                <span class="value" id="vram-usage">-</span>
            </div>
            <div class="stat">
                <span class="label">Models Loaded:</span>
                <span class="value" id="models-loaded">-</span>
            </div>
        </div>
        
        <button onclick="initBackend()" id="init-btn">Initialize Backend</button>
        <button onclick="testBackend()" id="test-btn" disabled>Run Test</button>
        <button onclick="openCamera()" id="camera-btn" disabled>Open Camera (Desktop)</button>
        
        <div id="test-results" style="margin-top: 20px;"></div>
        
        <script>
            // Show camera URL
            const hostname = window.location.hostname;
            const port = window.location.port;
            const cameraUrl = `http://${hostname}:${port}/camera`;
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
                            document.getElementById('vram-usage').textContent = data.vram_usage;
                            document.getElementById('models-loaded').textContent = data.models_loaded;
                            document.getElementById('test-btn').disabled = false;
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
            
            function testBackend() {
                document.getElementById('test-results').innerHTML = '<p>Running test...</p>';
                
                fetch('/api/test', {method: 'POST'})
                    .then(r => r.json())
                    .then(data => {
                        const html = `
                            <div class="status connected">
                                <h3>Test Results</h3>
                                <div class="stat">
                                    <span class="label">Detections:</span>
                                    <span class="value">${data.count} objects</span>
                                </div>
                                <div class="stat">
                                    <span class="label">Guidance:</span>
                                    <span class="value">${data.guidance}</span>
                                </div>
                                <div class="stat">
                                    <span class="label">Detection Time:</span>
                                    <span class="value">${data.timing.detection_ms.toFixed(1)}ms</span>
                                </div>
                                <div class="stat">
                                    <span class="label">Guidance Time:</span>
                                    <span class="value">${data.timing.guidance_ms.toFixed(1)}ms</span>
                                </div>
                                <div class="stat">
                                    <span class="label">Total Latency:</span>
                                    <span class="value">${data.timing.total_ms.toFixed(1)}ms</span>
                                </div>
                            </div>
                        `;
                        document.getElementById('test-results').innerHTML = html;
                    });
            }
            
            function openCamera() {
                window.location.href = '/camera';
            }
        </script>
    </body>
    </html>
    """


@app.route('/camera')
def camera():
    """Serve enhanced mobile camera interface"""
    return send_from_directory('.', 'camera_mobile.html')


@app.route('/api/init', methods=['POST'])
def init_backend():
    """Initialize the backend"""
    global backend
    
    try:
        if backend is None:
            print("🚀 Initializing AMD backend...")
            backend = AMDOptimizedBackend()
        
        # Get GPU info
        import torch
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "N/A"
        vram_allocated = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.device_count() > 0 else 0
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.device_count() > 0 else 0
        
        return jsonify({
            'success': True,
            'gpu_name': gpu_name,
            'vram_usage': f'{vram_allocated:.1f} / {vram_total:.1f} GB',
            'models_loaded': 'YOLOv8x + Rule-based Guidance'
        })
    
    except Exception as e:
        print(f"❌ Init error: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/test', methods=['POST'])
def test_backend():
    """Test the backend with dummy image"""
    global backend
    
    if backend is None:
        return jsonify({'error': 'Backend not initialized'}), 400
    
    try:
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Process
        result = run_async(backend.process_frame(dummy_image))
        
        return jsonify(result)
    
    except Exception as e:
        print(f"❌ Test error: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# SOCKET.IO EVENTS (Real-time streaming)
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Client connected"""
    print(f"📱 Client connected: {request.sid}")
    emit('status', {'message': 'Connected to AMD MI300X backend'})


@socketio.on('disconnect')
def handle_disconnect():
    """Client disconnected"""
    print(f"📱 Client disconnected: {request.sid}")


@socketio.on('start_stream')
def handle_start_stream():
    """Start video streaming"""
    global is_processing
    is_processing = True
    print(f"📹 Stream started for {request.sid}")


@socketio.on('frame')
def handle_frame(data):
    """Process incoming frame"""
    global backend, is_processing
    
    if not is_processing or backend is None:
        return
    
    try:
        # Decode image
        image = base64_to_image(data['image'])
        if image is None:
            return
        
        # Process frame
        result = run_async(backend.process_frame(image))
        
        # Send result back
        emit('result', result)
    
    except Exception as e:
        print(f"❌ Frame processing error: {e}")
        emit('error', {'message': str(e)})


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("🌟 AI VISUAL GUIDANCE SYSTEM")
    print("   AMD MI300X Backend + Google ADK Multi-Agent System")
    print("   NOW WITH TTS + HAPTIC FEEDBACK!")
    print("=" * 70)
    print()
    print("📍 Access the system:")
    print("   → Dashboard: http://0.0.0.0:5000")
    print("   → Mobile Camera: http://0.0.0.0:5000/camera")
    print()
    print("⚡ Starting server...")
    print("=" * 70)
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)