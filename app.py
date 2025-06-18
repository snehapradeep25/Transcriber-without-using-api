from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import numpy as np
import base64
import io
import wave
from src.transcriber import Transcriber
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global transcriber instance
transcriber = None

# Simplified rate limiting
last_audio_time = 0
audio_rate_limit = 0.1  # Much more responsive - 100ms between processing
processing_lock = threading.Lock()  # Prevent concurrent processing

def init_transcriber():
    """Initialize the transcriber in a separate thread to avoid blocking"""
    global transcriber
    try:
        print("🔄 Initializing transcriber...")
        transcriber = Transcriber()
        socketio.emit('transcriber_ready', {'status': 'ready'})
        print("✅ Transcriber initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing transcriber: {e}")
        import traceback
        traceback.print_exc()
        socketio.emit('transcriber_error', {'error': str(e)})

@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('🔗 Client connected')
    if transcriber is None:
        threading.Thread(target=init_transcriber, daemon=True).start()
        emit('transcriber_loading', {'status': 'loading'})
    else:
        emit('transcriber_ready', {'status': 'ready'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('❌ Client disconnected')

def is_valid_audio(audio_array):
    """Simplified audio validation"""
    if len(audio_array) < 800:  # Less than 0.05 seconds at 16kHz
        return False
    
    # Simple energy check
    audio_energy = np.sqrt(np.mean(audio_array**2))
    if audio_energy < 0.00001:
        print(f"🔇 Audio energy too low: {audio_energy:.6f}")
        return False
    
    # Check for obvious silence (all values near zero)
    if np.max(np.abs(audio_array)) < 0.001:
        print("🔇 Audio appears to be silence")
        return False
    
    return True

@socketio.on('audio_data')
def handle_audio_data(data):
    """Handle incoming audio data with improved processing"""
    global last_audio_time
    
    if transcriber is None:
        print("⚠️ Transcriber not ready")
        emit('transcription_error', {'error': 'Transcriber not ready'})
        return
    
    # Use lock to prevent concurrent processing
    if not processing_lock.acquire(blocking=False):
        print("🔒 Processing already in progress, skipping")
        return
    
    try:
        current_time = time.time()
        
        # Simple rate limiting
        if current_time - last_audio_time < audio_rate_limit:
            print(f"⏱️ Rate limited (last processed {current_time - last_audio_time:.2f}s ago)")
            return
        
        print("🎵 Received audio data")
        
        # Decode audio data
        audio_bytes = base64.b64decode(data['audio'])
        print(f"📊 Decoded {len(audio_bytes)} bytes")
        
        # Convert to numpy array
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
        print(f"🔢 Audio array shape: {audio_array.shape}")
        print(f"📈 Audio range: [{np.min(audio_array):.4f}, {np.max(audio_array):.4f}]")
        
        # Simple validation
        if not is_valid_audio(audio_array):
            print("🚫 Audio validation failed")
            return
        
        # Update timing
        last_audio_time = current_time
        
        # Transcribe
        print("🎤 Starting transcription...")
        start_time = time.time()
        text = transcriber.transcribe_chunk(audio_array)
        transcription_time = time.time() - start_time
        print(f"⏱️ Transcription took {transcription_time:.2f} seconds")
        
        if text and text.strip():
            print(f"📝 Transcription result: '{text}'")
            
            # Send result
            result_data = {
                'text': text.strip(),
                'timestamp': current_time,
                'mode': data.get('mode', 'normal'),
                'processing_time': transcription_time
            }
            emit('transcription_result', result_data)
            print(f"✅ Sent transcription: '{text.strip()}'")
        else:
            print("⚠️ No valid transcription result")
        
    except Exception as e:
        print(f"💥 Error processing audio: {e}")
        import traceback
        traceback.print_exc()
        emit('transcription_error', {'error': str(e)})
    
    finally:
        processing_lock.release()

@socketio.on('test_transcription')
def handle_test_transcription(data):
    """Handle test transcription request"""
    if transcriber is None:
        print("⚠️ Transcriber not ready for test")
        emit('test_result', {'error': 'Transcriber not ready'})
        return
    
    try:
        print("🧪 Starting test transcription...")
        
        # Decode audio data
        audio_bytes = base64.b64decode(data['audio'])
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
        
        print(f"🧪 Test audio: {len(audio_array)} samples, energy: {np.sqrt(np.mean(audio_array**2)):.6f}")
        
        # Test transcription
        start_time = time.time()
        text = transcriber.test_transcribe(audio_array)
        test_time = time.time() - start_time
        
        print(f"🧪 Test result: '{text}' (took {test_time:.2f}s)")
        
        emit('test_result', {
            'text': text if text else 'No speech detected',
            'timestamp': time.time(),
            'processing_time': test_time
        })
        
    except Exception as e:
        print(f"💥 Error in test transcription: {e}")
        import traceback
        traceback.print_exc()
        emit('test_result', {'error': str(e)})

@socketio.on('reset_transcriber')
def handle_reset_transcriber():
    """Reset the transcriber context"""
    global last_audio_time
    
    if transcriber:
        transcriber.reset_context()
        
    # Reset rate limiting
    last_audio_time = 0
    
    emit('transcriber_reset', {'status': 'reset'})
    print("🔄 Transcriber context reset")

@socketio.on('get_stats')
def handle_get_stats():
    """Get transcriber statistics"""
    if transcriber:
        stats = transcriber.get_stats()
        emit('transcriber_stats', stats)
        print(f"📊 Stats: {stats}")
    else:
        emit('transcriber_stats', {'error': 'Transcriber not ready'})

if __name__ == '__main__':
    print("🚀 Starting Real-Time English Transcriber Web App...")
    print("💡 Open your browser and navigate to http://localhost:5000")
    print("🔍 Debug mode enabled - check console for detailed logs")
    print("⚙️ Features: Improved accuracy, Reduced latency, Better memory management")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)