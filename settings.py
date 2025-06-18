# Real-Time English Transcriber Configuration
# ============================================

# Audio settings
SAMPLE_RATE = 16000  # Whisper works best with 16kHz
CHUNK_DURATION = 1.0  # Process audio in 1-second chunks
CHANNELS = 1  # Mono audio

# Whisper model settings
MODEL_SIZE = "small"  # Options: tiny, base, small, medium, large
DEVICE = "cpu"  # Use "cuda" if you have NVIDIA GPU with CUDA support

# Real-time processing settings
MIN_SILENCE_DURATION = 0.8  # Longer silence before processing (seconds)
ENERGY_THRESHOLD = 0.001  # Higher threshold to ignore background noise
MIN_SPEECH_DURATION = 1.0  # Minimum speech duration to process (seconds)

# Performance optimization
MAX_AUDIO_BUFFER_SIZE = 16000 * 3  # 3 seconds of audio buffer
CLEANUP_INTERVAL = 10  # Cleanup every N transcriptions
RATE_LIMIT_SECONDS = 0.1  # Minimum time between audio processing

# Web server settings
HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 5000  # Default port
DEBUG = True  # Set to False in production

# Transcription quality settings
BEAM_SIZE = 2  # Higher = better quality, slower processing
TEMPERATURE = 0.0  # Lower = more consistent, higher = more creative
VAD_THRESHOLD = 0.3  # Voice Activity Detection sensitivity