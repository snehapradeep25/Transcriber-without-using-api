# Transcriber-without-using-api
 🎤 Real-Time English Transcriber

A powerful real-time speech-to-text transcription application that runs locally without requiring any external APIs. Built with Faster-Whisper, Flask, and Socket.IO for seamless real-time audio processing.

## ✨ Features

- Real-time transcription using Faster-Whisper (no API required)
- Web-based interface with modern, responsive design
- Highlight mode to mark specific words in transcriptions
- Audio visualization with real-time waveform display
- PDF export functionality for saving transcriptions
- Test mode for quick transcription testing
- Optimized performance with intelligent duplicate detection
- Memory management for long-running sessions
- Cross-platform compatibility (Windows, macOS, Linux)

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- Microphone access
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Installation

1. **Clone or download the project**
   ```bash
   git clone <https://github.com/snehapradeep25/Transcriber-without-using-api.git>
   cd real-time-transcriber
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open your browser**
   - Navigate to `http://localhost:5000`
   - Allow microphone access when prompted
   - Wait for the model to load (first time may take a few minutes)

## 📁 Project Structure

```
real-time-transcriber/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── config/
│   └── settings.py       # Configuration settings
├── src/
│   └── transcriber.py    # Core transcription logic
└── templates/
    └── index.html        # Web interface
```

## 🔧 Configuration

### Audio Settings (config/settings.py)
```python
SAMPLE_RATE = 16000        # Optimal for Whisper
CHUNK_DURATION = 1.0       # Audio processing chunks
CHANNELS = 1               # Mono audio
```

### Model Settings
```python
MODEL_SIZE = "small"       # Options: tiny, base, small, medium, large
DEVICE = "cpu"             # Use "cuda" for GPU acceleration
```

### Performance Tuning
```python
MIN_SILENCE_DURATION = 0.8   # Silence detection
ENERGY_THRESHOLD = 0.001     # Background noise filter
MIN_SPEECH_DURATION = 1.0    # Minimum speech length
```

## 🎯 Usage

### Basic Transcription
1. Click **"Start Recording"** to begin
2. Speak clearly into your microphone
3. View real-time transcriptions in the text area
4. Click **"Stop Recording"** when finished

### Highlight Mode
1. Record some text first
2. Click **"Highlight Mode"** to activate
3. Click **"Start Highlighting"**
4. Speak words you want to highlight in the existing text
5. Words will be highlighted in real-time

### Test Mode
- Click **"Test Transcription"** for a 3-second test recording
- Useful for checking microphone and model functionality

### Export Options
- **Clear All**: Remove all transcriptions
- **Download PDF**: Export transcriptions as PDF file

## ⚙️ Advanced Configuration

### GPU Acceleration
For faster processing with NVIDIA GPUs:
```python
# In config/settings.py
DEVICE = "cuda"
```

### Model Selection
Choose based on your needs:
- `tiny`: Fastest, lowest accuracy
- `base`: Good balance of speed and accuracy
- `small`: Better accuracy (default)
- `medium`: High accuracy, slower
- `large`: Best accuracy, slowest

### Memory Optimization
The application includes automatic memory management:
- Duplicate detection to prevent redundant processing
- Periodic cleanup of old results
- Optimized audio buffer management

## 🔧 Troubleshooting

### Common Issues

**Model loading fails:**
- Ensure stable internet connection for initial download
- Check available disk space (models are 100MB-1GB)
- Try a smaller model size first

**No audio detected:**
- Check microphone permissions in browser
- Verify microphone is working in other applications
- Try adjusting `ENERGY_THRESHOLD` in settings

**Poor transcription quality:**
- Speak clearly and at moderate pace
- Reduce background noise
- Consider upgrading to a larger model
- Check microphone quality

**High CPU usage:**
- Use GPU acceleration if available
- Reduce model size
- Increase `CHUNK_DURATION` for less frequent processing

### Debug Mode

The application runs in debug mode by default. Check the console for detailed logs:
- Audio energy levels
- Processing times
- Transcription confidence scores
- Memory usage statistics

## 🛠️ Development

### Adding New Features

The modular architecture makes it easy to extend:

1. **Transcriber class** (`src/transcriber.py`): Core audio processing
2. **Flask routes** (`app.py`): Backend API endpoints
3. **Frontend** (`templates/index.html`): User interface

### Performance Monitoring

Built-in statistics tracking:
- Total transcriptions processed
- Success rate
- Processing times
- Memory usage

Access via browser console or add UI elements to display stats.

## 📊 Performance Benchmarks

### Typical Performance (on modern hardware):
- **Model loading**: 10-30 seconds (first time only)
- **Processing latency**: 0.5-2 seconds
- **Memory usage**: 200-500MB
- **CPU usage**: 10-30% (varies by model size)

### Optimization Tips:
- Use GPU acceleration for 3-5x speedup
- Smaller models for real-time requirements
- Larger models for accuracy-critical applications

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source. Please ensure compliance with Faster-Whisper and other dependency licenses.

## 🙏 Acknowledgments

- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) for efficient Whisper implementation
- [OpenAI Whisper](https://github.com/openai/whisper) for the base model
- Flask and Socket.IO communities for excellent web framework support

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review console logs for error details
3. Create an issue with detailed information about your problem

---

**Happy Transcribing! 🎉**
