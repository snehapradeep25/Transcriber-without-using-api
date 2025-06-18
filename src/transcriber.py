from faster_whisper import WhisperModel
import numpy as np
from config.settings import MODEL_SIZE, DEVICE
import time
from collections import deque
import re
import gc

class Transcriber:
    def __init__(self):
        print(f"Loading Whisper model '{MODEL_SIZE}'...")
        try:
            self.model = WhisperModel(
                MODEL_SIZE, 
                device=DEVICE,
                compute_type="float16" if DEVICE == "cuda" else "int8",
                cpu_threads=2,  # Reduced for better performance
                num_workers=1
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = WhisperModel(MODEL_SIZE, device=DEVICE)
        
        # Simplified duplicate tracking
        self.last_results = deque(maxlen=2)  # Only keep last 2 results
        self.last_transcription_time = 0
        self.min_gap_between_transcriptions = 0.3  # Reduced from 0.8
        
        # Performance tracking
        self.total_transcriptions = 0
        self.successful_transcriptions = 0
        
        # Memory management - less aggressive
        self.cleanup_counter = 0
        self.cleanup_interval = 10  # Cleanup every 10 transcriptions instead of 5
        
    def transcribe_chunk(self, audio_data):
        """Transcribe audio chunk with improved accuracy"""
        try:
            self.total_transcriptions += 1
            current_time = time.time()
            
            # Less aggressive energy check
            audio_energy = np.sqrt(np.mean(audio_data**2))
            if audio_energy < 0.00001:  # More lenient threshold
                print(f"DEBUG: Audio energy too low: {audio_energy:.6f}")
                return None
                
            print(f"DEBUG: Audio energy: {audio_energy:.6f}")
            
            # Minimal rate limiting - only prevent rapid-fire duplicates
            if current_time - self.last_transcription_time < 0.2:  # Very short gap
                print("DEBUG: Too soon after last transcription")
                return None
            
            # Gentle audio normalization
            audio_float32 = self._normalize_audio_gentle(audio_data)
                
            print(f"DEBUG: Processing {len(audio_float32)} samples")
            
            # Optimized transcription settings for accuracy
            segments, info = self.model.transcribe(
                audio_float32,
                language="en",
                beam_size=2,  # Slightly higher for better accuracy
                best_of=2,    # Better accuracy
                temperature=0.0,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=300,   # Shorter for responsiveness
                    speech_pad_ms=150,             # Adequate padding
                    min_speech_duration_ms=150,    # Shorter minimum
                    max_speech_duration_s=20,      # Longer max
                    threshold=0.3                  # Lower threshold for sensitivity
                ),
                condition_on_previous_text=False,
                no_speech_threshold=0.3,           # Lower threshold
                suppress_blank=True,
                without_timestamps=True,
                word_timestamps=False,
                hallucination_silence_threshold=1.5,  # Lower threshold
                suppress_tokens=[-1],
                repetition_penalty=1.1,            # Slight penalty for repetition
                no_repeat_ngram_size=2,            # Prevent 2-gram repetition
                patience=1.0
            )
            
            # Process segments with better filtering
            text_parts = []
            for segment in segments:
                text = segment.text.strip()
                if text and len(text) > 1:  # More lenient length check
                    print(f"DEBUG: Raw segment: '{text}' (confidence: {segment.avg_logprob:.2f})")
                    # More lenient confidence threshold
                    if segment.avg_logprob > -2.0 and self._is_valid_speech(text):
                        text_parts.append(text)
            
            if not text_parts:
                print("DEBUG: No valid segments found")
                return None
                
            full_text = " ".join(text_parts).strip()
            
            # Gentle text cleaning
            final_text = self._clean_text_gentle(full_text)
            
            if final_text and len(final_text) > 1:  # More lenient
                print(f"DEBUG: Cleaned text: '{final_text}'")
                
                # Simplified duplicate detection
                if self._is_simple_duplicate(final_text):
                    print(f"DEBUG: Skipping duplicate: '{final_text}'")
                    return None
                
                # Update tracking
                self.last_results.append(final_text.lower())
                self.last_transcription_time = current_time
                self.successful_transcriptions += 1
                
                # Periodic cleanup
                self.cleanup_counter += 1
                if self.cleanup_counter >= self.cleanup_interval:
                    self._light_cleanup()
                    self.cleanup_counter = 0
                
                print(f"DEBUG: Final result: '{final_text}' (Success rate: {self.successful_transcriptions}/{self.total_transcriptions})")
                return final_text
            else:
                print("DEBUG: Text was empty or too short after cleaning")
            
            return None
                
        except Exception as e:
            print(f"Transcription error: {e}")
            return None
    
    def _normalize_audio_gentle(self, audio_data):
        """Gentle audio normalization to preserve quality"""
        # Convert to float32 if needed
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Gentle DC offset removal
        mean_val = np.mean(audio_data)
        if abs(mean_val) > 0.01:  # Only remove significant DC offset
            audio_data = audio_data - mean_val
        
        # Gentle normalization - don't over-amplify
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0:
            audio_data = audio_data / max_val
        elif max_val > 0.1 and max_val < 0.5:  # Only boost if reasonably quiet
            audio_data = audio_data * 1.5  # Gentle boost
            
        return np.clip(audio_data, -1.0, 1.0)
    
    def _is_simple_duplicate(self, text):
        """Simple duplicate detection - only catch obvious duplicates"""
        if not text or len(text) < 2:
            return True
            
        text_lower = text.lower().strip()
        
        # Only check against the last result
        if len(self.last_results) > 0:
            last_text = self.last_results[-1]
            if text_lower == last_text:
                return True
        
        return False
    
    def _is_valid_speech(self, text):
        """Check if text represents valid speech"""
        if not text or len(text) <= 1:
            return False
            
        text_clean = text.strip().lower()
        
        # Filter out single characters and obvious non-speech
        if len(text_clean) <= 2 and text_clean in ["", " ", ".", ",", "a", "i", "o"]:
            return False
            
        # Check for excessive repetition of single character
        if len(text_clean) > 2 and len(set(text_clean.replace(' ', ''))) == 1:
            return False
        
        # Allow more content through - be less restrictive
        return True
    
    def _clean_text_gentle(self, text):
        """Gentle text cleaning to preserve content"""
        if not text:
            return ""
            
        # Basic whitespace cleanup
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Only remove extreme repetitions
        text = re.sub(r'\b(\w+)\s+\1\s+\1\s+\1\s+\1\b', r'\1', text)  # 5+ repetitions -> 1
        text = re.sub(r'([.!?]){5,}', r'\1', text)  # 5+ punctuation -> 1
        
        return text.strip()
    
    def _light_cleanup(self):
        """Light memory cleanup"""
        print("DEBUG: Performing light cleanup...")
        
        # Just force garbage collection occasionally
        gc.collect()
        
        print(f"DEBUG: Light cleanup complete. Recent results: {len(self.last_results)}")
    
    def test_transcribe(self, audio_data):
        """Test transcription with same settings as main transcription"""
        try:
            print("DEBUG: Test transcription starting...")
            
            # Use same normalization
            audio_float32 = self._normalize_audio_gentle(audio_data)
            
            # Same settings as main transcription
            segments, info = self.model.transcribe(
                audio_float32,
                language="en",
                beam_size=2,
                temperature=0.0,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=300,
                    speech_pad_ms=150,
                    min_speech_duration_ms=150,
                    threshold=0.3
                ),
                condition_on_previous_text=False,
                no_speech_threshold=0.3,
                without_timestamps=True
            )
            
            print(f"TEST: Detected language: {info.language} (confidence: {info.language_probability:.2f})")
            
            all_text = []
            for i, segment in enumerate(segments):
                print(f"TEST: Segment {i}: '{segment.text}' (confidence: {segment.avg_logprob:.2f})")
                if segment.text.strip():
                    all_text.append(segment.text.strip())
            
            result = " ".join(all_text)
            print(f"TEST: Combined result: '{result}'")
            
            return result if result else "No speech detected"
            
        except Exception as e:
            print(f"TEST Error: {e}")
            return f"Error: {str(e)}"
    
    def reset_context(self):
        """Reset transcription context"""
        self.last_results.clear()
        self.last_transcription_time = 0
        self.total_transcriptions = 0
        self.successful_transcriptions = 0
        self.cleanup_counter = 0
        
        # Light cleanup
        gc.collect()
        
        print("Context reset - ready for fresh transcription")
    
    def get_stats(self):
        """Get transcription statistics"""
        success_rate = (self.successful_transcriptions / max(self.total_transcriptions, 1)) * 100
        return {
            'total_transcriptions': self.total_transcriptions,
            'successful_transcriptions': self.successful_transcriptions,
            'success_rate': success_rate,
            'recent_results_count': len(self.last_results),
            'cleanup_counter': self.cleanup_counter
        }
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'model'):
                del self.model
            
            self.last_results.clear()
            gc.collect()
            
        except Exception as e:
            print(f"Cleanup error: {e}")
        
        print("Cleanup completed")