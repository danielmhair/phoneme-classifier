"""
Audio capture system for Epic 2 temporal brain CLI tool.

Provides real-time audio capture and preprocessing compatible
with Epic 1 model requirements.
"""
import threading
import time
from queue import Queue, Empty
from typing import Optional, Callable, Any
import numpy as np
import sounddevice as sd
import librosa


class AudioCaptureConfig:
    """Configuration for audio capture system."""
    
    def __init__(self,
                 sample_rate: int = 16000,
                 channels: int = 1,
                 dtype: str = 'float32',
                 blocksize: int = 1024,
                 device: Optional[int] = None):
        """Initialize audio capture configuration.
        
        Args:
            sample_rate: Audio sample rate in Hz (16kHz matches Epic 1)
            channels: Number of audio channels (mono=1)
            dtype: Audio data type
            blocksize: Number of frames per audio block
            device: Audio input device ID (None for default)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.blocksize = blocksize
        self.device = device


class AudioProcessor:
    """Processes raw audio for model inference."""
    
    def __init__(self, target_sample_rate: int = 16000):
        """Initialize audio processor.
        
        Args:
            target_sample_rate: Target sample rate for model input
        """
        self.target_sample_rate = target_sample_rate
    
    def process_audio_chunk(self, audio_data: np.ndarray, 
                          original_sample_rate: int) -> np.ndarray:
        """Process audio chunk for model inference.
        
        Args:
            audio_data: Raw audio data
            original_sample_rate: Original sample rate of audio data
            
        Returns:
            Processed audio features ready for model
        """
        # Convert to mono if needed
        if len(audio_data.shape) > 1:
            audio_data = librosa.to_mono(audio_data.T)
        
        # Resample if needed
        if original_sample_rate != self.target_sample_rate:
            audio_data = librosa.resample(
                audio_data, 
                orig_sr=original_sample_rate, 
                target_sr=self.target_sample_rate
            )
        
        # For Epic 1 compatibility, we might need to extract features
        # For now, return raw audio (models will handle feature extraction)
        return audio_data.astype(np.float32)


class AudioCapture:
    """Real-time audio capture system with callback support."""
    
    def __init__(self, config: Optional[AudioCaptureConfig] = None):
        """Initialize audio capture system.
        
        Args:
            config: Audio capture configuration
        """
        self.config = config or AudioCaptureConfig()
        self.processor = AudioProcessor(self.config.sample_rate)
        
        # State management
        self.is_recording = False
        self.stream: Optional[sd.InputStream] = None
        self.audio_queue = Queue(maxsize=100)  # Buffer audio chunks
        
        # Callback for processed audio
        self.audio_callback: Optional[Callable[[np.ndarray], Any]] = None
        self.processing_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.total_frames_captured = 0
        self.start_time: Optional[float] = None
    
    def set_audio_callback(self, callback: Callable[[np.ndarray], Any]):
        """Set callback function for processed audio chunks.
        
        Args:
            callback: Function to call with processed audio data
        """
        self.audio_callback = callback
    
    def start_capture(self):
        """Start audio capture."""
        if self.is_recording:
            print("⚠️  Already recording")
            return
        
        try:
            # Create audio stream
            self.stream = sd.InputStream(
                device=self.config.device,
                channels=self.config.channels,
                samplerate=self.config.sample_rate,
                dtype=self.config.dtype,
                blocksize=self.config.blocksize,
                callback=self._audio_stream_callback
            )
            
            # Start stream
            self.stream.start()
            self.is_recording = True
            self.start_time = time.time()
            self.total_frames_captured = 0
            
            # Start processing thread
            self.processing_thread = threading.Thread(
                target=self._audio_processing_loop,
                daemon=True
            )
            self.processing_thread.start()
            
            print(f"🎤 Audio capture started")
            print(f"   📊 Sample rate: {self.config.sample_rate} Hz")
            print(f"   🔊 Channels: {self.config.channels}")
            print(f"   📦 Block size: {self.config.blocksize}")
            
        except Exception as e:
            print(f"❌ Failed to start audio capture: {e}")
            self.is_recording = False
    
    def stop_capture(self):
        """Stop audio capture."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        
        duration = time.time() - self.start_time if self.start_time else 0
        print(f"🛑 Audio capture stopped")
        print(f"   ⏱️  Duration: {duration:.1f}s")
        print(f"   📊 Frames captured: {self.total_frames_captured}")
    
    def _audio_stream_callback(self, indata: np.ndarray, frames: int, 
                             time_info: Any, status: sd.CallbackFlags):
        """Callback for audio stream data.
        
        Args:
            indata: Input audio data
            frames: Number of frames
            time_info: Timing information
            status: Stream status flags
        """
        if status:
            print(f"⚠️  Audio stream status: {status}")
        
        try:
            # Add to queue for processing
            self.audio_queue.put_nowait(indata.copy())
            self.total_frames_captured += frames
        except:
            # Queue full, skip this chunk
            pass
    
    def _audio_processing_loop(self):
        """Main audio processing loop running in separate thread."""
        while self.is_recording:
            try:
                # Get audio chunk from queue
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Process audio
                processed_audio = self.processor.process_audio_chunk(
                    audio_chunk, 
                    self.config.sample_rate
                )
                
                # Call user callback if set
                if self.audio_callback:
                    self.audio_callback(processed_audio)
                    
            except Empty:
                continue  # No audio available, keep waiting
            except Exception as e:
                print(f"⚠️  Audio processing error: {e}")
    
    def get_capture_stats(self) -> dict:
        """Get capture statistics.
        
        Returns:
            Dictionary with capture statistics
        """
        duration = time.time() - self.start_time if self.start_time else 0
        return {
            'is_recording': self.is_recording,
            'duration': duration,
            'frames_captured': self.total_frames_captured,
            'sample_rate': self.config.sample_rate,
            'queue_size': self.audio_queue.qsize()
        }
    
    def list_audio_devices(self) -> str:
        """List available audio input devices.
        
        Returns:
            Formatted string with device information
        """
        devices = sd.query_devices()
        device_list = []
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:  # Input device
                device_list.append(
                    f"  {i}: {device['name']} "
                    f"({device['max_input_channels']} ch, "
                    f"{device['default_samplerate']:.0f} Hz)"
                )
        
        if device_list:
            return "🎤 Available input devices:\n" + "\n".join(device_list)
        else:
            return "❌ No input devices found"
    
    def __enter__(self):
        """Context manager entry."""
        self.start_capture()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_capture()
    
    def __repr__(self) -> str:
        """String representation of audio capture."""
        status = "recording" if self.is_recording else "stopped"
        return (f"AudioCapture({status}, "
                f"sr={self.config.sample_rate}, "
                f"frames={self.total_frames_captured})")


def get_default_audio_capture() -> AudioCapture:
    """Get default audio capture configured for Epic 1 compatibility.
    
    Returns:
        Configured AudioCapture instance
    """
    config = AudioCaptureConfig(
        sample_rate=16000,  # Epic 1 model requirement
        channels=1,         # Mono audio
        blocksize=1024      # ~64ms at 16kHz
    )
    return AudioCapture(config)