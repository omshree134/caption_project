"""
Desktop Captioning Application
A floating overlay window that captures audio and displays real-time captions
Supports both online (Reverie API) and offline (Whisper) transcription
"""

import sys
import json
import asyncio
import threading
import queue
import numpy as np
from pathlib import Path
import time

# ============================================================================
# CRITICAL: Load Whisper BEFORE PyQt5 to avoid CTranslate2/Qt conflict!
# The CTranslate2 library used by faster-whisper has threading conflicts with Qt
# ============================================================================

WHISPER_AVAILABLE = False
VAD_AVAILABLE = False
_WHISPER_MODEL = None  # Global pre-loaded model

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
    print("[Whisper] faster-whisper is available")
    
    # Pre-load the model BEFORE Qt to avoid crashes
    try:
        print("[Whisper] Pre-loading model before Qt initialization...")
        print("[Whisper] This may take a moment on first run...")
        _WHISPER_MODEL = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("[Whisper] Model pre-loaded successfully!")
    except Exception as e:
        print(f"[Whisper] Warning: Failed to pre-load model: {e}")
        _WHISPER_MODEL = None
        
except ImportError:
    print("[Whisper] faster-whisper not installed - offline mode unavailable")

try:
    import webrtcvad
    VAD_AVAILABLE = True
    print("[VAD] webrtcvad is available")
except ImportError:
    try:
        # Try the wheels version
        import webrtcvad_wheels as webrtcvad
        VAD_AVAILABLE = True
        print("[VAD] webrtcvad-wheels is available")
    except ImportError:
        print("[VAD] webrtcvad not installed - using energy-based VAD")

# Now safe to import PyQt5
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QTextEdit, QFrame, QSlider,
    QMessageBox, QCheckBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QColor, QPalette, QCursor

import websockets
import sounddevice as sd

# Load configuration
def load_config():
    config_path = Path(__file__).parent / 'config.json'
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except:
        return {
            "api_key": "",
            "app_id": "",
            "default_language": "hi",
            "default_domain": "generic"
        }

config = load_config()

# Supported languages
LANGUAGES = {
    "auto": "🔄 Auto Detect",
    "hi": "Hindi",
    "en": "English", 
    "hi-en": "Hindi + English",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
}


class AudioCapture(QThread):
    """Thread for capturing audio using sounddevice or WASAPI loopback"""
    audio_data = pyqtSignal(bytes)
    audio_level = pyqtSignal(float)
    error_signal = pyqtSignal(str)
    
    def __init__(self, mic_device=None, loopback_device=None, capture_mode="mic"):
        super().__init__()
        self.mic_device = mic_device
        self.loopback_device = loopback_device
        self.capture_mode = capture_mode  # "mic", "speaker", or "both"
        self.running = False
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 512
        
    def run(self):
        try:
            self.running = True
            
            if self.capture_mode == "speaker":
                # Use WASAPI loopback for system audio
                self._capture_wasapi_loopback()
            elif self.capture_mode == "both":
                # Capture both mic and system audio
                self._capture_both_wasapi()
            else:
                # Microphone only
                self._capture_single(self.mic_device)
                
        except Exception as e:
            print(f"[Audio] Error: {e}")
            import traceback
            traceback.print_exc()
            self.error_signal.emit(str(e))
    
    def _capture_wasapi_loopback(self):
        """Capture system audio using WASAPI loopback (PyAudioWPatch)"""
        try:
            import pyaudiowpatch as pyaudio
            from scipy import signal
            
            p = pyaudio.PyAudio()
            
            # Find the default WASAPI loopback device
            wasapi_info = None
            default_speakers = None
            
            try:
                # Get default WASAPI output device
                wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
                default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
                
                # Find the loopback device for default speakers
                loopback_device = None
                for i in range(p.get_device_count()):
                    dev = p.get_device_info_by_index(i)
                    if dev.get("isLoopbackDevice", False):
                        # Check if this loopback matches our output device
                        if default_speakers["name"] in dev["name"]:
                            loopback_device = dev
                            break
                
                if loopback_device is None:
                    # Just get any loopback device
                    for i in range(p.get_device_count()):
                        dev = p.get_device_info_by_index(i)
                        if dev.get("isLoopbackDevice", False):
                            loopback_device = dev
                            break
                            
                if loopback_device is None:
                    raise Exception("No WASAPI loopback device found")
                    
                print(f"[Audio] WASAPI Loopback: {loopback_device['name']}")
                print(f"[Audio] Sample rate: {int(loopback_device['defaultSampleRate'])}")
                
                device_rate = int(loopback_device['defaultSampleRate'])
                device_channels = min(loopback_device['maxInputChannels'], 2)
                
                # Open the loopback stream
                stream = p.open(
                    format=pyaudio.paFloat32,
                    channels=device_channels,
                    rate=device_rate,
                    input=True,
                    input_device_index=loopback_device['index'],
                    frames_per_buffer=int(self.chunk_size * device_rate / self.sample_rate)
                )
                
                print("[Audio] WASAPI loopback stream started")
                
                while self.running:
                    data = stream.read(int(self.chunk_size * device_rate / self.sample_rate), exception_on_overflow=False)
                    audio = np.frombuffer(data, dtype=np.float32)
                    
                    # Convert stereo to mono if needed
                    if device_channels == 2:
                        audio = audio.reshape(-1, 2).mean(axis=1)
                    
                    # Resample to 16kHz
                    if device_rate != self.sample_rate:
                        new_len = int(len(audio) * self.sample_rate / device_rate)
                        audio = signal.resample(audio, new_len)
                    
                    level = np.sqrt(np.mean(audio**2))
                    self.audio_level.emit(min(level * 5, 1.0))
                    
                    audio_int16 = (audio * 32767).astype(np.int16)
                    self.audio_data.emit(audio_int16.tobytes())
                
                stream.stop_stream()
                stream.close()
                
            finally:
                p.terminate()
                
        except ImportError:
            print("[Audio] PyAudioWPatch not available, falling back to Stereo Mix")
            self._capture_single(self.loopback_device)
        except Exception as e:
            print(f"[Audio] WASAPI loopback error: {e}")
            # Fallback to stereo mix
            print("[Audio] Falling back to Stereo Mix")
            self._capture_single(self.loopback_device)
    
    def _capture_both_wasapi(self):
        """Capture both microphone and system audio, mix them"""
        from scipy import signal
        import threading
        
        print("[Audio] Starting Mic + System Audio combined capture...")
        
        mic_buffer = []
        loopback_buffer = []
        buffer_lock = threading.Lock()
        mic_active = [False]
        loopback_active = [False]
        
        # Start microphone capture in a separate thread
        def mic_thread():
            try:
                device_info = sd.query_devices(self.mic_device) if self.mic_device is not None else sd.query_devices(kind='input')
                mic_rate = int(device_info['default_samplerate'])
                print(f"[Audio] Mic thread: {device_info['name']} @ {mic_rate}Hz")
                mic_active[0] = True
                
                def callback(indata, frames, time, status):
                    if status:
                        print(f"[Audio] Mic callback status: {status}")
                    if self.running:
                        audio = indata[:, 0].copy()
                        if mic_rate != self.sample_rate:
                            new_len = int(len(audio) * self.sample_rate / mic_rate)
                            audio = signal.resample(audio, new_len)
                        with buffer_lock:
                            mic_buffer.append(audio)
                
                print(f"[Audio] Starting mic stream with device={self.mic_device}")
                with sd.InputStream(device=self.mic_device, samplerate=mic_rate, channels=1,
                                   dtype='float32', blocksize=int(self.chunk_size * mic_rate / self.sample_rate),
                                   callback=callback):
                    print("[Audio] Mic stream started successfully")
                    while self.running:
                        sd.sleep(50)
                print("[Audio] Mic thread ended")
            except Exception as e:
                print(f"[Audio] Mic thread error: {e}")
                import traceback
                traceback.print_exc()
                mic_active[0] = False
        
        # Start loopback capture
        def loopback_thread():
            try:
                import pyaudiowpatch as pyaudio
                p = pyaudio.PyAudio()
                
                wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
                default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
                default_name = default_speakers['name'].lower()
                print(f"[Audio] Loopback thread: Finding loopback for '{default_speakers['name']}'")
                
                # Find the loopback device that matches the default output
                loopback_device = None
                all_loopbacks = []
                for i in range(p.get_device_count()):
                    dev = p.get_device_info_by_index(i)
                    if dev.get("isLoopbackDevice", False):
                        all_loopbacks.append(dev)
                        dev_name_lower = dev['name'].lower()
                        # Check if this loopback matches our default output device
                        # The loopback name usually contains the output device name
                        if default_name.split()[0].lower() in dev_name_lower:
                            loopback_device = dev
                            print(f"[Audio] Found matching loopback: {dev['name']}")
                            break
                
                # If no matching loopback found, use the first available loopback
                if loopback_device is None and all_loopbacks:
                    loopback_device = all_loopbacks[0]
                    print(f"[Audio] Using first available loopback: {loopback_device['name']}")
                
                if loopback_device:
                    device_rate = int(loopback_device['defaultSampleRate'])
                    device_channels = min(loopback_device['maxInputChannels'], 2)
                    print(f"[Audio] Loopback: {loopback_device['name']} @ {device_rate}Hz, {device_channels} channels")
                    loopback_active[0] = True
                    
                    stream = p.open(format=pyaudio.paFloat32, channels=device_channels,
                                   rate=device_rate, input=True,
                                   input_device_index=loopback_device['index'],
                                   frames_per_buffer=int(self.chunk_size * device_rate / self.sample_rate))
                    
                    print("[Audio] Loopback stream started")
                    while self.running:
                        data = stream.read(int(self.chunk_size * device_rate / self.sample_rate), exception_on_overflow=False)
                        audio = np.frombuffer(data, dtype=np.float32)
                        if device_channels == 2:
                            audio = audio.reshape(-1, 2).mean(axis=1)
                        if device_rate != self.sample_rate:
                            new_len = int(len(audio) * self.sample_rate / device_rate)
                            audio = signal.resample(audio, new_len)
                        with buffer_lock:
                            loopback_buffer.append(audio)
                    
                    stream.stop_stream()
                    stream.close()
                    print("[Audio] Loopback stream closed")
                else:
                    print("[Audio] No loopback device found!")
                p.terminate()
            except Exception as e:
                print(f"[Audio] Loopback thread error: {e}")
                import traceback
                traceback.print_exc()
                loopback_active[0] = False
        
        # Start both threads
        mic_t = threading.Thread(target=mic_thread, daemon=True)
        loop_t = threading.Thread(target=loopback_thread, daemon=True)
        mic_t.start()
        loop_t.start()
        
        print("[Audio] Both capture threads started, beginning mix loop...")
        
        # Give threads time to initialize
        import time
        time.sleep(0.2)
        
        mix_count = 0
        # Mix and send audio
        while self.running:
            sd.sleep(30)
            with buffer_lock:
                if mic_buffer or loopback_buffer:
                    mixed = None
                    if mic_buffer and loopback_buffer:
                        mic_data = np.concatenate(mic_buffer)
                        loop_data = np.concatenate(loopback_buffer)
                        min_len = min(len(mic_data), len(loop_data))
                        if min_len > 0:
                            mixed = (mic_data[:min_len] + loop_data[:min_len]) / 2
                            if mix_count % 100 == 0:
                                print(f"[Audio] Mixed: mic={len(mic_data)}, loop={len(loop_data)}, mixed={min_len}")
                    elif mic_buffer:
                        mixed = np.concatenate(mic_buffer)
                        if mix_count % 100 == 0:
                            print(f"[Audio] Mic only: {len(mixed)} samples")
                    elif loopback_buffer:
                        mixed = np.concatenate(loopback_buffer)
                        if mix_count % 100 == 0:
                            print(f"[Audio] Loopback only: {len(mixed)} samples")
                    
                    mic_buffer.clear()
                    loopback_buffer.clear()
                    
                    if mixed is not None and len(mixed) > 0:
                        level = np.sqrt(np.mean(mixed**2))
                        self.audio_level.emit(min(level * 5, 1.0))
                        audio_int16 = (mixed * 32767).astype(np.int16)
                        self.audio_data.emit(audio_int16.tobytes())
                        mix_count += 1
        
        print("[Audio] Mix loop ended")
            
    def _capture_single(self, device):
        """Capture from a single device"""
        device_info = sd.query_devices(device) if device is not None else sd.query_devices(kind='input')
        print(f"[Audio] Using device: {device_info['name']}")
        
        # Get the device's default sample rate
        device_sample_rate = int(device_info['default_samplerate'])
        print(f"[Audio] Device native sample rate: {device_sample_rate}")
        
        # We need 16kHz for the API, so we may need to resample
        need_resample = device_sample_rate != self.sample_rate
        if need_resample:
            print(f"[Audio] Will resample from {device_sample_rate} to {self.sample_rate}")
            from scipy import signal
        
        def callback(indata, frames, time, status):
            if status:
                print(f"[Audio] Status: {status}")
            if self.running:
                audio_data = indata[:, 0].copy()
                
                # Resample if needed
                if need_resample:
                    # Calculate new length
                    new_length = int(len(audio_data) * self.sample_rate / device_sample_rate)
                    audio_data = signal.resample(audio_data, new_length)
                
                level = np.sqrt(np.mean(audio_data**2))
                self.audio_level.emit(min(level * 5, 1.0))
                audio_int16 = (audio_data * 32767).astype(np.int16)
                self.audio_data.emit(audio_int16.tobytes())
        
        try:
            with sd.InputStream(
                device=device,
                samplerate=device_sample_rate,  # Use device's native rate
                channels=self.channels,
                dtype='float32',
                blocksize=int(self.chunk_size * device_sample_rate / self.sample_rate),
                callback=callback
            ):
                while self.running:
                    sd.sleep(50)
        except Exception as e:
            print(f"[Audio] Error opening stream: {e}")
            self.error_signal.emit(str(e))
                
    def _capture_both(self):
        """Capture from both microphone and loopback, mix together"""
        from scipy import signal
        
        try:
            mic_info = sd.query_devices(self.mic_device) if self.mic_device is not None else sd.query_devices(kind='input')
            loopback_info = sd.query_devices(self.loopback_device)
            print(f"[Audio] Mic: {mic_info['name']}")
            print(f"[Audio] Loopback: {loopback_info['name']}")
            
            # Get native sample rates
            mic_rate = int(mic_info['default_samplerate'])
            loopback_rate = int(loopback_info['default_samplerate'])
            print(f"[Audio] Mic rate: {mic_rate}, Loopback rate: {loopback_rate}")
        except Exception as e:
            print(f"[Audio] Device query error: {e}")
            self._capture_single(self.mic_device)
            return
        
        # Shared buffer for mixing (store resampled data)
        mic_buffer = []
        loopback_buffer = []
        buffer_lock = threading.Lock()
        
        def mic_callback(indata, frames, time, status):
            if self.running:
                audio = indata[:, 0].copy()
                # Resample to 16kHz if needed
                if mic_rate != self.sample_rate:
                    new_len = int(len(audio) * self.sample_rate / mic_rate)
                    audio = signal.resample(audio, new_len)
                with buffer_lock:
                    mic_buffer.append(audio)
                    
        def loopback_callback(indata, frames, time, status):
            if self.running:
                audio = indata[:, 0].copy()
                # Resample to 16kHz if needed
                if loopback_rate != self.sample_rate:
                    new_len = int(len(audio) * self.sample_rate / loopback_rate)
                    audio = signal.resample(audio, new_len)
                with buffer_lock:
                    loopback_buffer.append(audio)
        
        mic_stream = None
        loopback_stream = None
        
        try:
            mic_stream = sd.InputStream(
                device=self.mic_device,
                samplerate=mic_rate,
                channels=self.channels,
                dtype='float32',
                blocksize=int(self.chunk_size * mic_rate / self.sample_rate),
                callback=mic_callback
            )
            mic_stream.start()
            print("[Audio] Mic stream started")
        except Exception as e:
            print(f"[Audio] Mic stream error: {e}")
            mic_stream = None
        
        try:
            loopback_stream = sd.InputStream(
                device=self.loopback_device,
                samplerate=loopback_rate,
                channels=self.channels,
                dtype='float32',
                blocksize=int(self.chunk_size * loopback_rate / self.sample_rate),
                callback=loopback_callback
            )
            loopback_stream.start()
            print("[Audio] Loopback stream started")
        except Exception as e:
            print(f"[Audio] Loopback stream error: {e}")
            loopback_stream = None
        
        if mic_stream is None and loopback_stream is None:
            self.error_signal.emit("Could not open any audio device")
            return
            
        try:
            while self.running:
                sd.sleep(30)
                
                with buffer_lock:
                    if mic_buffer or loopback_buffer:
                        mixed = None
                        
                        if mic_buffer and loopback_buffer:
                            mic_data = np.concatenate(mic_buffer)
                            loop_data = np.concatenate(loopback_buffer)
                            min_len = min(len(mic_data), len(loop_data))
                            if min_len > 0:
                                mixed = (mic_data[:min_len] + loop_data[:min_len]) / 2
                        elif mic_buffer:
                            mixed = np.concatenate(mic_buffer)
                        elif loopback_buffer:
                            mixed = np.concatenate(loopback_buffer)
                            
                        mic_buffer.clear()
                        loopback_buffer.clear()
                        
                        if mixed is not None and len(mixed) > 0:
                            level = np.sqrt(np.mean(mixed**2))
                            self.audio_level.emit(min(level * 5, 1.0))
                            # Handle mono or stereo
                            if len(mixed.shape) > 1:
                                audio_int16 = (mixed[:, 0] * 32767).astype(np.int16)
                            else:
                                audio_int16 = (mixed * 32767).astype(np.int16)
                            self.audio_data.emit(audio_int16.tobytes())
        finally:
            # Clean up streams
            if mic_stream:
                try:
                    mic_stream.stop()
                    mic_stream.close()
                except:
                    pass
            if loopback_stream:
                try:
                    loopback_stream.stop()
                    loopback_stream.close()
                except:
                    pass
                
    def stop(self):
        self.running = False


class STTWorker(QThread):
    """Thread for WebSocket communication with Reverie STT API"""
    transcription = pyqtSignal(dict)
    status_changed = pyqtSignal(str, str)
    error_signal = pyqtSignal(str)
    
    def __init__(self, api_key, app_id, language='hi', domain='generic'):
        super().__init__()
        self.api_key = api_key
        self.app_id = app_id
        self.language = language
        self.domain = domain
        self.running = False
        self.audio_queue = queue.Queue(maxsize=100)  # Limit queue size
        self.websocket = None
        self.loop = None
            
    def build_url(self):
        lang = self.language
        if lang == 'auto' or lang == 'hi-en':
            lang = 'hi'
            
        params = {
            "apikey": self.api_key,
            "appid": self.app_id,
            "appname": "stt_stream",
            "src_lang": lang,
            "domain": self.domain,
            "timeout": "180",
            "silence": "1",
            "format": "16k_int16",
            "punctuate": "true",
            "continuous": "1",
            "logging": "false"
        }
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"wss://revapi.reverieinc.com/stream?{query_string}"
    
    def add_audio(self, audio_data):
        """Add audio data to the queue (non-blocking)"""
        if self.running:
            try:
                self.audio_queue.put_nowait(audio_data)
            except queue.Full:
                # Drop old data if queue is full (prevents lag)
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(audio_data)
                except:
                    pass
    
    def run(self):
        """Main thread loop"""
        self.running = True
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._run_async())
        except Exception as e:
            print(f"[STT] Loop error: {e}")
        finally:
            try:
                self.loop.close()
            except:
                pass
        
    async def _run_async(self):
        """Async WebSocket handler"""
        url = self.build_url()
        print(f"[STT] Connecting...")
        self.status_changed.emit("connecting", "Connecting...")
        
        try:
            async with websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=3
            ) as ws:
                self.websocket = ws
                print("[STT] Connected!")
                self.status_changed.emit("connected", "🟢 Connected")
                
                send_task = asyncio.create_task(self._send_audio())
                recv_task = asyncio.create_task(self._receive_transcription())
                
                done, pending = await asyncio.wait(
                    [send_task, recv_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                
        except websockets.exceptions.InvalidStatusCode as e:
            self.error_signal.emit(f"Auth failed (HTTP {e.status_code})")
        except Exception as e:
            print(f"[STT] Error: {e}")
            self.error_signal.emit(str(e))
        finally:
            self.status_changed.emit("disconnected", "Disconnected")
            self.websocket = None
            
    async def _send_audio(self):
        """Send audio data from queue to WebSocket"""
        while self.running and self.websocket:
            try:
                try:
                    audio_data = self.audio_queue.get_nowait()
                    await self.websocket.send(audio_data)
                except queue.Empty:
                    await asyncio.sleep(0.01)
            except Exception as e:
                if self.running:
                    print(f"[STT] Send error: {e}")
                break
                
    async def _receive_transcription(self):
        """Receive transcriptions from WebSocket"""
        try:
            async for message in self.websocket:
                if not self.running:
                    break
                try:
                    data = json.loads(message)
                    self.transcription.emit(data)
                except json.JSONDecodeError:
                    continue
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            if self.running:
                print(f"[STT] Receive error: {e}")
                
    def stop(self):
        """Stop the STT worker"""
        self.running = False
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break


class WhisperOfflineWorker(QThread):
    """
    Offline Speech-to-Text using Faster-Whisper with VAD
    Processes audio in batches for efficiency
    """
    transcription = pyqtSignal(dict)
    status_changed = pyqtSignal(str, str)
    error_signal = pyqtSignal(str)
    model_loaded = pyqtSignal(bool)
    
    # Class-level model cache to load once in main thread
    _cached_model = None
    _cached_model_size = None
    
    @classmethod
    def preload_model(cls, model_size="tiny", device="cpu"):
        """Pre-load model - use the global pre-loaded model"""
        global _WHISPER_MODEL
        
        # First check global model (loaded before Qt)
        if _WHISPER_MODEL is not None:
            print(f"[Whisper] Using globally pre-loaded model", flush=True)
            cls._cached_model = _WHISPER_MODEL
            cls._cached_model_size = model_size
            return _WHISPER_MODEL
        
        # Then check class cache
        if cls._cached_model is not None and cls._cached_model_size == model_size:
            print(f"[Whisper] Using cached model '{model_size}'", flush=True)
            return cls._cached_model
            
        if not WHISPER_AVAILABLE:
            print("[Whisper] faster-whisper not available", flush=True)
            return None
        
        # WARNING: Loading model after Qt can cause crashes!
        print(f"[Whisper] WARNING: Loading model after Qt - this may crash!", flush=True)
        print(f"[Whisper] Pre-loading model '{model_size}' on {device}...", flush=True)
        print(f"[Whisper] This may take a moment on first run...", flush=True)
        
        try:
            model = WhisperModel(
                model_size,
                device=device,
                compute_type="int8" if device == "cpu" else "float16"
            )
            
            cls._cached_model = model
            cls._cached_model_size = model_size
            print(f"[Whisper] Model pre-loaded successfully!", flush=True)
            return model
        except Exception as e:
            print(f"[Whisper] Failed to pre-load model: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None
    
    def __init__(self, model_size="tiny", language="en", device="cpu", model=None):
        super().__init__()
        self.model_size = model_size
        self.language = language if language not in ['auto', 'hi-en'] else None
        self.device = device
        self.running = False
        self.model = model  # Pre-loaded model passed in
        
        # Audio parameters
        self.sample_rate = 16000
        self.audio_queue = queue.Queue(maxsize=200)
        
        # VAD parameters
        self.vad = None
        self.vad_mode = 3  # Aggressiveness: 3 = most aggressive
        
        # Batching parameters
        self.min_speech_duration = 0.2  # minimum speech to transcribe (200ms)
        self.max_batch_duration = 5.0  # force transcription after 5 seconds
        
        # Silence detection - TWO LEVELS
        self.short_silence_frames = 6   # ~192ms = short pause (comma, continue line)
        self.long_silence_frames = 20   # ~640ms = long pause (full stop, new line)
        self.trailing_silence_frames = 3  # ~96ms trailing silence to include
        
        # State
        self.audio_buffer = []
        self.speech_buffer = []
        self.is_speaking = False
        self.silence_frames = 0
        self.speech_frames = 0
        self.last_transcription_time = 0
        self.pending_text = ""  # Accumulate text for same "sentence"
        
    def _init_vad(self):
        """Initialize Voice Activity Detection"""
        if VAD_AVAILABLE:
            try:
                import webrtcvad
                self.vad = webrtcvad.Vad(self.vad_mode)
                print(f"[VAD] WebRTC VAD initialized (mode={self.vad_mode})")
                return True
            except:
                try:
                    import webrtcvad_wheels as webrtcvad
                    self.vad = webrtcvad.Vad(self.vad_mode)
                    print(f"[VAD] WebRTC VAD (wheels) initialized (mode={self.vad_mode})")
                    return True
                except Exception as e:
                    print(f"[VAD] Failed to initialize: {e}")
        return False
    
    def _energy_vad(self, audio_chunk):
        """Simple energy-based VAD fallback"""
        energy = np.sqrt(np.mean(audio_chunk.astype(np.float32)**2))
        threshold = 500  # Adjust based on your mic sensitivity
        return energy > threshold
    
    def _check_vad(self, audio_bytes):
        """Check if audio contains speech using VAD"""
        if self.vad:
            try:
                # WebRTC VAD needs 10, 20, or 30ms frames at 16kHz
                # 16kHz * 0.03 = 480 samples = 960 bytes (16-bit)
                frame_duration = 30  # ms
                frame_size = int(self.sample_rate * frame_duration / 1000) * 2  # bytes
                
                # Check multiple frames and vote
                speech_frames = 0
                total_frames = 0
                
                for i in range(0, len(audio_bytes) - frame_size, frame_size):
                    frame = audio_bytes[i:i + frame_size]
                    if len(frame) == frame_size:
                        try:
                            is_speech = self.vad.is_speech(frame, self.sample_rate)
                            if is_speech:
                                speech_frames += 1
                            total_frames += 1
                        except:
                            pass
                
                # Return True if >30% frames have speech
                if total_frames > 0:
                    return speech_frames / total_frames > 0.3
                return False
            except Exception as e:
                # Fallback to energy VAD
                audio = np.frombuffer(audio_bytes, dtype=np.int16)
                return self._energy_vad(audio)
        else:
            # Energy-based fallback
            audio = np.frombuffer(audio_bytes, dtype=np.int16)
            return self._energy_vad(audio)
    
    def _load_model(self):
        """Load the Whisper model (use pre-loaded if available)"""
        if self.model is not None:
            print(f"[Whisper] Using pre-loaded model", flush=True)
            self.status_changed.emit("ready", f"🟢 Whisper ({self.model_size}) ready")
            self.model_loaded.emit(True)
            return True
            
        # Try to get cached model
        if WhisperOfflineWorker._cached_model is not None:
            self.model = WhisperOfflineWorker._cached_model
            print(f"[Whisper] Using cached model", flush=True)
            self.status_changed.emit("ready", f"🟢 Whisper ({self.model_size}) ready")
            self.model_loaded.emit(True)
            return True
            
        if not WHISPER_AVAILABLE:
            self.error_signal.emit("Whisper not installed. Run: pip install faster-whisper")
            return False
            
        try:
            print(f"[Whisper] Loading model '{self.model_size}' on {self.device}...", flush=True)
            print(f"[Whisper] This may take a moment on first run (downloading model)...", flush=True)
            self.status_changed.emit("loading", f"Loading Whisper {self.model_size}...")
            
            print(f"[Whisper] About to create WhisperModel...", flush=True)
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type="int8" if self.device == "cpu" else "float16"
            )
            print(f"[Whisper] WhisperModel created successfully!", flush=True)
            
            print(f"[Whisper] Model loaded successfully!", flush=True)
            self.status_changed.emit("ready", f"🟢 Whisper ({self.model_size}) ready")
            self.model_loaded.emit(True)
            return True
            
        except Exception as e:
            print(f"[Whisper] Failed to load model: {e}", flush=True)
            import traceback
            traceback.print_exc()
            self.error_signal.emit(f"Failed to load Whisper: {e}")
            self.model_loaded.emit(False)
            return False
    
    def add_audio(self, audio_data):
        """Add audio data to the processing queue"""
        if self.running:
            try:
                self.audio_queue.put_nowait(audio_data)
            except queue.Full:
                # Drop oldest if full
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(audio_data)
                except:
                    pass
    
    def run(self):
        """Main processing loop"""
        self.running = True
        print("[Whisper] Worker thread starting...")
        
        try:
            # Initialize VAD
            self._init_vad()
            
            # Load model
            if not self._load_model():
                print("[Whisper] Model failed to load, exiting worker")
                return
            
            self.status_changed.emit("connected", "🟢 Whisper (Offline)")
            print("[Whisper] Worker ready and listening...")
            
            while self.running:
                try:
                    # Get audio from queue
                    try:
                        audio_data = self.audio_queue.get(timeout=0.05)
                    except queue.Empty:
                        # Check if we should transcribe accumulated audio
                        self._check_and_transcribe()
                        continue
                    
                    # Check for speech using VAD
                    has_speech = self._check_vad(audio_data)
                    
                    if has_speech:
                        self.speech_frames += 1
                        self.silence_frames = 0
                        self.speech_buffer.append(audio_data)
                        
                        if not self.is_speaking:
                            self.is_speaking = True
                    else:
                        self.silence_frames += 1
                        
                        if self.is_speaking:
                            # Include trailing silence
                            if self.silence_frames < self.trailing_silence_frames:
                                self.speech_buffer.append(audio_data)
                            
                            # Short silence - transcribe but CONTINUE same line
                            elif self.silence_frames == self.short_silence_frames:
                                self._transcribe_buffer(is_sentence_end=False)
                            
                            # Long silence - transcribe and START NEW line
                            elif self.silence_frames >= self.long_silence_frames:
                                self.is_speaking = False
                                self._transcribe_buffer(is_sentence_end=True)
                    
                    # Force transcription if buffer is too long (continuous speech)
                    buffer_duration = len(self.speech_buffer) * 0.032
                    if buffer_duration >= self.max_batch_duration:
                        self._transcribe_buffer(is_sentence_end=False)
                        
                except Exception as e:
                    print(f"[Whisper] Processing error: {e}")
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            print(f"[Whisper] Worker thread error: {e}")
            import traceback
            traceback.print_exc()
            self.error_signal.emit(f"Worker error: {e}")
        finally:
            print("[Whisper] Worker thread ending")
                
    def _check_and_transcribe(self):
        """Check if we should transcribe based on time"""
        if not self.speech_buffer:
            return
            
        buffer_duration = len(self.speech_buffer) * 0.032
        time_since_last = time.time() - self.last_transcription_time
        
        # Timeout - flush pending text as new sentence
        if time_since_last > 2.0 and self.pending_text:
            self._flush_pending_text()
    
    def _flush_pending_text(self):
        """Send accumulated pending text as final result"""
        if self.pending_text:
            print(f"[Whisper] Final: {self.pending_text[:80]}")
            self.transcription.emit({
                'success': True,
                'text': self.pending_text,
                'display_text': self.pending_text,
                'final': True,
                'cause': 'whisper_offline',
                'source': 'offline'
            })
            self.pending_text = ""
    
    def _transcribe_buffer(self, is_sentence_end=False):
        """Transcribe the accumulated speech buffer"""
        if not self.speech_buffer or not self.model:
            if is_sentence_end and self.pending_text:
                self._flush_pending_text()
            return
            
        try:
            # Combine audio chunks
            audio_bytes = b''.join(self.speech_buffer)
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Clear buffer immediately
            self.speech_buffer.clear()
            self.speech_frames = 0
            
            # Skip if too short
            if len(audio) < self.sample_rate * self.min_speech_duration:
                if is_sentence_end and self.pending_text:
                    self._flush_pending_text()
                return
            
            # Transcribe
            segments, info = self.model.transcribe(
                audio,
                beam_size=1,
                best_of=1,
                language=self.language,
                task="transcribe",
                vad_filter=False,
                condition_on_previous_text=False,
                word_timestamps=False,
            )
            
            # Collect results
            text_parts = []
            for segment in segments:
                text = segment.text.strip()
                # Remove trailing punctuation - we'll add our own
                text = text.rstrip('.,!?。，')
                if text:
                    text_parts.append(text)
            
            new_text = ' '.join(text_parts).strip()
            
            if new_text:
                # Accumulate text
                if self.pending_text:
                    self.pending_text += ", " + new_text
                else:
                    self.pending_text = new_text
                
                if is_sentence_end:
                    # Long pause - send as complete sentence
                    self.pending_text += "."
                    self._flush_pending_text()
                else:
                    # Short pause - show partial (will continue)
                    print(f"[Whisper] Partial: {self.pending_text[:80]}...")
                    self.transcription.emit({
                        'success': True,
                        'text': self.pending_text,
                        'display_text': self.pending_text + "...",
                        'final': False,
                        'cause': 'partial',
                        'source': 'offline'
                    })
            elif is_sentence_end and self.pending_text:
                # No new text but sentence ended
                self.pending_text += "."
                self._flush_pending_text()
            
            self.last_transcription_time = time.time()
            
        except Exception as e:
            print(f"[Whisper] Error: {e}")
    
    def stop(self):
        """Stop the worker"""
        self.running = False
        # Transcribe any remaining audio
        if self.speech_buffer:
            self._transcribe_buffer(is_sentence_end=True)
        # Flush any pending text
        if self.pending_text:
            self._flush_pending_text()
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break


class CaptionOverlay(QMainWindow):
    """Main overlay window for displaying captions"""
    
    def __init__(self):
        super().__init__()
        self.audio_capture = None
        self.stt_worker = None
        self.whisper_worker = None  # Offline Whisper worker
        self.is_recording = False
        self.drag_position = None
        self.partial_text = ""  # Store partial transcription
        self.last_final_pos = 0  # Track where final text ends in caption box
        self.use_offline_mode = False  # Toggle for offline-only mode
        self.api_failed = False  # Track if API has failed (for auto-fallback)
        
        self.init_ui()
        self.setup_audio_devices()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Live Captions")
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint | 
            Qt.FramelessWindowHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self.setFixedSize(800, 280)
        screen = QApplication.primaryScreen().geometry()
        self.move((screen.width() - 800) // 2, screen.height() - 320)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        
        container = QFrame()
        container.setObjectName("container")
        container.setStyleSheet("""
            #container {
                background-color: rgba(15, 23, 42, 0.95);
                border-radius: 16px;
                border: 1px solid rgba(71, 85, 105, 0.5);
            }
        """)
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(15, 10, 15, 15)
        container_layout.setSpacing(8)
        
        # Header
        header = QHBoxLayout()
        
        title_label = QLabel("🎙️ Live Captions")
        title_label.setStyleSheet("color: #818cf8; font-size: 14px; font-weight: bold;")
        title_label.setCursor(QCursor(Qt.OpenHandCursor))
        header.addWidget(title_label)
        header.addStretch()
        
        # Audio source selector
        self.source_combo = QComboBox()
        self.source_combo.setStyleSheet("""
            QComboBox {
                background-color: #334155; color: white;
                border: 1px solid #475569; border-radius: 6px;
                padding: 5px 10px; min-width: 180px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #1e293b; color: white;
                selection-background-color: #6366f1;
            }
        """)
        header.addWidget(QLabel("🔊"))
        header.addWidget(self.source_combo)
        
        # Language selector
        self.lang_combo = QComboBox()
        self.lang_combo.setStyleSheet("""
            QComboBox {
                background-color: #334155; color: white;
                border: 1px solid #475569; border-radius: 6px;
                padding: 5px 10px; min-width: 100px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #1e293b; color: white;
                selection-background-color: #6366f1;
            }
        """)
        for code, name in LANGUAGES.items():
            self.lang_combo.addItem(name, code)
        header.addWidget(QLabel("🌐"))
        header.addWidget(self.lang_combo)
        
        # Offline mode checkbox
        self.offline_checkbox = QCheckBox("Offline")
        self.offline_checkbox.setStyleSheet("""
            QCheckBox {
                color: #94a3b8;
                font-size: 11px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border-radius: 3px;
                border: 1px solid #475569;
                background-color: #334155;
            }
            QCheckBox::indicator:checked {
                background-color: #10b981;
                border-color: #10b981;
            }
        """)
        self.offline_checkbox.setToolTip("Use offline Whisper model (no internet required)")
        self.offline_checkbox.setEnabled(WHISPER_AVAILABLE)
        if not WHISPER_AVAILABLE:
            self.offline_checkbox.setToolTip("Install faster-whisper for offline mode")
        header.addWidget(self.offline_checkbox)
        
        # Start/Stop button
        self.start_btn = QPushButton("▶ Start")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #6366f1; color: white;
                border: none; border-radius: 6px;
                padding: 8px 20px; font-weight: bold;
            }
            QPushButton:hover { background-color: #818cf8; }
        """)
        self.start_btn.clicked.connect(self.toggle_recording)
        header.addWidget(self.start_btn)
        
        # Close button
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(30, 30)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #ef4444; color: white;
                border: none; border-radius: 6px;
            }
            QPushButton:hover { background-color: #f87171; }
        """)
        close_btn.clicked.connect(self.close)
        header.addWidget(close_btn)
        
        container_layout.addLayout(header)
        
        # Audio level + status row
        level_row = QHBoxLayout()
        
        self.audio_level_bar = QFrame()
        self.audio_level_bar.setFixedHeight(6)
        self.audio_level_bar.setStyleSheet("background-color: #334155; border-radius: 3px;")
        self.audio_level_bar.setFixedWidth(80)
        level_row.addWidget(self.audio_level_bar)
        
        self.audio_level_fill = QFrame(self.audio_level_bar)
        self.audio_level_fill.setGeometry(0, 0, 0, 6)
        self.audio_level_fill.setStyleSheet("background-color: #10b981; border-radius: 3px;")
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        level_row.addWidget(self.status_label)
        level_row.addStretch()
        
        container_layout.addLayout(level_row)
        
        # Caption display area
        self.caption_display = QTextEdit()
        self.caption_display.setReadOnly(True)
        self.caption_display.setStyleSheet("""
            QTextEdit {
                background-color: rgba(30, 41, 59, 0.8);
                color: #f8fafc;
                border: 1px solid #475569;
                border-radius: 8px;
                padding: 10px;
                font-size: 18px;
            }
        """)
        self.caption_display.setPlaceholderText("Captions will appear here...")
        container_layout.addWidget(self.caption_display)
        
        # Bottom controls
        bottom = QHBoxLayout()
        
        bottom.addWidget(QLabel("Opacity:"))
        opacity_slider = QSlider(Qt.Horizontal)
        opacity_slider.setRange(30, 100)
        opacity_slider.setValue(95)
        opacity_slider.setFixedWidth(80)
        opacity_slider.setStyleSheet("""
            QSlider::groove:horizontal { background: #475569; height: 4px; border-radius: 2px; }
            QSlider::handle:horizontal { background: #6366f1; width: 12px; height: 12px; margin: -4px 0; border-radius: 6px; }
        """)
        opacity_slider.valueChanged.connect(lambda v: self.setWindowOpacity(v / 100))
        bottom.addWidget(opacity_slider)
        bottom.addStretch()
        
        clear_btn = QPushButton("Clear")
        clear_btn.setStyleSheet("""
            QPushButton { background-color: #475569; color: white; border: none; border-radius: 4px; padding: 5px 15px; }
            QPushButton:hover { background-color: #64748b; }
        """)
        clear_btn.clicked.connect(lambda: self.caption_display.clear())
        bottom.addWidget(clear_btn)
        
        copy_btn = QPushButton("Copy")
        copy_btn.setStyleSheet("""
            QPushButton { background-color: #475569; color: white; border: none; border-radius: 4px; padding: 5px 15px; }
            QPushButton:hover { background-color: #64748b; }
        """)
        copy_btn.clicked.connect(self.copy_captions)
        bottom.addWidget(copy_btn)
        
        container_layout.addLayout(bottom)
        layout.addWidget(container)
        
        self.setStyleSheet("QLabel { color: #94a3b8; font-size: 12px; }")
        
    def setup_audio_devices(self):
        """Setup audio device options - detect default output for system audio capture"""
        try:
            devices = sd.query_devices()
            self.source_combo.clear()
            
            # Find default devices
            self.mic_device = None
            self.loopback_device = None
            
            # Print all devices for debugging
            print("\n[Audio] Available devices:")
            for i, device in enumerate(devices):
                dev_type = "IN" if device['max_input_channels'] > 0 else ""
                dev_type += "/OUT" if device['max_output_channels'] > 0 else ""
                print(f"  [{i}] {device['name']} ({dev_type})")
            
            # Get default input (microphone) - but we want a REAL microphone, not Stereo Mix
            try:
                default_input = sd.query_devices(kind='input')
                default_name = default_input['name'].lower()
                
                # Check if default is actually a microphone, not a loopback/stereo mix
                loopback_indicators = ['stereo mix', 'what u hear', 'wave out mix', 'loopback', 
                                       'cable output', 'voicemeeter', 'virtual cable']
                is_loopback = any(indicator in default_name for indicator in loopback_indicators)
                
                if is_loopback:
                    print(f"[Audio] Default input is loopback device ({default_input['name']}), searching for real mic...")
                    # Find a real microphone - prioritize "Microphone Array" for laptops
                    # First pass: look for "microphone array" (built-in laptop mic)
                    for i, device in enumerate(devices):
                        if device['max_input_channels'] > 0:
                            name = device['name'].lower()
                            if 'microphone array' in name:
                                self.mic_device = i
                                print(f"[Audio] Found MIC ARRAY: [{i}] {device['name']}")
                                break
                    
                    # Second pass: look for "microphone" if array not found
                    if self.mic_device is None:
                        for i, device in enumerate(devices):
                            if device['max_input_channels'] > 0:
                                name = device['name'].lower()
                                is_loop = any(ind in name for ind in loopback_indicators)
                                # Match "microphone" but not mapper/mix/loopback
                                if 'microphone' in name and 'mapper' not in name and not is_loop:
                                    self.mic_device = i
                                    print(f"[Audio] Found MIC: [{i}] {device['name']}")
                                    break
                    
                    # Third pass: look for headset input (bluetooth headset)
                    if self.mic_device is None:
                        for i, device in enumerate(devices):
                            if device['max_input_channels'] > 0:
                                name = device['name'].lower()
                                if 'headset' in name and 'mapper' not in name:
                                    self.mic_device = i
                                    print(f"[Audio] Found HEADSET MIC: [{i}] {device['name']}")
                                    break
                    
                    if self.mic_device is None:
                        print("[Audio] No real microphone found, using default")
                        self.mic_device = default_input.get('index')
                else:
                    self.mic_device = default_input.get('index')
                    
                if self.mic_device is not None:
                    mic_info = sd.query_devices(self.mic_device)
                    print(f"[Audio] Using MIC: [{self.mic_device}] {mic_info['name']}")
            except Exception as e:
                print(f"[Audio] No default input: {e}")
            
            # Get default output device info
            try:
                default_output = sd.query_devices(kind='output')
                print(f"[Audio] Default OUTPUT: {default_output['name']}")
            except:
                default_output = None
            
            # Strategy to find loopback device:
            # 1. First look for Stereo Mix (captures all system audio)
            # 2. Then look for WASAPI loopback devices
            # 3. Then look for virtual audio cables
            
            loopback_keywords = [
                'stereo mix',      # Windows built-in
                'what u hear',     # Some Realtek drivers
                'wave out mix',    # Some drivers
                'loopback',        # Generic loopback
                'wasapi',          # WASAPI loopback
                'cable output',    # VB-Cable
                'voicemeeter',     # VoiceMeeter
                'virtual cable',   # Virtual cables
                'vb-audio',        # VB-Audio
            ]
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    name = device['name'].lower()
                    for keyword in loopback_keywords:
                        if keyword in name:
                            self.loopback_device = i
                            print(f"[Audio] Found LOOPBACK: [{i}] {device['name']}")
                            break
                    if self.loopback_device is not None:
                        break
            
            # Add options with device info
            mic_name = ""
            if self.mic_device is not None:
                try:
                    mic_name = sd.query_devices(self.mic_device)['name'][:25]
                except:
                    pass
                    
            loopback_name = ""
            if self.loopback_device is not None:
                try:
                    loopback_name = sd.query_devices(self.loopback_device)['name'][:25]
                except:
                    pass
            
            self.source_combo.addItem(f"🎤 Microphone", "mic")
            self.source_combo.addItem(f"🔊 System Audio (Stereo Mix)", "speaker")
            self.source_combo.addItem(f"🎤+🔊 Mic + System Audio", "both")
            
            # Set default based on availability
            if self.loopback_device is not None:
                self.source_combo.setCurrentIndex(2)  # both
                self.status_label.setText(f"✓ Loopback: {loopback_name}")
            else:
                self.source_combo.setCurrentIndex(0)  # mic only
                self.status_label.setText("⚠️ Enable 'Stereo Mix' in Sound settings for system audio")
                print("\n[Audio] ⚠️ No loopback device found!")
                print("[Audio] To enable system audio capture:")
                print("[Audio]   1. Right-click speaker icon → Sound settings")
                print("[Audio]   2. More sound settings → Recording tab")
                print("[Audio]   3. Right-click empty area → Show Disabled Devices")
                print("[Audio]   4. Right-click 'Stereo Mix' → Enable")
                        
        except Exception as e:
            print(f"[Audio] Setup error: {e}")
            self.source_combo.addItem("🎤 Microphone", "mic")
            
    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
            
    def start_recording(self):
        # Make sure any previous workers are cleaned up
        if self.audio_capture or self.stt_worker or self.whisper_worker:
            print("[DEBUG] Cleaning up previous workers before starting")
            self._cleanup_workers()
        
        mode = self.source_combo.currentData()
        lang = self.lang_combo.currentData()
        use_offline = self.offline_checkbox.isChecked()
        
        print(f"[DEBUG] Starting recording: mode={mode}, lang={lang}, offline={use_offline}")
        print(f"[DEBUG] WHISPER_AVAILABLE={WHISPER_AVAILABLE}")
        
        # Check API credentials for online mode
        if not use_offline and (not config.get('api_key') or not config.get('app_id')):
            QMessageBox.warning(self, "Error", "API credentials not configured in config.json\nTry offline mode instead.")
            return
        
        # Check if offline mode is available
        if use_offline and not WHISPER_AVAILABLE:
            QMessageBox.warning(self, "Offline Mode Unavailable", 
                "Install faster-whisper for offline mode:\n\npip install faster-whisper")
            return
            
        # Check if loopback is available for speaker/both modes
        if mode in ["speaker", "both"] and self.loopback_device is None:
            QMessageBox.warning(
                self, "System Audio Not Available",
                "Enable 'Stereo Mix' in Windows Sound settings:\n\n"
                "1. Right-click speaker icon → Sound settings\n"
                "2. More sound settings → Recording tab\n"
                "3. Right-click → Show Disabled Devices\n"
                "4. Enable 'Stereo Mix'"
            )
            mode = "mic"
        
        # Start audio capture
        self.audio_capture = AudioCapture(
            mic_device=self.mic_device,
            loopback_device=self.loopback_device,
            capture_mode=mode
        )
        self.audio_capture.audio_data.connect(self.on_audio_data)
        self.audio_capture.audio_level.connect(self.on_audio_level)
        self.audio_capture.error_signal.connect(self.on_audio_error)
        
        self.use_offline_mode = use_offline
        self.api_failed = False
        
        if use_offline:
            # Start Whisper offline worker
            # Map language codes to Whisper language codes
            whisper_lang = lang if lang not in ['auto', 'hi-en'] else None
            if lang == 'hi':
                whisper_lang = 'hi'
            elif lang == 'en':
                whisper_lang = 'en'
            
            print(f"[DEBUG] Creating WhisperOfflineWorker with model=tiny, lang={whisper_lang}")
            
            # Pre-load model in main thread to avoid threading issues with CTranslate2
            self.status_label.setText("Loading Whisper model...")
            QApplication.processEvents()  # Update UI immediately
            
            model = WhisperOfflineWorker.preload_model(model_size="tiny", device="cpu")
            if model is None:
                QMessageBox.warning(self, "Model Load Failed", 
                    "Failed to load Whisper model. Check console for errors.")
                return
            
            self.whisper_worker = WhisperOfflineWorker(
                model_size="tiny",  # Options: tiny, base, small, medium, large-v3
                language=whisper_lang,
                device="cpu",
                model=model  # Pass pre-loaded model
            )
            self.whisper_worker.transcription.connect(self.on_transcription)
            self.whisper_worker.status_changed.connect(self.on_status_changed)
            self.whisper_worker.error_signal.connect(self.on_whisper_error)
            self.whisper_worker.start()
            
            print("[DEBUG] WhisperOfflineWorker started")
        else:
            # Start online STT worker
            self.stt_worker = STTWorker(
                api_key=config['api_key'],
                app_id=config['app_id'],
                language=lang,
                domain=config.get('default_domain', 'generic')
            )
            self.stt_worker.transcription.connect(self.on_transcription)
            self.stt_worker.status_changed.connect(self.on_status_changed)
            self.stt_worker.error_signal.connect(self.on_stt_error)
            self.stt_worker.start()
        
        self.audio_capture.start()
        
        self.is_recording = True
        self.partial_text = ""
        self.start_btn.setText("⏹ Stop")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #ef4444; color: white;
                border: none; border-radius: 6px;
                padding: 8px 20px; font-weight: bold;
            }
            QPushButton:hover { background-color: #f87171; }
        """)
        self.source_combo.setEnabled(False)
        self.lang_combo.setEnabled(False)
        self.offline_checkbox.setEnabled(False)
        
    def stop_recording(self):
        """Stop recording - use non-blocking cleanup to prevent UI freeze"""
        print("[DEBUG] stop_recording called")
        
        # Signal all workers to stop first (non-blocking)
        if self.audio_capture:
            self.audio_capture.stop()
            
        if self.stt_worker:
            self.stt_worker.stop()
        
        if self.whisper_worker:
            self.whisper_worker.stop()
        
        # Update UI immediately (don't wait for threads)
        self.is_recording = False
        self.start_btn.setText("▶ Start")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #6366f1; color: white;
                border: none; border-radius: 6px;
                padding: 8px 20px; font-weight: bold;
            }
            QPushButton:hover { background-color: #818cf8; }
        """)
        self.source_combo.setEnabled(True)
        self.lang_combo.setEnabled(True)
        self.offline_checkbox.setEnabled(WHISPER_AVAILABLE)
        self.status_label.setText("Stopped")
        self.audio_level_fill.setGeometry(0, 0, 0, 6)
        
        # Clean up threads in background using QTimer
        QTimer.singleShot(100, self._cleanup_workers)
        
    def _cleanup_workers(self):
        """Clean up worker threads (called after UI update)"""
        if self.audio_capture:
            if not self.audio_capture.wait(200):
                self.audio_capture.terminate()
            self.audio_capture = None
            
        if self.stt_worker:
            if not self.stt_worker.wait(200):
                self.stt_worker.terminate()
            self.stt_worker = None
        
        if self.whisper_worker:
            if not self.whisper_worker.wait(500):
                self.whisper_worker.terminate()
            self.whisper_worker = None
            
        print("[DEBUG] Workers cleaned up")
        
    def on_audio_data(self, data):
        """Send audio to the appropriate worker (online or offline)"""
        if self.use_offline_mode:
            if self.whisper_worker:
                self.whisper_worker.add_audio(data)
        else:
            if self.stt_worker:
                self.stt_worker.add_audio(data)
            # If API failed and we have whisper as fallback, send there too
            if self.api_failed and self.whisper_worker:
                self.whisper_worker.add_audio(data)
    
    def on_audio_level(self, level):
        width = int(level * 80)
        self.audio_level_fill.setGeometry(0, 0, width, 6)
        color = "#ef4444" if level > 0.7 else "#10b981" if level > 0.2 else "#6366f1"
        self.audio_level_fill.setStyleSheet(f"background-color: {color}; border-radius: 3px;")
            
    def on_audio_error(self, error):
        self.status_label.setText(f"Audio Error: {error}")
        self.stop_recording()
    
    def on_whisper_error(self, error):
        """Handle Whisper offline errors"""
        print(f"[Whisper] Error: {error}")
        self.status_label.setText(f"Whisper Error: {error}")
        if not self.stt_worker:  # Only stop if we don't have online fallback
            self.stop_recording()
        
    def on_transcription(self, data):
        """Handle transcription - show in caption box with real-time updates"""
        if not data.get('success'):
            return
        
        cause = data.get('cause', '')
        if cause == 'ready':
            return
            
        text = data.get('display_text') or data.get('text', '')
        is_final = data.get('final', False)
        
        # Debug output
        print(f"[Caption] final={is_final}, cause={cause}, text={text[:50] if text else 'empty'}")
        
        if not text or not text.strip():
            return
        
        if is_final:
            # Final result - clear partial and add final text
            # Remove any partial text first
            current_text = self.caption_display.toPlainText()
            if self.partial_text and current_text.endswith(self.partial_text):
                # Remove the partial text
                new_text = current_text[:-len(self.partial_text)]
                self.caption_display.setPlainText(new_text)
            
            # Add final text with newline
            self.caption_display.append(text)
            self.partial_text = ""
            
            # Auto-scroll
            scrollbar = self.caption_display.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            
        elif cause != 'silence detected':
            # Partial result - show live in caption box
            current_text = self.caption_display.toPlainText()
            
            # Remove old partial text if exists
            if self.partial_text and current_text.endswith(self.partial_text):
                new_text = current_text[:-len(self.partial_text)]
                self.caption_display.setPlainText(new_text)
            
            # Add new partial text (will be replaced on next update)
            cursor = self.caption_display.textCursor()
            cursor.movePosition(cursor.End)
            self.caption_display.setTextCursor(cursor)
            self.caption_display.insertPlainText(text)
            self.partial_text = text
            
            # Update status with short preview
            self.status_label.setText(f"🎤 Listening...")
            
            # Auto-scroll
            scrollbar = self.caption_display.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            
    def on_status_changed(self, status, message):
        if status == "connected":
            self.status_label.setStyleSheet("color: #10b981; font-size: 11px;")
        elif status == "error":
            self.status_label.setStyleSheet("color: #ef4444; font-size: 11px;")
        elif status == "loading":
            self.status_label.setStyleSheet("color: #f59e0b; font-size: 11px;")
        elif status == "ready":
            self.status_label.setStyleSheet("color: #10b981; font-size: 11px;")
        else:
            self.status_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.status_label.setText(message)
        
    def on_stt_error(self, error):
        """Handle online API errors - optionally fallback to offline"""
        print(f"[STT] Error: {error}")
        
        # Check if we should fallback to offline mode
        if WHISPER_AVAILABLE and not self.whisper_worker and self.is_recording:
            self.api_failed = True
            self.status_label.setText(f"API Error - Switching to offline mode...")
            QApplication.processEvents()
            
            # Pre-load model in main thread
            model = WhisperOfflineWorker.preload_model(model_size="tiny", device="cpu")
            if model is None:
                self.status_label.setText(f"Error: {error} (offline fallback failed)")
                self.stop_recording()
                return
            
            # Start whisper worker as fallback
            lang = self.lang_combo.currentData()
            whisper_lang = lang if lang not in ['auto', 'hi-en'] else None
            
            self.whisper_worker = WhisperOfflineWorker(
                model_size="tiny",
                language=whisper_lang,
                device="cpu",
                model=model  # Pass pre-loaded model
            )
            self.whisper_worker.transcription.connect(self.on_transcription)
            self.whisper_worker.status_changed.connect(self.on_status_changed)
            self.whisper_worker.error_signal.connect(self.on_whisper_error)
            self.whisper_worker.start()
        else:
            self.status_label.setText(f"Error: {error}")
            self.stop_recording()
            
    def copy_captions(self):
        QApplication.clipboard().setText(self.caption_display.toPlainText())
        self.status_label.setText("Copied!")
        QTimer.singleShot(1500, lambda: self.status_label.setText("🟢 Connected") if self.is_recording else None)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.drag_position:
            self.move(event.globalPos() - self.drag_position)
            
    def mouseReleaseEvent(self, event):
        self.drag_position = None
        
    def closeEvent(self, event):
        self.stop_recording()
        event.accept()


def exception_hook(exctype, value, tb):
    """Global exception hook to catch unhandled exceptions"""
    import traceback
    print("=" * 50)
    print("UNHANDLED EXCEPTION:")
    print("=" * 50)
    traceback.print_exception(exctype, value, tb)
    print("=" * 50)
    sys.__excepthook__(exctype, value, tb)


# Global whisper model - MUST be loaded before QApplication due to CTranslate2/Qt conflict
def main():
    # Set up global exception hook
    sys.excepthook = exception_hook
    
    # Note: Whisper model is pre-loaded at module import time (before Qt)
    if WHISPER_AVAILABLE:
        if _WHISPER_MODEL is not None:
            print("[Whisper] Using pre-loaded model")
        else:
            print("[Whisper] Warning: Model not pre-loaded, offline mode may not work")
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(15, 23, 42))
    palette.setColor(QPalette.WindowText, QColor(248, 250, 252))
    palette.setColor(QPalette.Base, QColor(30, 41, 59))
    palette.setColor(QPalette.Text, QColor(248, 250, 252))
    palette.setColor(QPalette.Button, QColor(51, 65, 85))
    palette.setColor(QPalette.ButtonText, QColor(248, 250, 252))
    palette.setColor(QPalette.Highlight, QColor(99, 102, 241))
    app.setPalette(palette)
    
    window = CaptionOverlay()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
