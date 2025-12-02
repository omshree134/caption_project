"""
Audio capture module for the Caption App
Handles microphone, system audio (WASAPI loopback), and mixed capture
"""

import threading
import numpy as np
import sounddevice as sd
from PyQt5.QtCore import QThread, pyqtSignal


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
                
    def stop(self):
        self.running = False
