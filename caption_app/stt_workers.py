"""
Speech-to-Text workers for the Caption App
Includes online (Reverie API) and offline (Whisper) transcription
"""

import json
import asyncio
import queue
import time
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

import websockets

from .constants import WHISPER_AVAILABLE, VAD_AVAILABLE, get_whisper_model


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
        self.audio_queue = queue.Queue(maxsize=100)
        self.websocket = None
        self.loop = None
            
    def build_url(self):
        lang = self.language
            
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
            # Add connection timeout to detect offline faster
            self.websocket = await asyncio.wait_for(
                websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=3
                ),
                timeout=10.0  # 10 second connection timeout
            )
            
            async with self.websocket:
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
                
        except asyncio.TimeoutError:
            print("[STT] Connection timeout - no internet?")
            self.error_signal.emit("Connection timeout - check internet connection")
        except websockets.exceptions.InvalidStatusCode as e:
            self.error_signal.emit(f"Auth failed (HTTP {e.status_code})")
        except (OSError, ConnectionRefusedError, ConnectionResetError) as e:
            print(f"[STT] Network error: {e}")
            self.error_signal.emit(f"Network error: {e}")
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
    
    # Class-level model cache
    _cached_model = None
    _cached_model_size = None
    
    @classmethod
    def preload_model(cls, model_size="tiny", device="cpu"):
        """Pre-load model - use the global pre-loaded model"""
        # First check global model (loaded before Qt)
        global_model = get_whisper_model()
        if global_model is not None:
            print(f"[Whisper] Using globally pre-loaded model", flush=True)
            cls._cached_model = global_model
            cls._cached_model_size = model_size
            return global_model
        
        # Then check class cache
        if cls._cached_model is not None and cls._cached_model_size == model_size:
            print(f"[Whisper] Using cached model '{model_size}'", flush=True)
            return cls._cached_model
            
        if not WHISPER_AVAILABLE:
            print("[Whisper] faster-whisper not available", flush=True)
            return None
        
        print(f"[Whisper] WARNING: Loading model after Qt - this may crash!", flush=True)
        
        try:
            from faster_whisper import WhisperModel
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
            return None
    
    def __init__(self, model_size="tiny", language="en", device="cpu", model=None):
        super().__init__()
        self.model_size = model_size
        self.language = language
        self.device = device
        self.running = False
        self.model = model
        
        # Audio parameters
        self.sample_rate = 16000
        self.audio_queue = queue.Queue(maxsize=200)
        
        # VAD parameters
        self.vad = None
        self.vad_mode = 3
        
        # Batching parameters
        self.min_speech_duration = 0.2
        self.max_batch_duration = 5.0
        
        # Silence detection - TWO LEVELS
        self.short_silence_frames = 6   # ~192ms
        self.long_silence_frames = 20   # ~640ms
        self.trailing_silence_frames = 3
        
        # State
        self.audio_buffer = []
        self.speech_buffer = []
        self.is_speaking = False
        self.silence_frames = 0
        self.speech_frames = 0
        self.last_transcription_time = 0
        self.pending_text = ""
        
    def _init_vad(self):
        """Initialize Voice Activity Detection"""
        if VAD_AVAILABLE:
            try:
                import webrtcvad
                self.vad = webrtcvad.Vad(self.vad_mode)
                print(f"[VAD] WebRTC VAD initialized (mode={self.vad_mode})")
                return True
            except Exception as e:
                print(f"[VAD] Failed to initialize: {e}")
        return False
    
    def _energy_vad(self, audio_chunk):
        """Simple energy-based VAD fallback"""
        energy = np.sqrt(np.mean(audio_chunk.astype(np.float32)**2))
        threshold = 500
        return energy > threshold
    
    def _check_vad(self, audio_bytes):
        """Check if audio contains speech using VAD"""
        if self.vad:
            try:
                frame_duration = 30
                frame_size = int(self.sample_rate * frame_duration / 1000) * 2
                
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
                
                if total_frames > 0:
                    return speech_frames / total_frames > 0.3
                return False
            except Exception as e:
                audio = np.frombuffer(audio_bytes, dtype=np.int16)
                return self._energy_vad(audio)
        else:
            audio = np.frombuffer(audio_bytes, dtype=np.int16)
            return self._energy_vad(audio)
    
    def _load_model(self):
        """Load the Whisper model (use pre-loaded if available)"""
        if self.model is not None:
            print(f"[Whisper] Using pre-loaded model", flush=True)
            self.status_changed.emit("ready", f"🟢 Whisper ({self.model_size}) ready")
            self.model_loaded.emit(True)
            return True
            
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
            from faster_whisper import WhisperModel
            print(f"[Whisper] Loading model '{self.model_size}' on {self.device}...", flush=True)
            self.status_changed.emit("loading", f"Loading Whisper {self.model_size}...")
            
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type="int8" if self.device == "cpu" else "float16"
            )
            
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
            self._init_vad()
            
            if not self._load_model():
                print("[Whisper] Model failed to load, exiting worker")
                return
            
            self.status_changed.emit("connected", "🟢 Whisper (Offline)")
            print("[Whisper] Worker ready and listening...")
            
            while self.running:
                try:
                    try:
                        audio_data = self.audio_queue.get(timeout=0.05)
                    except queue.Empty:
                        self._check_and_transcribe()
                        continue
                    
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
                            if self.silence_frames < self.trailing_silence_frames:
                                self.speech_buffer.append(audio_data)
                            elif self.silence_frames == self.short_silence_frames:
                                self._transcribe_buffer(is_sentence_end=False)
                            elif self.silence_frames >= self.long_silence_frames:
                                self.is_speaking = False
                                self._transcribe_buffer(is_sentence_end=True)
                    
                    buffer_duration = len(self.speech_buffer) * 0.032
                    if buffer_duration >= self.max_batch_duration:
                        self._transcribe_buffer(is_sentence_end=False)
                        
                except Exception as e:
                    print(f"[Whisper] Processing error: {e}")
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            print(f"[Whisper] Worker thread error: {e}")
            self.error_signal.emit(f"Worker error: {e}")
        finally:
            print("[Whisper] Worker thread ending")
                
    def _check_and_transcribe(self):
        """Check if we should transcribe based on time"""
        if not self.speech_buffer:
            return
            
        time_since_last = time.time() - self.last_transcription_time
        
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
            audio_bytes = b''.join(self.speech_buffer)
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            self.speech_buffer.clear()
            self.speech_frames = 0
            
            if len(audio) < self.sample_rate * self.min_speech_duration:
                if is_sentence_end and self.pending_text:
                    self._flush_pending_text()
                return
            
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
            
            text_parts = []
            for segment in segments:
                text = segment.text.strip()
                text = text.rstrip('.,!?。，')
                if text:
                    text_parts.append(text)
            
            new_text = ' '.join(text_parts).strip()
            
            if new_text:
                if self.pending_text:
                    self.pending_text += ", " + new_text
                else:
                    self.pending_text = new_text
                
                if is_sentence_end:
                    self.pending_text += "."
                    self._flush_pending_text()
                else:
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
                self.pending_text += "."
                self._flush_pending_text()
            
            self.last_transcription_time = time.time()
            
        except Exception as e:
            print(f"[Whisper] Error: {e}")
    
    def stop(self):
        """Stop the worker"""
        self.running = False
        if self.speech_buffer:
            self._transcribe_buffer(is_sentence_end=True)
        if self.pending_text:
            self._flush_pending_text()
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break


class LanguageDetector(QThread):
    """
    Language detection using Whisper's transcribe function.
    Runs independently and emits detected language for auto-switching.
    Uses stability checks to prevent constant language flipping.
    """
    language_detected = pyqtSignal(str, float)  # language_code, confidence
    status_changed = pyqtSignal(str)
    
    def __init__(self, model=None):
        super().__init__()
        self.model = model
        self.running = False
        self.audio_queue = queue.Queue(maxsize=100)  # Larger queue
        self.sample_rate = 16000
        self.last_detection_time = 0
        self.audio_buffer = []
        
        # Initial detection settings - need more audio for non-English
        self.initial_detection_interval = 2.0  # Wait 2 seconds
        self.initial_min_audio = 2.5  # Need 2.5 seconds of audio for accuracy
        self.initial_confidence = 0.50  # Lower threshold - 50% is enough for initial
        
        # Subsequent detection settings  
        self.stable_detection_interval = 4.0  # Check every 4 seconds
        self.stable_min_audio = 3.0  # Need 3 seconds for switching
        self.stable_confidence = 0.60  # 60% confidence to switch
        self.consecutive_required = 2  # Need 2 consecutive for switch
        self.switch_cooldown = 8.0  # 8 second cooldown
        
        # Tracking state
        self.consecutive_detections = {}  # lang_code -> count
        self.last_switch_time = 0
        self.current_language = None  # Currently active language
        self.initial_detection_done = False  # Track if we've done initial detection
        
    def add_audio(self, audio_data):
        """Add audio data for language detection"""
        if self.running:
            try:
                self.audio_queue.put_nowait(audio_data)
            except queue.Full:
                pass  # Skip if queue is full
    
    def run(self):
        """Main detection loop"""
        if not WHISPER_AVAILABLE:
            self.status_changed.emit("Whisper not available for language detection")
            return
        
        # Get or load model
        if self.model is None:
            self.model = get_whisper_model()
        
        if self.model is None:
            self.status_changed.emit("Failed to load Whisper model")
            return
        
        self.running = True
        self.initial_detection_done = False
        self.status_changed.emit("Language detection active")
        print("[LangDetect] Started - fast initial detection mode")
        
        total_audio_bytes = 0  # Track total audio received
        
        while self.running:
            try:
                # Collect audio data
                try:
                    audio_data = self.audio_queue.get(timeout=0.1)
                    self.audio_buffer.append(audio_data)
                    total_audio_bytes += len(audio_data)
                except queue.Empty:
                    continue
                
                # Use different settings for initial vs subsequent detection
                if not self.initial_detection_done:
                    detection_interval = self.initial_detection_interval
                    min_audio = self.initial_min_audio
                else:
                    detection_interval = self.stable_detection_interval
                    min_audio = self.stable_min_audio
                
                # Calculate buffer duration based on actual bytes
                # 16kHz, 16-bit (2 bytes per sample), mono = 32000 bytes per second
                buffer_bytes = sum(len(chunk) for chunk in self.audio_buffer)
                buffer_duration = buffer_bytes / 32000.0  # seconds
                
                current_time = time.time()
                time_since_last = current_time - self.last_detection_time
                
                if (time_since_last >= detection_interval and buffer_duration >= min_audio):
                    print(f"[LangDetect] Processing {buffer_duration:.1f}s of audio...")
                    
                    # Concatenate audio buffer
                    audio_bytes = b''.join(self.audio_buffer)
                    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Detect language using Whisper with language=None for auto-detection
                    try:
                        segments, info = self.model.transcribe(
                            audio,
                            language=None,  # Let Whisper auto-detect language
                            beam_size=5,    # Better accuracy with higher beam
                            best_of=3,      # Try multiple hypotheses
                            task="transcribe",
                            vad_filter=True,
                            vad_parameters=dict(
                                min_silence_duration_ms=300,
                                speech_pad_ms=200,
                            ),
                            condition_on_previous_text=False,
                            word_timestamps=False,
                        )
                        # Consume the generator and get sample text
                        sample_text = ""
                        for seg in segments:
                            sample_text += seg.text + " "
                            if len(sample_text) > 50:
                                break
                        
                        detected_lang = info.language
                        confidence = info.language_probability
                        
                        # Log with sample text for debugging
                        sample_preview = sample_text[:40].strip() if sample_text else "(no speech)"
                        print(f"[LangDetect] Result: {detected_lang} ({confidence:.1%}) - \"{sample_preview}...\"")
                        
                        # Process the detection with stability checks
                        self._process_detection(detected_lang, confidence, current_time)
                        
                    except Exception as e:
                        print(f"[LangDetect] Detection error: {e}")
                    
                    # Reset for next detection
                    self.last_detection_time = current_time
                    self.audio_buffer = []
                    
                # Limit buffer size to prevent memory issues (5 seconds = 160000 bytes)
                max_buffer_bytes = 160000
                current_bytes = sum(len(chunk) for chunk in self.audio_buffer)
                if current_bytes > max_buffer_bytes:
                    # Trim from the beginning
                    while current_bytes > max_buffer_bytes and self.audio_buffer:
                        removed = self.audio_buffer.pop(0)
                        current_bytes -= len(removed)
                    
            except Exception as e:
                print(f"[LangDetect] Error: {e}")
                time.sleep(0.1)
        
        print("[LangDetect] Stopped language detection")
    
    def _process_detection(self, detected_lang, confidence, current_time):
        """Process a language detection with stability checks"""
        
        # Use different confidence thresholds for initial vs subsequent
        if not self.initial_detection_done:
            required_confidence = self.initial_confidence
        else:
            required_confidence = self.stable_confidence
        
        # Check confidence threshold
        if confidence < required_confidence:
            print(f"[LangDetect] Low confidence: {detected_lang} ({confidence:.1%}) < {required_confidence:.0%}")
            return
        
        # FAST PATH: First detection - immediately set language
        if not self.initial_detection_done:
            print(f"[LangDetect] ✓ Initial detection: {detected_lang} ({confidence:.1%})")
            self.current_language = detected_lang
            self.initial_detection_done = True
            self.last_switch_time = current_time
            self.language_detected.emit(detected_lang, confidence)
            return
        
        # Check cooldown period after last switch
        time_since_switch = current_time - self.last_switch_time
        if time_since_switch < self.switch_cooldown:
            remaining = self.switch_cooldown - time_since_switch
            print(f"[LangDetect] Cooldown: {remaining:.1f}s remaining ({detected_lang} {confidence:.1%})")
            return
        
        # If same as current language, no action needed (just confirm)
        if detected_lang == self.current_language:
            # Reset any other language counts when we confirm current
            self.consecutive_detections = {}
            return
        
        # Track consecutive detections for this new language
        if detected_lang not in self.consecutive_detections:
            self.consecutive_detections = {detected_lang: 1}
        else:
            self.consecutive_detections[detected_lang] += 1
        
        count = self.consecutive_detections[detected_lang]
        print(f"[LangDetect] New lang: {detected_lang} ({confidence:.1%}) - {count}/{self.consecutive_required}")
        
        # Check if we have enough consecutive detections to switch
        if count >= self.consecutive_required:
            print(f"[LangDetect] ✓ Switching: {self.current_language} → {detected_lang}")
            self.current_language = detected_lang
            self.last_switch_time = current_time
            self.consecutive_detections = {}
            self.language_detected.emit(detected_lang, confidence)
    
    def stop(self):
        """Stop detection"""
        self.running = False
        self.audio_buffer = []
        self.consecutive_detections = {}
        self.initial_detection_done = False
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break
