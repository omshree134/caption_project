"""
Constants and global variables for the Caption App
CRITICAL: Whisper must be loaded BEFORE PyQt5 to avoid CTranslate2/Qt threading conflicts
"""

import sys

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


def get_whisper_model():
    """Get the pre-loaded Whisper model"""
    return _WHISPER_MODEL


def get_vad():
    """Get the VAD module"""
    if VAD_AVAILABLE:
        try:
            import webrtcvad
            return webrtcvad
        except ImportError:
            import webrtcvad_wheels as webrtcvad
            return webrtcvad
    return None


# Supported languages for ASR (source languages)
LANGUAGES = {
    "hi": "Hindi",
    "en": "English", 
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "ur": "Urdu",
}

# All languages supported for translation (IndicTrans2 supports all 22 scheduled Indian languages)
TRANSLATION_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "ur": "Urdu",
    "ne": "Nepali",
    "sa": "Sanskrit",
    "kok": "Konkani",
    "mai": "Maithili",
    "doi": "Dogri",
    "sat": "Santali",
    "ks": "Kashmiri",
    "mni": "Manipuri",
    "sd": "Sindhi",
    "brx": "Bodo",
}
