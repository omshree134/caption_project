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


# Supported languages for ASR (source languages) - shown in native script
LANGUAGES = {
    "hi": "हिन्दी",
    "en": "English", 
    "bn": "বাংলা",
    "ta": "தமிழ்",
    "te": "తెలుగు",
    "mr": "मराठी",
    "gu": "ગુજરાતી",
    "kn": "ಕನ್ನಡ",
    "ml": "മലയാളം",
    "pa": "ਪੰਜਾਬੀ",
    "or": "ଓଡ଼ିଆ",
    "as": "অসমীয়া",
    "ur": "اردو",
}

# All languages supported for translation (IndicTrans2 supports all 22 scheduled Indian languages)
# Shown in native script
TRANSLATION_LANGUAGES = {
    "en": "English",
    "hi": "हिन्दी",
    "bn": "বাংলা",
    "ta": "தமிழ்",
    "te": "తెలుగు",
    "mr": "मराठी",
    "gu": "ગુજરાતી",
    "kn": "ಕನ್ನಡ",
    "ml": "മലയാളം",
    "pa": "ਪੰਜਾਬੀ",
    "or": "ଓଡ଼ିଆ",
    "as": "অসমীয়া",
    "ur": "اردو",
    "ne": "नेपाली",
    "sa": "संस्कृतम्",
    "kok": "कोंकणी",
    "mai": "मैथिली",
    "doi": "डोगरी",
    "sat": "Santali (संताली)",
    "ks": "کٲشُر",
    "mni": "মৈতৈলোন্",
    "sd": "سنڌي",
    "brx": "बर'",
}

# Reverie API supported languages (subset of full translation languages)
# Shown in native script
REVERIE_LANGUAGES = {
    "en": "English",
    "hi": "हिन्दी",
    "bn": "বাংলা",
    "ta": "தமிழ்",
    "te": "తెలుగు",
    "mr": "मराठी",
    "gu": "ગુજરાતી",
    "kn": "ಕನ್ನಡ",
    "ml": "മലയാളം",
    "pa": "ਪੰਜਾਬੀ",
    "or": "ଓଡ଼ିଆ",
    "as": "অসমীয়া",
    "ur": "اردو",
    "ne": "नेपाली",
    "kok": "कोंकणी",
    "mai": "मैथिली",
}

# ============================================================================
# SENTENCE TERMINATORS FOR ALL SUPPORTED LANGUAGES
# Used to detect sentence boundaries in translation
# ============================================================================

# Full stop / Period equivalents by language
SENTENCE_TERMINATORS = {
    # Full stops (end of sentence)
    "en": ['.', '!', '?'],                    # English
    "hi": ['।', '॥', '.', '!', '?'],          # Hindi - Devanagari Danda & Double Danda
    "bn": ['।', '॥', '.', '!', '?'],          # Bengali - uses Danda
    "ta": ['।', '.', '!', '?'],               # Tamil
    "te": ['।', '.', '!', '?'],               # Telugu
    "mr": ['।', '॥', '.', '!', '?'],          # Marathi - Devanagari
    "gu": ['।', '॥', '.', '!', '?'],          # Gujarati - uses Danda
    "kn": ['।', '.', '!', '?'],               # Kannada
    "ml": ['।', '.', '!', '?'],               # Malayalam
    "pa": ['।', '॥', '.', '!', '?'],          # Punjabi (Gurmukhi)
    "or": ['।', '॥', '.', '!', '?'],          # Odia - uses Danda
    "as": ['।', '॥', '.', '!', '?'],          # Assamese - uses Bengali script
    "ur": ['۔', '؟', '!', '.'],               # Urdu - Arabic full stop
    "ne": ['।', '॥', '.', '!', '?'],          # Nepali - Devanagari
    "kok": ['।', '॥', '.', '!', '?'],         # Konkani - Devanagari
    "mai": ['।', '॥', '.', '!', '?'],         # Maithili - Devanagari
    "sa": ['।', '॥', '.', '!', '?'],          # Sanskrit - Devanagari
    "doi": ['।', '॥', '.', '!', '?'],         # Dogri
    "sat": ['।', '.', '!', '?'],              # Santali
    "ks": ['۔', '؟', '!', '.'],               # Kashmiri
    "mni": ['।', '.', '!', '?'],              # Manipuri
    "sd": ['۔', '؟', '!', '.'],               # Sindhi - Arabic script
    "brx": ['।', '.', '!', '?'],              # Bodo - Devanagari
}

# Clause terminators (partial sentence, comma-like pauses)
CLAUSE_TERMINATORS = {
    "en": [',', ';', ':', '-'],
    "hi": [',', '॰', ';', ':'],               # Hindi comma, abbreviation mark
    "bn": [',', ';', ':'],
    "ta": [',', ';', ':'],
    "te": [',', ';', ':'],
    "mr": [',', ';', ':'],
    "gu": [',', ';', ':'],
    "kn": [',', ';', ':'],
    "ml": [',', ';', ':'],
    "pa": [',', ';', ':'],
    "or": [',', ';', ':'],
    "as": [',', ';', ':'],
    "ur": ['،', '؛', ':'],                    # Urdu comma, semicolon
    "ne": [',', ';', ':'],
    "kok": [',', ';', ':'],
    "mai": [',', ';', ':'],
    "sa": [',', ';', ':'],
    "doi": [',', ';', ':'],
    "sat": [',', ';', ':'],
    "ks": ['،', '؛', ':'],
    "mni": [',', ';', ':'],
    "sd": ['،', '؛', ':'],
    "brx": [',', ';', ':'],
}

def is_sentence_complete(text, lang="en"):
    """Check if text ends with a sentence terminator for the given language"""
    if not text:
        return False
    text = text.strip()
    if not text:
        return False
    
    terminators = SENTENCE_TERMINATORS.get(lang, SENTENCE_TERMINATORS["en"])
    return text[-1] in terminators

def is_clause_complete(text, lang="en"):
    """Check if text ends with a clause terminator (comma, semicolon, etc.)"""
    if not text:
        return False
    text = text.strip()
    if not text:
        return False
    
    clause_terms = CLAUSE_TERMINATORS.get(lang, CLAUSE_TERMINATORS["en"])
    sentence_terms = SENTENCE_TERMINATORS.get(lang, SENTENCE_TERMINATORS["en"])
    return text[-1] in clause_terms or text[-1] in sentence_terms
