"""
Translation module for the Caption App
Supports:
- Reverie API for online translation (low latency)
- IndicTrans2 for offline translation of Indian languages
"""

import queue
import requests
from PyQt5.QtCore import QThread, pyqtSignal

from .config import config

# Global model cache
_INDICTRANS_EN_INDIC_MODEL = None
_INDICTRANS_INDIC_EN_MODEL = None
_INDICTRANS_TOKENIZER_EN_INDIC = None
_INDICTRANS_TOKENIZER_INDIC_EN = None
_INDIC_PROCESSOR = None
INDICTRANS_AVAILABLE = False

# Language code mapping: Our codes -> IndicTrans2 codes
INDICTRANS_LANG_MAP = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "pa": "pan_Guru",
    "or": "ory_Orya",
    "as": "asm_Beng",
    "ur": "urd_Arab",
    "ne": "npi_Deva",
    "sa": "san_Deva",
    "kok": "gom_Deva",
    "mai": "mai_Deva",
    "doi": "doi_Deva",
    "sat": "sat_Olck",
    "ks": "kas_Deva",
    "mni": "mni_Beng",
    "sd": "snd_Arab",
    "brx": "brx_Deva",
}

# ============== REVERIE API TRANSLATION ==============

REVERIE_TRANSLATION_URL = "https://revapi.reverieinc.com/"

def translate_reverie(text, src_lang, tgt_lang, timeout=3.0):
    """
    Translate text using Reverie Translation API (online, low latency)
    
    Args:
        text: Text to translate (string or list of strings)
        src_lang: Source language code (e.g., 'hi', 'en')
        tgt_lang: Target language code (e.g., 'en', 'hi')
        timeout: Request timeout in seconds
    
    Returns:
        Translated text (string or list), or original text on error
    """
    api_key = config.get('api_key', '')
    app_id = config.get('app_id', '')
    
    if not api_key or not app_id:
        print("[Reverie Translation] Missing API credentials")
        return text
    
    # Handle single string or list
    is_single = isinstance(text, str)
    texts = [text] if is_single else text
    
    # Filter empty texts
    texts = [t.strip() for t in texts if t and t.strip()]
    if not texts:
        return "" if is_single else []
    
    try:
        headers = {
            'Content-Type': 'application/json',
            'REV-API-KEY': api_key,
            'REV-APP-ID': app_id,
            'REV-APPNAME': 'localization',
            'REV-APPVERSION': '3.0',
            'src_lang': src_lang,
            'tgt_lang': tgt_lang,
            'domain': config.get('default_domain', 'generic'),
        }
        
        payload = {
            'data': texts,
            'enableNmt': True,
            'enableLookup': True,
        }
        
        print(f"[Reverie Translation] Request: {src_lang} → {tgt_lang}, text: {texts[0][:50] if texts else 'empty'}...")
        
        response = requests.post(
            REVERIE_TRANSLATION_URL,
            headers=headers,
            json=payload,
            timeout=timeout
        )
        
        print(f"[Reverie Translation] Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            response_list = result.get('responseList', [])
            
            if response_list:
                # Extract translations
                translations = []
                for item in response_list:
                    out_string = item.get('outString', '')
                    if out_string:
                        translations.append(out_string)
                        print(f"[Reverie Translation] Success: {item.get('inString', '')[:30]}... → {out_string[:30]}...")
                    else:
                        # Fallback to input
                        translations.append(item.get('inString', ''))
                
                return translations[0] if is_single else translations
            else:
                print(f"[Reverie Translation] Empty response list: {result}")
                return text
        else:
            try:
                error_msg = response.json().get('message', response.text)
            except:
                error_msg = response.text
            print(f"[Reverie Translation] API error {response.status_code}: {error_msg}")
            return text
            
    except requests.Timeout:
        print("[Reverie Translation] Request timeout")
        return text
    except Exception as e:
        print(f"[Reverie Translation] Error: {e}")
        return text


# ============== INDICTRANS2 OFFLINE TRANSLATION ==============

# Check if IndicTrans2 is available - use lazy loading to avoid slow startup
try:
    import importlib.util
    _transformers_spec = importlib.util.find_spec("transformers")
    _torch_spec = importlib.util.find_spec("torch")
    
    if _transformers_spec is not None and _torch_spec is not None:
        INDICTRANS_AVAILABLE = True
        print("[Translation] IndicTrans2 dependencies found (will load on demand)")
    else:
        print("[Translation] Missing transformers or torch packages")
except Exception as e:
    print(f"[Translation] IndicTrans2 check failed: {e}")


def load_indictrans_models(device="cpu"):
    """Load IndicTrans2 models for translation"""
    global _INDICTRANS_EN_INDIC_MODEL, _INDICTRANS_INDIC_EN_MODEL
    global _INDICTRANS_TOKENIZER_EN_INDIC, _INDICTRANS_TOKENIZER_INDIC_EN
    global _INDIC_PROCESSOR
    
    if not INDICTRANS_AVAILABLE:
        print("[Translation] IndicTrans2 not available")
        return False
    
    # Check if already loaded
    if _INDIC_PROCESSOR is not None:
        print("[Translation] Models already loaded")
        return True
    
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from IndicTransToolkit.processor import IndicProcessor
        
        print("[Translation] Loading IndicTrans2 models (this may take a while)...")
        
        # Determine torch dtype
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        # Load En->Indic model (distilled 200M)
        en_indic_model = "ai4bharat/indictrans2-en-indic-dist-200M"
        print(f"[Translation] Loading {en_indic_model}...")
        _INDICTRANS_TOKENIZER_EN_INDIC = AutoTokenizer.from_pretrained(
            en_indic_model, trust_remote_code=True
        )
        _INDICTRANS_EN_INDIC_MODEL = AutoModelForSeq2SeqLM.from_pretrained(
            en_indic_model,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        _INDICTRANS_EN_INDIC_MODEL.eval()  # Set to eval mode for faster inference
        if device == "cuda":
            _INDICTRANS_EN_INDIC_MODEL = _INDICTRANS_EN_INDIC_MODEL.to(device)
        print("[Translation] En->Indic model loaded!")
        
        # Load Indic->En model (distilled 200M)
        indic_en_model = "ai4bharat/indictrans2-indic-en-dist-200M"
        print(f"[Translation] Loading {indic_en_model}...")
        _INDICTRANS_TOKENIZER_INDIC_EN = AutoTokenizer.from_pretrained(
            indic_en_model, trust_remote_code=True
        )
        _INDICTRANS_INDIC_EN_MODEL = AutoModelForSeq2SeqLM.from_pretrained(
            indic_en_model,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        _INDICTRANS_INDIC_EN_MODEL.eval()  # Set to eval mode for faster inference
        if device == "cuda":
            _INDICTRANS_INDIC_EN_MODEL = _INDICTRANS_INDIC_EN_MODEL.to(device)
        print("[Translation] Indic->En model loaded!")
        
        # Initialize processor
        _INDIC_PROCESSOR = IndicProcessor(inference=True)
        print("[Translation] IndicProcessor initialized!")
        
        return True
        
    except Exception as e:
        print(f"[Translation] Failed to load models: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_indictrans_models():
    """Get the loaded IndicTrans2 models"""
    return {
        'en_indic_model': _INDICTRANS_EN_INDIC_MODEL,
        'indic_en_model': _INDICTRANS_INDIC_EN_MODEL,
        'en_indic_tokenizer': _INDICTRANS_TOKENIZER_EN_INDIC,
        'indic_en_tokenizer': _INDICTRANS_TOKENIZER_INDIC_EN,
        'processor': _INDIC_PROCESSOR,
    }


def is_indic_language(lang_code):
    """Check if a language is an Indic language (not English)"""
    return lang_code != "en" and lang_code in INDICTRANS_LANG_MAP


def translate_text(text, src_lang, tgt_lang, device="cpu"):
    """
    Translate text using IndicTrans2
    
    Args:
        text: Text to translate (string or list of strings)
        src_lang: Source language code (e.g., 'hi', 'en')
        tgt_lang: Target language code (e.g., 'en', 'hi')
        device: 'cpu' or 'cuda'
    
    Returns:
        Translated text (string or list)
    """
    if not INDICTRANS_AVAILABLE:
        return text
    
    import torch
    
    models = get_indictrans_models()
    if not models['processor']:
        print("[Translation] Models not loaded")
        return text
    
    # Convert to list for batch processing
    is_single = isinstance(text, str)
    texts = [text] if is_single else text
    
    # Filter empty texts
    texts = [t.strip() for t in texts if t and t.strip()]
    if not texts:
        return "" if is_single else []
    
    # Get IndicTrans2 language codes
    src_indic = INDICTRANS_LANG_MAP.get(src_lang)
    tgt_indic = INDICTRANS_LANG_MAP.get(tgt_lang)
    
    if not src_indic or not tgt_indic:
        print(f"[Translation] Unsupported language pair: {src_lang} -> {tgt_lang}")
        return text
    
    try:
        # Select appropriate model based on direction
        if src_lang == "en":
            # English -> Indic
            model = models['en_indic_model']
            tokenizer = models['en_indic_tokenizer']
        elif tgt_lang == "en":
            # Indic -> English
            model = models['indic_en_model']
            tokenizer = models['indic_en_tokenizer']
        else:
            # Indic -> Indic (pivot through English)
            # First translate to English
            intermediate = translate_text(texts, src_lang, "en", device)
            # Then translate to target
            return translate_text(intermediate, "en", tgt_lang, device)
        
        if model is None or tokenizer is None:
            print("[Translation] Model not loaded")
            return text
        
        processor = models['processor']
        
        # Preprocess
        batch = processor.preprocess_batch(texts, src_lang=src_indic, tgt_lang=tgt_indic)
        
        # Tokenize
        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        # Move to device if needed
        if device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate translation - use greedy decoding for speed (no beam search)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=128,  # Reduced for faster inference
                num_beams=1,  # Greedy decoding - much faster than beam search
                do_sample=False,  # Deterministic output
                num_return_sequences=1,
            )
        
        # Decode
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        
        # Postprocess
        translations = processor.postprocess_batch(decoded, lang=tgt_indic)
        
        return translations[0] if is_single else translations
        
    except Exception as e:
        print(f"[Translation] Error: {e}")
        import traceback
        traceback.print_exc()
        return text


class TranslationWorker(QThread):
    """Background worker for translation to avoid blocking UI
    
    Supports two modes:
    - Online: Uses Reverie API (fast, requires internet)
    - Offline: Uses IndicTrans2 models (slower, no internet needed)
    """
    translation_ready = pyqtSignal(str, str)  # original, translated
    model_loaded = pyqtSignal(str)  # model name
    error_signal = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    loading_started = pyqtSignal()  # Signal when model loading begins
    
    def __init__(self, tgt_lang="en", device="cpu", preload_only=False, use_online=True):
        super().__init__()
        self.tgt_lang = tgt_lang
        self.device = device
        self.running = False
        self.models_loaded = False
        self.preload_only = preload_only  # If True, just load models and stay ready
        self.use_online = use_online  # True = Reverie API, False = IndicTrans2
        self.translation_queue = queue.Queue()
        self._skip_older_than = 0  # Timestamp to skip old queued items
        self._last_processed_text = ""  # Track last processed to avoid duplicates
        self._offline_models_loaded = False  # Track if offline models are loaded
        
    def set_target_language(self, tgt_lang):
        """Update target language"""
        self.tgt_lang = tgt_lang
        print(f"[Translation Worker] Target language set to: {tgt_lang}")
    
    def set_online_mode(self, use_online):
        """Switch between online (Reverie) and offline (IndicTrans2) translation"""
        old_mode = self.use_online
        self.use_online = use_online
        
        if old_mode != use_online:
            mode_name = "online (Reverie API)" if use_online else "offline (IndicTrans2)"
            print(f"[Translation Worker] Switched to {mode_name}")
            
            # If switching to offline mode and models not loaded, need to load them
            if not use_online and not self._offline_models_loaded and self.running:
                self._load_offline_models()
        
    def add_text(self, text, src_lang):
        """Add text to translation queue with source language"""
        # For online mode, we can process immediately
        # For offline mode, need models loaded
        if self.running and text and text.strip():
            if self.use_online or self._offline_models_loaded:
                import time
                self.translation_queue.put((text.strip(), src_lang, time.time()))
    
    def clear_queue(self):
        """Clear pending translations (called when newer text arrives)"""
        import time
        self._skip_older_than = time.time()
    
    def reset_translation_cache(self):
        """Reset incremental translation cache (call when starting new sentence)"""
        self._cached_translation = ""
        self._cached_source = ""
        print("[Translation Worker] Cache reset for new sentence")
        
    def is_ready(self):
        """Check if worker is ready for translation"""
        if self.use_online:
            return self.running  # Online mode always ready if running
        else:
            return self.running and self._offline_models_loaded
    
    def _load_offline_models(self):
        """Load offline IndicTrans2 models"""
        if self._offline_models_loaded:
            return True
            
        self.loading_started.emit()
        self.status_changed.emit("Loading offline translation models...")
        print("[Translation Worker] Loading IndicTrans2 models...")
        
        success = load_indictrans_models(self.device)
        if success:
            self._offline_models_loaded = True
            self.model_loaded.emit("IndicTrans2")
            self.status_changed.emit("Offline translation ready")
            print("[Translation Worker] IndicTrans2 models loaded successfully")
            return True
        else:
            self.error_signal.emit("Failed to load offline translation models")
            return False
    
    def run(self):
        """Main translation loop"""
        self.running = True
        
        if self.use_online:
            # Online mode - ready immediately
            self.models_loaded = True
            self.model_loaded.emit("Reverie API")
            self.status_changed.emit("Online translation ready")
            print("[Translation Worker] Online mode - using Reverie API")
        else:
            # Offline mode - need to load models
            self.loading_started.emit()
            self.status_changed.emit("Loading translation models...")
            print("[Translation Worker] Starting offline model loading...")
            
            success = load_indictrans_models(self.device)
            if success:
                self._offline_models_loaded = True
                self.models_loaded = True
                self.model_loaded.emit("IndicTrans2")
                self.status_changed.emit("Translation ready")
                print("[Translation Worker] Models loaded successfully")
            else:
                self.error_signal.emit("Failed to load translation models")
                self.running = False
                return
        
        while self.running:
            try:
                # Get text from queue with timeout
                try:
                    text, src_lang, timestamp = self.translation_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Skip old items if we've cleared the queue
                if timestamp < self._skip_older_than:
                    continue
                
                # Skip duplicate consecutive texts
                if text == self._last_processed_text:
                    continue
                self._last_processed_text = text
                
                # NOTE: We no longer skip translation when src_lang == tgt_lang
                # because the source language might be auto-detected/guessed incorrectly
                # The API will handle same-language gracefully, and failed translations
                # are detected in on_translation_ready()
                
                # Translate using appropriate method - prioritize SPEED
                import time
                start_time = time.time()
                
                if self.use_online:
                    # Use Reverie API - 2 second timeout for longer texts
                    translated = translate_reverie(text, src_lang, self.tgt_lang, timeout=2.0)
                else:
                    # Use IndicTrans2 - offline
                    if not self._offline_models_loaded:
                        if not self._load_offline_models():
                            self.translation_ready.emit(text, text)
                            continue
                    translated = translate_text(text, src_lang, self.tgt_lang, self.device)
                
                elapsed = time.time() - start_time
                print(f"[Trans] {src_lang}→{self.tgt_lang} in {elapsed:.2f}s: {text[:30]}... → {translated[:30]}...")
                
                self.translation_ready.emit(text, translated)
                
            except Exception as e:
                print(f"[Translation Worker] Error: {e}")
                import traceback
                traceback.print_exc()
                self.error_signal.emit(str(e))
    
    def stop(self):
        """Stop the worker"""
        self.running = False
        # Clear queue
        while not self.translation_queue.empty():
            try:
                self.translation_queue.get_nowait()
            except:
                break


# Global flag for preloaded models
_models_preloaded = False

def preload_translation_models(device="cpu"):
    """Preload translation models in background (call early to warm up)"""
    global _models_preloaded
    if _models_preloaded:
        return True
    
    success = load_indictrans_models(device)
    if success:
        _models_preloaded = True
    return success

def are_models_loaded():
    """Check if translation models are already loaded"""
    global _en_indic_model, _indic_en_model
    return _en_indic_model is not None or _indic_en_model is not None
