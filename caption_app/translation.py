"""
Translation module for the Caption App
Supports IndicTrans2 for offline translation of Indian languages
"""

import queue
from PyQt5.QtCore import QThread, pyqtSignal

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
    """Background worker for translation to avoid blocking UI"""
    translation_ready = pyqtSignal(str, str)  # original, translated
    model_loaded = pyqtSignal(str)  # model name
    error_signal = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    loading_started = pyqtSignal()  # Signal when model loading begins
    
    def __init__(self, tgt_lang="en", device="cpu", preload_only=False):
        super().__init__()
        self.tgt_lang = tgt_lang
        self.device = device
        self.running = False
        self.models_loaded = False
        self.preload_only = preload_only  # If True, just load models and stay ready
        self.translation_queue = queue.Queue()
        self._skip_older_than = 0  # Timestamp to skip old queued items
        self._last_processed_text = ""  # Track last processed to avoid duplicates
        
    def set_target_language(self, tgt_lang):
        """Update target language"""
        self.tgt_lang = tgt_lang
        print(f"[Translation Worker] Target language set to: {tgt_lang}")
        
    def add_text(self, text, src_lang):
        """Add text to translation queue with source language"""
        if self.running and self.models_loaded and text and text.strip():
            import time
            self.translation_queue.put((text.strip(), src_lang, time.time()))
    
    def clear_queue(self):
        """Clear pending translations (called when newer text arrives)"""
        import time
        self._skip_older_than = time.time()
        
    def is_ready(self):
        """Check if models are loaded and worker is ready"""
        return self.running and self.models_loaded
    
    def run(self):
        """Main translation loop"""
        self.running = True
        
        # Load models first
        self.loading_started.emit()
        self.status_changed.emit("Loading translation models...")
        print("[Translation Worker] Starting model loading...")
        
        success = load_indictrans_models(self.device)
        if success:
            self.models_loaded = True
            self.model_loaded.emit("IndicTrans2")
            self.status_changed.emit("Translation ready")
            print("[Translation Worker] Models loaded successfully")
        else:
            self.error_signal.emit("Failed to load translation models")
            self.running = False
            return
        
        # If preload_only mode, just keep running but don't process queue
        # (queue will be processed when preload_only is set to False later)
        
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
                
                # Skip translation if same language
                if src_lang == self.tgt_lang:
                    self.translation_ready.emit(text, text)
                    continue
                
                # Translate from detected source language to target
                print(f"[Translation Worker] Translating: {src_lang} → {self.tgt_lang}")
                translated = translate_text(text, src_lang, self.tgt_lang, self.device)
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
