"""
Caption App Package
A desktop captioning application with online and offline transcription support.

Modules:
    - constants: Global constants and Whisper model pre-loading
    - config: Configuration loading from config.json
    - audio: Audio capture (mic, system audio, mixed)
    - stt_workers: Speech-to-text processing (online and offline)
    - translation: Translation support with IndicTrans2
    - dialogs: UI dialogs (settings)
    - main_window: Main overlay window
"""

from .constants import LANGUAGES, TRANSLATION_LANGUAGES, WHISPER_AVAILABLE, VAD_AVAILABLE
from .config import load_config, config

__version__ = "1.1.0"
__all__ = [
    'LANGUAGES',
    'TRANSLATION_LANGUAGES',
    'WHISPER_AVAILABLE', 
    'VAD_AVAILABLE', 
    'load_config', 
    'config'
]
