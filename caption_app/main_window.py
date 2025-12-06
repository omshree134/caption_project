"""
Main overlay window for displaying captions
"""

import sys
import os
import socket
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSlider, QFrame,
    QTextEdit, QApplication, QCheckBox, QMessageBox,
    QDialog, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtGui import QCursor, QColor, QFontMetrics
from PyQt5.QtWidgets import QGraphicsOpacityEffect

import sounddevice as sd

from .constants import LANGUAGES, TRANSLATION_LANGUAGES, WHISPER_AVAILABLE
from .config import config
from .audio import AudioCapture
from .stt_workers import STTWorker, WhisperOfflineWorker, LanguageDetector
from .dialogs import CaptionSettingsDialog
from .translation import (
    INDICTRANS_AVAILABLE, translate_text, load_indictrans_models,
    TranslationWorker
)


class CaptionOverlay(QMainWindow):
    """Main overlay window for displaying captions"""
    
    def __init__(self):
        super().__init__()
        self.audio_capture = None
        self.stt_worker = None
        self.whisper_worker = None  # Offline Whisper worker
        self.language_detector = None  # Auto language detection
        self.auto_language_mode = False  # Whether auto language detection is active
        self.current_detected_lang = None  # Currently detected language
        self.is_recording = False
        self.drag_position = None
        self.partial_text = ""  # Store partial transcription
        self.last_final_pos = 0  # Track where final text ends in caption box
        self.use_offline_mode = False  # Toggle for offline-only mode
        self.api_failed = False  # Track if API has failed (for auto-fallback)
        self.auto_switched_offline = False  # Track if we auto-switched to offline
        self.response_watchdog_timer = None  # Timer to check if we're getting responses
        self.online_retry_timer = None  # Timer to retry online mode after auto-switch
        self.last_online_response_time = 0  # Track when we last got a response from online API
        self.last_audio_sent_time = 0  # Track when we last sent audio data
        self.watchdog_timeout = 5.0  # Seconds without response before switching to offline
        
        # Caption history storage - stores ALL captions and translations
        self.caption_history = []  # List of all caption texts
        self.translation_history = []  # List of all translation texts
        
        # UI scale factor for accessibility
        self.ui_scale = 1.0  # 1.0 = 100%, can go from 0.8 to 1.5
        
        # Resize handling
        self.resize_margin = 10  # Pixels from edge to trigger resize
        self.resizing = False
        self.resize_edge = None  # Which edge/corner is being resized
        self.resize_start_pos = None
        self.resize_start_geo = None
        
        # Store original geometry for single-line mode
        self.multi_line_geometry = None
        self.single_line_text = ""  # Store accumulated text for single-line mode
        self.ticker_label = None  # Label for single-line ticker mode
        
        # Translation settings
        self.translation_enabled = False
        self.translation_target_lang = "en"
        self.translation_worker = None
        self.translation_models_loaded = False
        self.show_original_text = config.get('show_original_text', False)
        
        # Dual captioning settings
        self.dual_captioning_enabled = False
        self.translation_target_lang_2 = "hi"  # Second language default
        self.translation_worker_2 = None  # Second translation worker
        
        # Caption appearance settings
        self.caption_settings = {
            'font_family': 'Segoe UI',
            'font_size': 18,
            'font_weight': 'Normal',
            'text_color': '#f8fafc',
            'text_opacity': 100,
            'bg_color': '#1e293b',
            'bg_opacity': 80,
            'border_color': '#475569',
            'border_width': 1,
            'caption_mode': 'multi',  # 'single' or 'multi'
        }
        
        # Store current translations for status updates
        self._current_translations = self._get_translations().get('en')
        
        self.init_ui()
        self.setup_audio_devices()
        
        # Apply saved interface language on startup
        saved_lang = config.get('interface_language', 'en')
        self._apply_interface_language(saved_lang)
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Live Captions")
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint | 
            Qt.FramelessWindowHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_Hover)  # Enable hover events for cursor updates
        
        # Allow resizing - set minimum and default size
        self.setMinimumSize(500, 220)
        self.resize(900, 300)
        screen = QApplication.primaryScreen().geometry()
        self.move((screen.width() - 900) // 2, screen.height() - 340)
        
        # Enable mouse tracking for resize cursor
        self.setMouseTracking(True)
        
        central = QWidget()
        central.setMouseTracking(True)
        central.setAttribute(Qt.WA_Hover)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        # Add margins around container for resize grip area
        layout.setContentsMargins(10, 10, 10, 10)
        
        self.container = QFrame()
        self.container.setObjectName("container")
        self.container.setMouseTracking(True)
        self.container.setStyleSheet("""
            #container {
                background-color: rgba(15, 23, 42, 0.95);
                border-radius: 16px;
                border: 1px solid rgba(71, 85, 105, 0.5);
            }
        """)
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(15, 10, 15, 15)
        container_layout.setSpacing(8)
        
        # Header row 1 - Main controls
        header = QHBoxLayout()
        header.setSpacing(8)
        
        # Interface language selector - FIRST element so users can change language immediately
        self.interface_lang_combo = QComboBox()
        self.interface_lang_combo.setMaxVisibleItems(5)
        self.interface_lang_combo.view().setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.interface_lang_combo.setStyleSheet("""
            QComboBox {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #4f46e5, stop:0.5 #7c3aed, stop:1 #a855f7);
                color: white;
                border: 2px solid #818cf8;
                border-radius: 8px;
                padding: 4px 10px;
                min-width: 85px;
                font-size: 12px;
                font-weight: bold;
            }
            QComboBox:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #6366f1, stop:0.5 #8b5cf6, stop:1 #c084fc);
                border: 2px solid #a5b4fc;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid white;
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: #1e1b4b;
                color: white;
                selection-background-color: #7c3aed;
                border: 2px solid #818cf8;
                border-radius: 6px;
                padding: 4px;
            }
            QComboBox QAbstractItemView::item {
                padding: 6px 10px;
                border-radius: 4px;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #4c1d95;
            }
        """)
        self.interface_lang_combo.setToolTip("🌍 Interface Language / भाषा / ভাষা")
        # Add all languages - shown in native script so users can find their language
        # Languages with full UI translations
        interface_languages = [
            ("en", "English"),
            ("hi", "हिन्दी"),
            ("bn", "বাংলা"),
            ("ta", "தமிழ்"),
            ("te", "తెలుగు"),
            ("mr", "मराठी"),
            ("gu", "ગુજરાતી"),
            ("kn", "ಕನ್ನಡ"),
            ("ml", "മലയാളം"),
            ("pa", "ਪੰਜਾਬੀ"),
            ("or", "ଓଡ଼ିଆ"),
            ("as", "অসমীয়া"),
            ("ur", "اردو"),
            ("ne", "नेपाली"),
            ("sa", "संस्कृतम्"),
            ("kok", "कोंकणी"),
            ("mai", "मैथिली"),
            ("doi", "डोगरी"),
            ("sat", "ᱥᱟᱱᱛᱟᱲᱤ"),
            ("ks", "کٲشُر"),
            ("mni", "মৈতৈলোন্"),
            ("sd", "سنڌي"),
            ("brx", "बर'"),
        ]
        for code, name in interface_languages:
            self.interface_lang_combo.addItem(name, code)
        # Load saved interface language from config
        saved_lang = config.get('interface_language', 'en')
        lang_idx = self.interface_lang_combo.findData(saved_lang)
        if lang_idx >= 0:
            self.interface_lang_combo.setCurrentIndex(lang_idx)
        self.interface_lang_combo.currentIndexChanged.connect(self.on_interface_lang_changed)
        # Interface lang combo will be added to bottom row, not header
        
        self.title_label = QLabel("🎙️ Live Captions")
        self.title_label.setStyleSheet("color: #818cf8; font-size: 14px; font-weight: bold;")
        self.title_label.setCursor(QCursor(Qt.OpenHandCursor))
        header.addWidget(self.title_label)
        header.addStretch()
        
        # Audio source selector
        self.source_label = QLabel("🔊")
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
        header.addWidget(self.source_label)
        header.addWidget(self.source_combo)
        
        # Language selector
        self.lang_label = QLabel("🌐")
        self.lang_combo = QComboBox()
        self.lang_combo.setMaxVisibleItems(5)  # Scrollable dropdown
        self.lang_combo.view().setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
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
            QScrollBar:vertical {
                background-color: #1e293b;
                width: 8px;
            }
            QScrollBar::handle:vertical {
                background-color: #475569;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        # Add Auto option first (requires Whisper)
        self.lang_combo.addItem("🔄 Auto", "auto")
        for code, name in LANGUAGES.items():
            self.lang_combo.addItem(name, code)
        # Set Auto as default if Whisper is available, otherwise English
        if WHISPER_AVAILABLE:
            self.lang_combo.setCurrentIndex(0)  # Auto
        else:
            en_index = self.lang_combo.findData("en")
            if en_index >= 0:
                self.lang_combo.setCurrentIndex(en_index)
            # Disable Auto option if Whisper not available
            self.lang_combo.model().item(0).setEnabled(False)
            self.lang_combo.setItemData(0, "Requires Whisper for auto-detection", Qt.ToolTipRole)
        header.addWidget(self.lang_label)
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
        self.offline_checkbox.toggled.connect(self.on_offline_toggled)
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
        
        # Settings button
        self.settings_btn = QPushButton("⚙")
        self.settings_btn.setFixedSize(30, 30)
        self.settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #475569; color: white;
                border: none; border-radius: 6px;
                font-size: 16px;
            }
            QPushButton:hover { background-color: #64748b; }
        """)
        self.settings_btn.setToolTip("Caption Settings")
        self.settings_btn.clicked.connect(self.open_settings)
        header.addWidget(self.settings_btn)
        
        # Close button
        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedSize(30, 30)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: #ef4444; color: white;
                border: none; border-radius: 6px;
            }
            QPushButton:hover { background-color: #f87171; }
        """)
        self.close_btn.clicked.connect(self.close)
        header.addWidget(self.close_btn)
        
        container_layout.addLayout(header)
        
        # Header row 2 - Translation controls
        translation_row = QHBoxLayout()
        translation_row.setSpacing(8)
        
        self.translate_emoji_label = QLabel("📝")
        translation_row.addWidget(self.translate_emoji_label)
        
        # Translation toggle
        self.translate_checkbox = QCheckBox("Translate")
        self.translate_checkbox.setStyleSheet("""
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
                background-color: #f59e0b;
                border-color: #f59e0b;
            }
        """)
        self.translate_checkbox.setToolTip("Translate captions to another language")
        self.translate_checkbox.setEnabled(INDICTRANS_AVAILABLE)
        if not INDICTRANS_AVAILABLE:
            self.translate_checkbox.setToolTip("Install IndicTrans2 for translation:\npip install transformers torch\npip install git+https://github.com/VarunGumma/IndicTransToolkit.git")
        self.translate_checkbox.toggled.connect(self.on_translate_toggled)
        translation_row.addWidget(self.translate_checkbox)
        
        # Translation target language selector
        self.translate_lang_combo = QComboBox()
        self.translate_lang_combo.setMaxVisibleItems(5)  # Scrollable dropdown
        self.translate_lang_combo.view().setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.translate_lang_combo.setStyleSheet("""
            QComboBox {
                background-color: #334155; color: white;
                border: 1px solid #475569; border-radius: 6px;
                padding: 3px 8px; min-width: 80px; font-size: 11px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #1e293b; color: white;
                selection-background-color: #f59e0b;
            }
            QScrollBar:vertical {
                background-color: #1e293b;
                width: 8px;
            }
            QScrollBar::handle:vertical {
                background-color: #475569;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        self.translate_lang_combo.setToolTip("Translate to this language")
        # Add placeholder option first
        self.translate_lang_combo.addItem("Select Language", "")
        # Add all translation languages
        for code, name in TRANSLATION_LANGUAGES.items():
            self.translate_lang_combo.addItem(name, code)
        # Keep "Select" as default (index 0)
        self.translate_lang_combo.setCurrentIndex(0)
        # Language selector always enabled - can change before enabling translation
        self.translate_lang_combo.currentIndexChanged.connect(self.on_translate_lang_changed)
        translation_row.addWidget(self.translate_lang_combo)
        
        # Show original text checkbox
        self.show_original_checkbox = QCheckBox("Show Original")
        self.show_original_checkbox.setStyleSheet("""
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
                background-color: #8b5cf6;
                border-color: #8b5cf6;
            }
        """)
        self.show_original_checkbox.setToolTip("Show original text alongside translation")
        self.show_original_checkbox.setChecked(config.get('show_original_text', True))
        self.show_original_checkbox.toggled.connect(self.on_show_original_toggled)
        translation_row.addWidget(self.show_original_checkbox)
        
        # Add stretch to push interface language and zoom controls to far right
        translation_row.addStretch()
        
        # Zoom controls for accessibility (at far right)
        self.zoom_out_btn = QPushButton("−")
        self.zoom_out_btn.setFixedSize(24, 24)
        self.zoom_out_btn.setStyleSheet("""
            QPushButton {
                background-color: #334155; color: white;
                border: 1px solid #475569; border-radius: 4px;
                font-size: 16px; font-weight: bold;
            }
            QPushButton:hover { background-color: #475569; }
        """)
        self.zoom_out_btn.setToolTip("Decrease size (Ctrl+-)")
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        translation_row.addWidget(self.zoom_out_btn)
        
        self.zoom_label = QLabel("100%")
        self.zoom_label.setStyleSheet("color: #94a3b8; font-size: 10px; min-width: 35px;")
        self.zoom_label.setAlignment(Qt.AlignCenter)
        translation_row.addWidget(self.zoom_label)
        
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setFixedSize(24, 24)
        self.zoom_in_btn.setStyleSheet("""
            QPushButton {
                background-color: #334155; color: white;
                border: 1px solid #475569; border-radius: 4px;
                font-size: 16px; font-weight: bold;
            }
            QPushButton:hover { background-color: #475569; }
        """)
        self.zoom_in_btn.setToolTip("Increase size (Ctrl++)")
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        translation_row.addWidget(self.zoom_in_btn)
        
        container_layout.addLayout(translation_row)
        
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
        
        # Caption display area (multi-line mode) - for ORIGINAL text
        self.caption_display = QTextEdit()
        self.caption_display.setReadOnly(True)
        self.caption_display.setPlaceholderText("Original captions will appear here...")
        container_layout.addWidget(self.caption_display)
        
        # Translation display area (multi-line mode) - for TRANSLATED text
        # Hidden by default, shown when translation + show_original is enabled
        self.translation_display = QTextEdit()
        self.translation_display.setReadOnly(True)
        self.translation_display.setPlaceholderText("🌐 Translations will appear here...")
        self.translation_display.hide()  # Hidden by default
        container_layout.addWidget(self.translation_display)
        
        # Second translation display area (multi-line mode) - for DUAL CAPTIONING
        # Hidden by default, shown when dual captioning is enabled
        self.translation_display_2 = QTextEdit()
        self.translation_display_2.setReadOnly(True)
        self.translation_display_2.setPlaceholderText("🌐 Second language translations...")
        self.translation_display_2.hide()  # Hidden by default
        container_layout.addWidget(self.translation_display_2)
        
        # Ticker-style caption (single-line mode) - hidden by default
        # Simple label that shows the latest caption text cleanly
        self.ticker_label = QLabel("")
        self.ticker_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.ticker_label.setWordWrap(False)  # Single line only
        self.ticker_label.setTextFormat(Qt.PlainText)
        self.ticker_label.hide()  # Hidden by default
        container_layout.addWidget(self.ticker_label)
        
        # Translation ticker (single-line mode) - for translations
        self.translation_ticker = QLabel("")
        self.translation_ticker.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.translation_ticker.setWordWrap(False)
        self.translation_ticker.setTextFormat(Qt.PlainText)
        self.translation_ticker.hide()
        container_layout.addWidget(self.translation_ticker)
        
        # Second translation ticker (single-line mode) - for dual captioning
        self.translation_ticker_2 = QLabel("")
        self.translation_ticker_2.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.translation_ticker_2.setWordWrap(False)
        self.translation_ticker_2.setTextFormat(Qt.PlainText)
        self.translation_ticker_2.hide()
        container_layout.addWidget(self.translation_ticker_2)
        
        self.apply_caption_settings()  # Apply initial caption styling
        
        # Bottom controls
        bottom = QHBoxLayout()
        
        self.opacity_label = QLabel("🔆 Opacity:")
        bottom.addWidget(self.opacity_label)
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(95)
        self.opacity_slider.setFixedWidth(100)
        self.opacity_slider.setStyleSheet("""
            QSlider::groove:horizontal { background: #475569; height: 4px; border-radius: 2px; }
            QSlider::handle:horizontal { background: #6366f1; width: 12px; height: 12px; margin: -4px 0; border-radius: 6px; }
        """)
        self.opacity_slider.valueChanged.connect(self.update_background_opacity)
        bottom.addWidget(self.opacity_slider)
        bottom.addStretch()
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setStyleSheet("""
            QPushButton { background-color: #475569; color: white; border: none; border-radius: 4px; padding: 5px 15px; }
            QPushButton:hover { background-color: #64748b; }
        """)
        self.clear_btn.clicked.connect(self.clear_all_captions)
        bottom.addWidget(self.clear_btn)
        
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.setStyleSheet("""
            QPushButton { background-color: #475569; color: white; border: none; border-radius: 4px; padding: 5px 15px; }
            QPushButton:hover { background-color: #64748b; }
        """)
        self.copy_btn.clicked.connect(self.copy_captions)
        bottom.addWidget(self.copy_btn)
        
        # Add interface language selector at the end of bottom row
        bottom.addWidget(self.interface_lang_combo)
        
        container_layout.addLayout(bottom)
        layout.addWidget(self.container)
        
        self.setStyleSheet("QLabel { color: #94a3b8; font-size: 12px; }")
    
    def update_background_opacity(self, value):
        """Update UI opacity - affects everything EXCEPT caption display and the opacity slider itself"""
        opacity = value / 100.0
        
        # Update container background
        self.container.setStyleSheet(f"""
            #container {{
                background-color: rgba(15, 23, 42, {opacity});
                border-radius: 16px;
                border: 1px solid rgba(71, 85, 105, {opacity * 0.5});
            }}
        """)
        
        # Caption display keeps its own settings (NOT affected by overlay opacity)
        
        # Apply opacity effect to UI elements (keeps layout stable, just fades them)
        ui_elements = [
            self.title_label, self.source_label, self.lang_label,
            self.source_combo, self.lang_combo, self.offline_checkbox,
            self.start_btn, self.settings_btn, self.close_btn,
            self.audio_level_bar, self.status_label, self.opacity_label,
            self.clear_btn, self.copy_btn,
            # Translation row elements
            self.translate_emoji_label, self.translate_checkbox, self.translate_lang_combo, self.show_original_checkbox,
            # Interface language and zoom elements
            self.interface_lang_combo,
            self.zoom_out_btn, self.zoom_label, self.zoom_in_btn
        ]
        
        for widget in ui_elements:
            effect = widget.graphicsEffect()
            if effect is None:
                effect = QGraphicsOpacityEffect(widget)
                widget.setGraphicsEffect(effect)
            effect.setOpacity(opacity)
        
        # Disable interaction when fully transparent
        interactive = value > 5
        self.source_combo.setEnabled(interactive and not self.is_recording)
        self.lang_combo.setEnabled(interactive and not self.is_recording)
        self.offline_checkbox.setEnabled(interactive)
        self.start_btn.setEnabled(interactive)
        self.settings_btn.setEnabled(interactive)
        self.close_btn.setEnabled(interactive)
        self.clear_btn.setEnabled(interactive)
        self.copy_btn.setEnabled(interactive)
    
    def open_settings(self):
        """Open the caption settings dialog"""
        # Include dual captioning settings in the dialog
        settings_with_dual = self.caption_settings.copy()
        settings_with_dual['dual_captioning_enabled'] = self.dual_captioning_enabled
        settings_with_dual['translation_target_lang_2'] = self.translation_target_lang_2
        
        dialog = CaptionSettingsDialog(self, settings_with_dual)
        if dialog.exec_() == QDialog.Accepted:
            new_settings = dialog.get_settings()
            
            # Extract dual captioning settings
            dual_enabled = new_settings.pop('dual_captioning_enabled', False)
            target_lang_2 = new_settings.pop('translation_target_lang_2', 'hi')
            
            # Apply dual captioning changes
            if dual_enabled != self.dual_captioning_enabled:
                self.dual_captioning_enabled = dual_enabled
                if dual_enabled and self.translation_enabled:
                    self._start_translation_worker_2(target_lang_2)
                elif not dual_enabled:
                    self._stop_translation_worker_2()
            
            # Update second language if changed
            if target_lang_2 != self.translation_target_lang_2:
                self.translation_target_lang_2 = target_lang_2
                if self.translation_worker_2 is not None and self.translation_worker_2.isRunning():
                    self.translation_worker_2.set_target_language(target_lang_2)
            
            # Apply caption settings
            self.caption_settings = new_settings
            self.apply_caption_settings()
            
            # Update display visibility for dual mode
            self._update_dual_display_visibility()
    
    def apply_caption_settings(self):
        """Apply caption settings to the caption display"""
        s = self.caption_settings
        
        # Parse colors
        text_color = QColor(s['text_color'])
        bg_color = QColor(s['bg_color'])
        border_color = QColor(s['border_color'])
        
        text_opacity = s['text_opacity'] / 100.0
        bg_opacity = s['bg_opacity'] / 100.0
        
        font_weight = "bold" if s['font_weight'] == "Bold" else "normal"
        
        # Base style for original captions
        self.caption_display.setStyleSheet(f"""
            QTextEdit {{
                background-color: rgba({bg_color.red()}, {bg_color.green()}, {bg_color.blue()}, {bg_opacity});
                color: rgba({text_color.red()}, {text_color.green()}, {text_color.blue()}, {text_opacity});
                border: {s['border_width']}px solid {border_color.name()};
                border-radius: 8px;
                padding: 10px;
                font-family: '{s['font_family']}';
                font-size: {s['font_size']}px;
                font-weight: {font_weight};
            }}
        """)
        
        # Style for translation display - slightly different color scheme
        trans_text_color = QColor("#60a5fa")  # Blue tint for translations
        self.translation_display.setStyleSheet(f"""
            QTextEdit {{
                background-color: rgba({bg_color.red()}, {bg_color.green()}, {bg_color.blue()}, {bg_opacity * 0.8});
                color: rgba({trans_text_color.red()}, {trans_text_color.green()}, {trans_text_color.blue()}, {text_opacity});
                border: {s['border_width']}px solid #3b82f6;
                border-radius: 8px;
                padding: 10px;
                font-family: '{s['font_family']}';
                font-size: {s['font_size']}px;
                font-weight: {font_weight};
            }}
        """)
        
        # Style for second translation display (dual captioning) - green tint
        trans_text_color_2 = QColor("#34d399")  # Green tint for second language
        self.translation_display_2.setStyleSheet(f"""
            QTextEdit {{
                background-color: rgba({bg_color.red()}, {bg_color.green()}, {bg_color.blue()}, {bg_opacity * 0.8});
                color: rgba({trans_text_color_2.red()}, {trans_text_color_2.green()}, {trans_text_color_2.blue()}, {text_opacity});
                border: {s['border_width']}px solid #10b981;
                border-radius: 8px;
                padding: 10px;
                font-family: '{s['font_family']}';
                font-size: {s['font_size']}px;
                font-weight: {font_weight};
            }}
        """)
        
        # Handle single-line vs multi-line mode
        is_single_line = s.get('caption_mode', 'multi') == 'single'
        if is_single_line:
            # Save current geometry before switching to single-line
            if self.multi_line_geometry is None:
                self.multi_line_geometry = self.geometry()
            
            # Hide multi-line displays, show ticker
            self.caption_display.hide()
            self.translation_display.hide()
            self.ticker_label.show()
            
            # Style the ticker label with elide for overflow
            self.ticker_label.setStyleSheet(f"""
                QLabel {{
                    background-color: rgba({bg_color.red()}, {bg_color.green()}, {bg_color.blue()}, {bg_opacity});
                    color: rgba({text_color.red()}, {text_color.green()}, {text_color.blue()}, {text_opacity});
                    border: {s['border_width']}px solid {border_color.name()};
                    border-radius: 8px;
                    padding: 10px 15px;
                    font-family: '{s['font_family']}';
                    font-size: {s['font_size']}px;
                    font-weight: {font_weight};
                }}
            """)
            
            # Style translation ticker
            self.translation_ticker.setStyleSheet(f"""
                QLabel {{
                    background-color: rgba({bg_color.red()}, {bg_color.green()}, {bg_color.blue()}, {bg_opacity * 0.8});
                    color: rgba({trans_text_color.red()}, {trans_text_color.green()}, {trans_text_color.blue()}, {text_opacity});
                    border: {s['border_width']}px solid #3b82f6;
                    border-radius: 8px;
                    padding: 10px 15px;
                    font-family: '{s['font_family']}';
                    font-size: {s['font_size']}px;
                    font-weight: {font_weight};
                }}
            """)
            
            # Style second translation ticker (dual captioning)
            self.translation_ticker_2.setStyleSheet(f"""
                QLabel {{
                    background-color: rgba({bg_color.red()}, {bg_color.green()}, {bg_color.blue()}, {bg_opacity * 0.8});
                    color: rgba({trans_text_color_2.red()}, {trans_text_color_2.green()}, {trans_text_color_2.blue()}, {text_opacity});
                    border: {s['border_width']}px solid #10b981;
                    border-radius: 8px;
                    padding: 10px 15px;
                    font-family: '{s['font_family']}';
                    font-size: {s['font_size']}px;
                    font-weight: {font_weight};
                }}
            """)
            
            # Calculate compact height for single line
            line_height = s['font_size'] + 35  # font size + padding + border
            self.ticker_label.setFixedHeight(line_height)
            self.translation_ticker.setFixedHeight(line_height)
            self.translation_ticker_2.setFixedHeight(line_height)
            
            # Resize entire overlay to compact single-line height
            compact_height = line_height + 160  # Extra space for two header rows
            self.setMinimumSize(500, compact_height)
            self.setMaximumHeight(compact_height)
            self.resize(self.width(), compact_height)
            
            # Clear ticker buffer and show last text if any
            self.ticker_buffer = []
            self.partial_text = ""
            if self.single_line_text:
                self._update_ticker_display(self.single_line_text)
        else:
            # Multi-line mode - show text edit, hide ticker
            self.ticker_label.hide()
            self.translation_ticker.hide()
            self.translation_ticker_2.hide()
            self.caption_display.show()
            # translation_display visibility controlled by toggle_translation_display()
            
            # Reset height constraints
            self.caption_display.setMaximumHeight(16777215)  # Qt's default max
            self.caption_display.setMinimumHeight(60)
            self.caption_display.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            
            self.translation_display.setMaximumHeight(16777215)
            self.translation_display.setMinimumHeight(80)  # Larger for translation
            self.translation_display.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            
            self.translation_display_2.setMaximumHeight(16777215)
            self.translation_display_2.setMinimumHeight(80)
            self.translation_display_2.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            
            # Restore original window geometry
            self.setMinimumSize(500, 220)
            self.setMaximumHeight(16777215)
            if self.multi_line_geometry is not None:
                self.setGeometry(self.multi_line_geometry)
                self.multi_line_geometry = None
        
        # Clear partial text when mode changes
        self.partial_text = ""
        
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
            self.source_combo.addItem(f"🔊 System Audio", "speaker")
            self.source_combo.addItem(f"🎤+🔊 Mic + System Audio", "both")
            
            # Set default based on availability
            if self.loopback_device is not None:
                self.source_combo.setCurrentIndex(1)  # System Audio
                self.status_label.setText(f"✓ System Audio Ready")
            else:
                self.source_combo.setCurrentIndex(0)  # mic only
                self.status_label.setText("⚠️ System audio capture not available")
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
        # Debounce: prevent rapid toggling which causes crashes
        if hasattr(self, '_toggle_in_progress') and self._toggle_in_progress:
            print("[DEBUG] Toggle already in progress, ignoring")
            return
        
        self._toggle_in_progress = True
        try:
            if self.is_recording:
                self.stop_recording()
            else:
                self.start_recording()
        finally:
            # Allow next toggle after a short delay
            QTimer.singleShot(300, self._reset_toggle_lock)
    
    def _reset_toggle_lock(self):
        """Reset the toggle lock after delay"""
        self._toggle_in_progress = False
            
    def start_recording(self):
        # Double-check we're not already recording
        if self.is_recording:
            print("[DEBUG] Already recording, ignoring start_recording call")
            return
        
        # Check if cleanup is still in progress
        if hasattr(self, '_recording_lock') and self._recording_lock:
            print("[DEBUG] Cleanup in progress, ignoring start_recording call")
            return
            
        # Set the lock during startup
        self._recording_lock = True
            
        # Make sure any previous workers are cleaned up SYNCHRONOUSLY
        self._cleanup_workers_sync()
        
        mode = self.source_combo.currentData()
        lang = self.lang_combo.currentData()
        use_offline = self.offline_checkbox.isChecked()
        
        # Check if auto language mode
        self.auto_language_mode = (lang == "auto")
        if self.auto_language_mode:
            if not WHISPER_AVAILABLE:
                QMessageBox.warning(self, "Auto Language Unavailable", 
                    "Auto language detection requires Whisper.\n\npip install faster-whisper")
                self._recording_lock = False  # Release lock on failure
                return
            # Start with English as default, will be updated by detector
            lang = "en"
            # Set to None so first detection always triggers an update
            self.current_detected_lang = None
        
        print(f"[DEBUG] Starting recording: mode={mode}, lang={lang}, offline={use_offline}, auto_lang={self.auto_language_mode}")
        print(f"[DEBUG] WHISPER_AVAILABLE={WHISPER_AVAILABLE}")
        
        # Check API credentials for online mode
        if not use_offline and (not config.get('api_key') or not config.get('app_id')):
            QMessageBox.warning(self, "Error", "API credentials not configured in config.json\nTry offline mode instead.")
            self._recording_lock = False  # Release lock on failure
            return
        
        # Check if offline mode is available
        if use_offline and not WHISPER_AVAILABLE:
            QMessageBox.warning(self, "Offline Mode Unavailable", 
                "Install faster-whisper for offline mode:\n\npip install faster-whisper")
            self._recording_lock = False  # Release lock on failure
            return
            
        # Check if loopback is available for speaker/both modes
        if mode in ["speaker", "both"] and self.loopback_device is None:
            QMessageBox.warning(
                self, "System Audio Not Available",
                "System audio capture requires either:\n\n"
                "• WASAPI loopback (automatic)\n"
                "• Or enable 'Stereo Mix' in Windows Sound settings:\n"
                "  1. Right-click speaker icon → Sounds\n"
                "  2. Recording tab → Show Disabled Devices\n"
                "  3. Enable 'Stereo Mix'"
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
            whisper_lang = lang  # Use selected language directly
            
            print(f"[DEBUG] Creating WhisperOfflineWorker with model=tiny, lang={whisper_lang}")
            
            # Pre-load model in main thread to avoid threading issues with CTranslate2
            self.status_label.setText("Loading Whisper model...")
            QApplication.processEvents()  # Update UI immediately
            
            model = WhisperOfflineWorker.preload_model(model_size="tiny", device="cpu")
            if model is None:
                QMessageBox.warning(self, "Model Load Failed", 
                    "Failed to load Whisper model. Check console for errors.")
                self._recording_lock = False  # Release lock on failure
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
            
            # Start response watchdog for online mode
            self._start_response_watchdog()
        
        # Start language detector if auto mode is enabled
        if self.auto_language_mode and WHISPER_AVAILABLE:
            from .constants import get_whisper_model
            model = get_whisper_model()
            self.language_detector = LanguageDetector(model=model)
            self.language_detector.language_detected.connect(self.on_language_detected)
            self.language_detector.status_changed.connect(lambda s: print(f"[LangDetect] {s}"))
            self.language_detector.start()
            self.status_label.setText("🔄 Detecting language...")
            print("[DEBUG] Language detector started")
        
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
        # Note: offline_checkbox stays enabled for live switching
        
        # Release the recording lock - startup complete
        self._recording_lock = False
        print("[DEBUG] Recording started, lock released")
        
    def on_offline_toggled(self, checked):
        """Handle offline checkbox toggle - allows live switching during recording"""
        # Always update translation mode to match STT mode
        self._update_translation_mode()
        
        if not self.is_recording:
            # Not recording, just update the preference
            return
        
        # Live switch during recording
        if checked and not self.use_offline_mode:
            # Switch to offline mode
            self._switch_to_offline_mode_live()
        elif not checked and self.use_offline_mode:
            # Switch to online mode
            self._switch_to_online_mode_live()
    
    def _switch_to_offline_mode_live(self):
        """Switch to offline mode while recording is active"""
        if not WHISPER_AVAILABLE:
            self.offline_checkbox.setChecked(False)
            return
        
        print("[Live Switch] Switching to offline mode...")
        self.status_label.setText("🔄 Switching to offline...")
        QApplication.processEvents()
        
        # Stop online STT worker
        if self.stt_worker:
            self.stt_worker.stop()
            if not self.stt_worker.wait(300):
                self.stt_worker.terminate()
            self.stt_worker = None
        
        # Stop response watchdog since user explicitly chose offline
        self._stop_response_watchdog()
        self.auto_switched_offline = False
        
        # Pre-load Whisper model
        model = WhisperOfflineWorker.preload_model(model_size="tiny", device="cpu")
        if model is None:
            self.status_label.setText("❌ Failed to load offline model")
            self.offline_checkbox.setChecked(False)
            return
        
        # Get current language - use detected lang if in auto mode
        if self.auto_language_mode and self.current_detected_lang:
            lang = self.current_detected_lang
        else:
            lang = self.lang_combo.currentData()
            if lang == "auto":
                lang = "en"  # Fallback
        
        # Start Whisper worker
        self.whisper_worker = WhisperOfflineWorker(
            model_size="tiny",
            language=lang,
            device="cpu",
            model=model
        )
        self.whisper_worker.transcription.connect(self.on_transcription)
        self.whisper_worker.status_changed.connect(self.on_status_changed)
        self.whisper_worker.error_signal.connect(self.on_whisper_error)
        self.whisper_worker.start()
        
        self.use_offline_mode = True
        self.api_failed = False
        print(f"[Live Switch] Now in offline mode with lang={lang}")
    
    def _switch_to_online_mode_live(self):
        """Switch to online mode while recording is active"""
        if not config.get('api_key') or not config.get('app_id'):
            self.offline_checkbox.setChecked(True)
            self.status_label.setText("⚠️ No API credentials configured")
            return
        
        print("[Live Switch] Switching to online mode...")
        self.status_label.setText("🔄 Switching to online...")
        QApplication.processEvents()
        
        # Stop Whisper worker
        if self.whisper_worker:
            self.whisper_worker.stop()
            if not self.whisper_worker.wait(500):
                self.whisper_worker.terminate()
            self.whisper_worker = None
        
        # Get current language - use detected lang if in auto mode
        if self.auto_language_mode and self.current_detected_lang:
            lang = self.current_detected_lang
        else:
            lang = self.lang_combo.currentData()
            if lang == "auto":
                lang = "en"  # Fallback
        
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
        
        # Start response watchdog for online mode
        self._start_response_watchdog()
        
        self.use_offline_mode = False
        self.auto_switched_offline = False
        print(f"[Live Switch] Now in online mode with lang={lang}")
    
    def on_translate_toggled(self, checked):
        """Handle translation checkbox toggle"""
        if checked:
            # Check if a valid language is selected
            target_lang = self.translate_lang_combo.currentData()
            if not target_lang:  # Empty string means "Select Language" placeholder
                self.status_label.setText("⚠️ Please select a language first")
                self.translate_checkbox.setChecked(False)
                return
        
        self.translation_enabled = checked
        # Language combo stays enabled always - can change language anytime
        self.show_original_checkbox.setEnabled(checked)
        
        if checked:
            # Initialize translation worker if not already running
            if self.translation_worker is None or not self.translation_worker.isRunning():
                target_lang = self.translate_lang_combo.currentData()
                self._start_translation_worker(target_lang)
            elif self.translation_worker and not self.translation_worker.is_ready():
                # Worker is running but models not loaded yet - just wait
                self.status_label.setText("🔄 Translation models loading...")
            
            # Also start second worker if dual captioning is enabled (from settings)
            if self.dual_captioning_enabled:
                if self.translation_worker_2 is None or not self.translation_worker_2.isRunning():
                    self._start_translation_worker_2(self.translation_target_lang_2)
        else:
            # Don't stop the worker - just disable adding to queue
            # This way, if user toggles on again, models are already loaded
            pass
        
        # Update dual display visibility
        self._update_dual_display_visibility()
    
    def on_show_original_toggled(self, checked):
        """Handle show original text checkbox toggle - controls dual display mode"""
        # Update config
        config['show_original_text'] = checked
        
        # Update display visibility for dual-caption mode
        self._update_dual_display_visibility()
    
    def on_interface_lang_changed(self, index):
        """Handle interface language change"""
        lang_code = self.interface_lang_combo.currentData()
        # Store the interface language preference and save to file
        config['interface_language'] = lang_code
        self._save_config()
        
        # Apply the language
        self._apply_interface_language(lang_code)
    
    def _get_translations(self):
        """Get all interface translations"""
        return {
            "en": {
                "title": "🎙️ Live Captions",
                "translate": "Translate",
                "show_original": "Show Original",
                "offline": "Offline",
                "start": "▶ Start",
                "stop": "⏹ Stop",
                "ready": "Ready",
                "recording": "Recording...",
                "select_lang": "Select Language",
                "connected": "🟢 Connected",
                "system_audio": "🔊 System Audio",
                "mic_system": "🎤+🔊 Mic + System",
                "auto": "🔄 Auto",
                "opacity": "🔆 Opacity:",
                "clear": "Clear",
                "copy": "Copy",
            },
            "hi": {
                "title": "🎙️ लाइव कैप्शन",
                "translate": "अनुवाद",
                "show_original": "मूल दिखाएं",
                "offline": "ऑफ़लाइन",
                "start": "▶ शुरू",
                "stop": "⏹ रोकें",
                "ready": "तैयार",
                "recording": "रिकॉर्डिंग...",
                "select_lang": "भाषा चुनें",
                "connected": "🟢 जुड़ा हुआ",
                "system_audio": "🔊 सिस्टम ऑडियो",
                "mic_system": "🎤+🔊 माइक + सिस्टम",
                "auto": "🔄 स्वचालित",
                "opacity": "🔆 पारदर्शिता:",
                "clear": "साफ़",
                "copy": "कॉपी",
            },
            "bn": {
                "title": "🎙️ লাইভ ক্যাপশন",
                "translate": "অনুবাদ",
                "show_original": "মূল দেখান",
                "offline": "অফলাইন",
                "start": "▶ শুরু",
                "stop": "⏹ থামুন",
                "ready": "প্রস্তুত",
                "recording": "রেকর্ডিং...",
                "select_lang": "ভাষা নির্বাচন",
                "connected": "🟢 সংযুক্ত",
                "system_audio": "🔊 সিস্টেম অডিও",
                "mic_system": "🎤+🔊 মাইক + সিস্টেম",
                "auto": "🔄 স্বয়ংক্রিয়",
                "opacity": "🔆 স্বচ্ছতা:",
                "clear": "মুছুন",
                "copy": "কপি",
            },
            "ta": {
                "title": "🎙️ நேரடி வசனங்கள்",
                "translate": "மொழிபெயர்",
                "show_original": "மூலம் காட்டு",
                "offline": "ஆஃப்லைன்",
                "start": "▶ தொடங்கு",
                "stop": "⏹ நிறுத்து",
                "ready": "தயார்",
                "recording": "பதிவு...",
                "select_lang": "மொழி தேர்வு",
                "connected": "🟢 இணைக்கப்பட்டது",
                "system_audio": "🔊 சிஸ்டம் ஆடியோ",
                "mic_system": "🎤+🔊 மைக் + சிஸ்டம்",
                "auto": "🔄 தானியங்கி",
                "opacity": "🔆 ஒளிபுகா:",
                "clear": "அழி",
                "copy": "நகல்",
            },
            "te": {
                "title": "🎙️ లైవ్ క్యాప్షన్స్",
                "translate": "అనువాదం",
                "show_original": "అసలు చూపు",
                "offline": "ఆఫ్‌లైన్",
                "start": "▶ ప్రారంభం",
                "stop": "⏹ ఆపు",
                "ready": "సిద్ధం",
                "recording": "రికార్డింగ్...",
                "select_lang": "భాష ఎంచుకో",
                "connected": "🟢 కనెక్ట్ అయింది",
                "system_audio": "🔊 సిస్టమ్ ఆడియో",
                "mic_system": "🎤+🔊 మైక్ + సిస్టమ్",
                "auto": "🔄 ఆటో",
                "opacity": "🔆 అపారదర్శకత:",
                "clear": "క్లియర్",
                "copy": "కాపీ",
            },
            "mr": {
                "title": "🎙️ लाइव्ह कॅप्शन",
                "translate": "भाषांतर",
                "show_original": "मूळ दाखवा",
                "offline": "ऑफलाइन",
                "start": "▶ सुरू",
                "stop": "⏹ थांबा",
                "ready": "तयार",
                "recording": "रेकॉर्डिंग...",
                "select_lang": "भाषा निवडा",
                "connected": "🟢 जोडलेले",
                "system_audio": "🔊 सिस्टम ऑडिओ",
                "mic_system": "🎤+🔊 माइक + सिस्टम",
                "auto": "🔄 स्वयंचलित",
                "opacity": "🔆 अपारदर्शकता:",
                "clear": "साफ करा",
                "copy": "कॉपी",
            },
            "gu": {
                "title": "🎙️ લાઇવ કેપ્શન",
                "translate": "અનુવાદ",
                "show_original": "મૂળ બતાવો",
                "offline": "ઓફલાઇન",
                "start": "▶ શરૂ",
                "stop": "⏹ બંધ",
                "ready": "તૈયાર",
                "recording": "રેકોર્ડિંગ...",
                "select_lang": "ભાષા પસંદ કરો",
                "connected": "🟢 જોડાયેલ",
                "system_audio": "🔊 સિસ્ટમ ઑડિયો",
                "mic_system": "🎤+🔊 માઇક + સિસ્ટમ",
                "auto": "🔄 સ્વચાલિત",
                "opacity": "🔆 અપારદર્શિતા:",
                "clear": "સાફ",
                "copy": "કૉપિ",
            },
            "kn": {
                "title": "🎙️ ಲೈವ್ ಕ್ಯಾಪ್ಷನ್",
                "translate": "ಅನುವಾದ",
                "show_original": "ಮೂಲ ತೋರಿಸಿ",
                "offline": "ಆಫ್‌ಲೈನ್",
                "start": "▶ ಪ್ರಾರಂಭ",
                "stop": "⏹ ನಿಲ್ಲಿಸಿ",
                "ready": "ಸಿದ್ಧ",
                "recording": "ರೆಕಾರ್ಡಿಂಗ್...",
                "select_lang": "ಭಾಷೆ ಆಯ್ಕೆ",
                "connected": "🟢 ಸಂಪರ್ಕಿತ",
                "system_audio": "🔊 ಸಿಸ್ಟಮ್ ಆಡಿಯೋ",
                "mic_system": "🎤+🔊 ಮೈಕ್ + ಸಿಸ್ಟಮ್",
                "auto": "🔄 ಸ್ವಯಂ",
                "opacity": "🔆 ಅಪಾರದರ್ಶಕತೆ:",
                "clear": "ತೆರವು",
                "copy": "ನಕಲು",
            },
            "ml": {
                "title": "🎙️ ലൈവ് ക്യാപ്ഷൻ",
                "translate": "വിവർത്തനം",
                "show_original": "മൂലം കാണുക",
                "offline": "ഓഫ്‌ലൈൻ",
                "start": "▶ തുടങ്ങുക",
                "stop": "⏹ നിർത്തുക",
                "ready": "തയ്യാറായി",
                "recording": "റെക്കോർഡിംഗ്...",
                "select_lang": "ഭാഷ തിരഞ്ഞെടുക്കുക",
                "connected": "🟢 കണക്റ്റ് ആയി",
                "system_audio": "🔊 സിസ്റ്റം ഓഡിയോ",
                "mic_system": "🎤+🔊 മൈക്ക് + സിസ്റ്റം",
                "auto": "🔄 ഓട്ടോ",
                "opacity": "🔆 അതാര്യത:",
                "clear": "മായ്ക്കുക",
                "copy": "പകർത്തുക",
            },
            "pa": {
                "title": "🎙️ ਲਾਈਵ ਕੈਪਸ਼ਨ",
                "translate": "ਅਨੁਵਾਦ",
                "show_original": "ਮੂਲ ਦਿਖਾਓ",
                "offline": "ਔਫ਼ਲਾਈਨ",
                "start": "▶ ਸ਼ੁਰੂ",
                "stop": "⏹ ਰੁਕੋ",
                "ready": "ਤਿਆਰ",
                "recording": "ਰਿਕਾਰਡਿੰਗ...",
                "select_lang": "ਭਾਸ਼ਾ ਚੁਣੋ",
                "connected": "🟢 ਜੁੜਿਆ",
                "system_audio": "🔊 ਸਿਸਟਮ ਆਡੀਓ",
                "mic_system": "🎤+🔊 ਮਾਈਕ + ਸਿਸਟਮ",
                "auto": "🔄 ਆਟੋ",
                "opacity": "🔆 ਧੁੰਦਲਾਪਨ:",
                "clear": "ਸਾਫ਼",
                "copy": "ਕਾਪੀ",
            },
            "or": {
                "title": "🎙️ ଲାଇଭ୍ କ୍ୟାପ୍ସନ୍",
                "translate": "ଅନୁବାଦ",
                "show_original": "ମୂଳ ଦେଖାଅ",
                "offline": "ଅଫଲାଇନ୍",
                "start": "▶ ଆରମ୍ଭ",
                "stop": "⏹ ବନ୍ଦ",
                "ready": "ପ୍ରସ୍ତୁତ",
                "recording": "ରେକର୍ଡିଂ...",
                "select_lang": "ଭାଷା ବାଛନ୍ତୁ",
                "connected": "🟢 ସଂଯୁକ୍ତ",
                "system_audio": "🔊 ସିଷ୍ଟମ୍ ଅଡିଓ",
                "mic_system": "🎤+🔊 ମାଇକ୍ + ସିଷ୍ଟମ୍",
                "auto": "🔄 ସ୍ୱୟଂଚାଳିତ",
                "opacity": "🔆 ଅସ୍ୱଚ୍ଛତା:",
                "clear": "ସଫା",
                "copy": "କପି",
            },
            "ur": {
                "title": "🎙️ لائیو کیپشن",
                "translate": "ترجمہ",
                "show_original": "اصل دکھائیں",
                "offline": "آف لائن",
                "start": "▶ شروع",
                "stop": "⏹ رکیں",
                "ready": "تیار",
                "recording": "ریکارڈنگ...",
                "select_lang": "زبان منتخب کریں",
                "connected": "🟢 منسلک",
                "system_audio": "🔊 سسٹم آڈیو",
                "mic_system": "🎤+🔊 مائیک + سسٹم",
                "auto": "🔄 خودکار",
                "opacity": "🔆 دھندلاپن:",
                "clear": "صاف",
                "copy": "کاپی",
            },
            "as": {
                "title": "🎙️ লাইভ কেপচন",
                "translate": "অনুবাদ",
                "show_original": "মূল দেখুৱাওক",
                "offline": "অফলাইন",
                "start": "▶ আৰম্ভ",
                "stop": "⏹ বন্ধ",
                "ready": "সাজু",
                "recording": "ৰেকৰ্ডিং...",
                "select_lang": "ভাষা বাছক",
                "connected": "🟢 সংযুক্ত",
                "system_audio": "🔊 চিষ্টেম অডিঅ'",
                "mic_system": "🎤+🔊 মাইক + চিষ্টেম",
                "auto": "🔄 স্বয়ংক্ৰিয়",
                "opacity": "🔆 স্বচ্ছতা:",
                "clear": "মচক",
                "copy": "কপি",
            },
            "ne": {
                "title": "🎙️ लाइभ क्याप्शन",
                "translate": "अनुवाद",
                "show_original": "मूल देखाउनुहोस्",
                "offline": "अफलाइन",
                "start": "▶ सुरु",
                "stop": "⏹ रोक्नुहोस्",
                "ready": "तयार",
                "recording": "रेकर्डिङ...",
                "select_lang": "भाषा छान्नुहोस्",
                "connected": "🟢 जोडिएको",
                "system_audio": "🔊 सिस्टम अडियो",
                "mic_system": "🎤+🔊 माइक + सिस्टम",
                "auto": "🔄 स्वचालित",
                "opacity": "🔆 अपारदर्शिता:",
                "clear": "खाली",
                "copy": "कपी",
            },
            "sa": {
                "title": "🎙️ प्रत्यक्ष शीर्षकाः",
                "translate": "अनुवादः",
                "show_original": "मूलं दर्शयतु",
                "offline": "अपरेखा",
                "start": "▶ आरभताम्",
                "stop": "⏹ स्थगयतु",
                "ready": "सज्जम्",
                "recording": "अभिलेखनम्...",
                "select_lang": "भाषां चिनुत",
                "connected": "🟢 संयुक्तम्",
                "system_audio": "🔊 तन्त्र ध्वनिः",
                "mic_system": "🎤+🔊 वाक्यन्त्रं + तन्त्रम्",
                "auto": "🔄 स्वचालितम्",
                "opacity": "🔆 अपारदर्शिता:",
                "clear": "शोधयतु",
                "copy": "प्रतिलिपिः",
            },
            "kok": {
                "title": "🎙️ लायव्ह कॅप्शन",
                "translate": "भाशांतर",
                "show_original": "मूळ दाखयात",
                "offline": "ऑफलायन",
                "start": "▶ सुरू",
                "stop": "⏹ थांबयात",
                "ready": "तयार",
                "recording": "रेकॉर्डींग...",
                "select_lang": "भास वेंचात",
                "connected": "🟢 जोडलां",
                "system_audio": "🔊 सिस्टम ऑडियो",
                "mic_system": "🎤+🔊 मायक + सिस्टम",
                "auto": "🔄 आपसूक",
                "opacity": "🔆 अपारदर्शकताय:",
                "clear": "साफ",
                "copy": "कॉपी",
            },
            "mai": {
                "title": "🎙️ लाइव कैप्शन",
                "translate": "अनुवाद",
                "show_original": "मूल देखाउ",
                "offline": "ऑफलाइन",
                "start": "▶ शुरू",
                "stop": "⏹ रुकू",
                "ready": "तैयार",
                "recording": "रिकॉर्डिंग...",
                "select_lang": "भाषा चुनू",
                "connected": "🟢 जुड़ल",
                "system_audio": "🔊 सिस्टम ऑडियो",
                "mic_system": "🎤+🔊 माइक + सिस्टम",
                "auto": "🔄 स्वचालित",
                "opacity": "🔆 पारदर्शिता:",
                "clear": "साफ",
                "copy": "कॉपी",
            },
            "doi": {
                "title": "🎙️ लाइव कैप्शन",
                "translate": "अनुवाद",
                "show_original": "असली दस्सो",
                "offline": "ऑफलाइन",
                "start": "▶ शुरू करो",
                "stop": "⏹ रुको",
                "ready": "तैयार",
                "recording": "रिकॉर्डिंग...",
                "select_lang": "बोली चुणो",
                "connected": "🟢 जुड़ेआ",
                "system_audio": "🔊 सिस्टम आडियो",
                "mic_system": "🎤+🔊 माइक + सिस्टम",
                "auto": "🔄 आपूआप",
                "opacity": "🔆 धुंदलापन:",
                "clear": "साफ",
                "copy": "कॉपी",
            },
            "sat": {
                "title": "🎙️ ᱞᱟᱭᱤᱵ ᱠᱮᱯᱥᱚᱱ",
                "translate": "ᱛᱚᱨᱡᱚᱢᱟ",
                "show_original": "ᱢᱩᱞ ᱫᱮᱠᱷᱟᱣ",
                "offline": "ᱚᱯᱷᱞᱟᱭᱤᱱ",
                "start": "▶ ᱮᱦᱚᱵ",
                "stop": "⏹ ᱛᱷᱟᱢ",
                "ready": "ᱛᱮᱭᱟᱨ",
                "recording": "ᱨᱮᱠᱚᱨᱰᱤᱝ...",
                "select_lang": "ᱯᱟᱹᱨᱥᱤ ᱵᱟᱪᱷᱟᱣ",
                "connected": "🟢 ᱡᱚᱲᱟᱣ",
                "system_audio": "🔊 ᱥᱤᱥᱴᱮᱢ ᱚᱰᱤᱭᱚ",
                "mic_system": "🎤+🔊 ᱢᱟᱭᱠ + ᱥᱤᱥᱴᱮᱢ",
                "auto": "🔄 ᱚᱴᱚ",
                "opacity": "🔆 ᱚᱯᱮᱥᱤᱴᱤ:",
                "clear": "ᱥᱟᱯᱷᱟ",
                "copy": "ᱠᱚᱯᱤ",
            },
            "ks": {
                "title": "🎙️ لائیو کیپشن",
                "translate": "ترجمہٕ",
                "show_original": "اصل ہاوِن",
                "offline": "آف لائن",
                "start": "▶ شروع",
                "stop": "⏹ بَند",
                "ready": "تیٲر",
                "recording": "ریکارڈنگ...",
                "select_lang": "زبان چُنو",
                "connected": "🟢 جوٗڑمُت",
                "system_audio": "🔊 سِسٹم آڈیو",
                "mic_system": "🎤+🔊 مایک + سِسٹم",
                "auto": "🔄 آٹو",
                "opacity": "🔆 دُھندلاپَن:",
                "clear": "صاف",
                "copy": "کاپی",
            },
            "mni": {
                "title": "🎙️ লাইভ কেপশন",
                "translate": "অনুবাদ",
                "show_original": "অশেংবা উৎখৌ",
                "offline": "অফলাইন",
                "start": "▶ হৌদোক্লু",
                "stop": "⏹ থোক্লো",
                "ready": "শেম্লবা",
                "recording": "রেকোর্দিং...",
                "select_lang": "লোন খন্নবা",
                "connected": "🟢 চেকসিনবা",
                "system_audio": "🔊 সিস্তেম ওদিও",
                "mic_system": "🎤+🔊 মাইক + সিস্তেম",
                "auto": "🔄 মশা নাইনা",
                "opacity": "🔆 থিংনবা:",
                "clear": "মাংহনবা",
                "copy": "কপি",
            },
            "sd": {
                "title": "🎙️ لائيو ڪيپشن",
                "translate": "ترجمو",
                "show_original": "اصل ڏيکاريو",
                "offline": "آف لائن",
                "start": "▶ شروع",
                "stop": "⏹ بند",
                "ready": "تيار",
                "recording": "رڪارڊنگ...",
                "select_lang": "ٻولي چونڊيو",
                "connected": "🟢 ڳنڍيل",
                "system_audio": "🔊 سسٽم آڊيو",
                "mic_system": "🎤+🔊 مائڪ + سسٽم",
                "auto": "🔄 خودڪار",
                "opacity": "🔆 شفافيت:",
                "clear": "صاف",
                "copy": "ڪاپي",
            },
            "brx": {
                "title": "🎙️ लाइभ केपसन",
                "translate": "राव सोलायनाय",
                "show_original": "गिबि दिन्थि",
                "offline": "अफलाइन",
                "start": "▶ जागाय",
                "stop": "⏹ थाखाय",
                "ready": "थाखाय गोनां",
                "recording": "रेकर्ड खालाम...",
                "select_lang": "राव बासिन",
                "connected": "🟢 गोथां जानाय",
                "system_audio": "🔊 सिस्टेम अडिअ'",
                "mic_system": "🎤+🔊 माइक + सिस्टेम",
                "auto": "🔄 गसर जानाय",
                "opacity": "🔆 फोरमायनाय:",
                "clear": "खालि",
                "copy": "कपि",
            },
        }
    
    def _apply_interface_language(self, lang_code):
        """Apply interface language to all UI elements"""
        translations = self._get_translations()
        t = translations.get(lang_code, translations["en"])
        
        # Update UI text
        self.title_label.setText(t["title"])
        self.translate_checkbox.setText(t["translate"])
        self.show_original_checkbox.setText(t["show_original"])
        self.offline_checkbox.setText(t["offline"])
        
        # Update start button based on recording state
        if not self.is_recording:
            self.start_btn.setText(t["start"])
            self.status_label.setText(t["ready"])
        else:
            self.start_btn.setText(t["stop"])
        
        # Update select language placeholder
        self.translate_lang_combo.setItemText(0, t["select_lang"])
        
        # Update opacity label
        self.opacity_label.setText(t["opacity"])
        
        # Update clear and copy buttons
        self.clear_btn.setText(t["clear"])
        self.copy_btn.setText(t["copy"])
        
        # Update source combo items (System Audio, Mic + System)
        for i in range(self.source_combo.count()):
            item_data = self.source_combo.itemData(i)
            if item_data == "speaker":
                self.source_combo.setItemText(i, t["system_audio"])
            elif item_data == "both":
                self.source_combo.setItemText(i, t["mic_system"])
        
        # Update Auto option in language combo
        for i in range(self.lang_combo.count()):
            item_data = self.lang_combo.itemData(i)
            if item_data == "auto":
                self.lang_combo.setItemText(i, t["auto"])
                break
        
        # Store current translations for status updates
        self._current_translations = t
    
    def _save_config(self):
        """Save config to file"""
        import json
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Config] Failed to save: {e}")
    
    def zoom_in(self):
        """Increase UI scale"""
        if self.ui_scale < 1.5:
            self.ui_scale += 0.1
            self._apply_ui_scale()
    
    def zoom_out(self):
        """Decrease UI scale"""
        if self.ui_scale > 0.7:
            self.ui_scale -= 0.1
            self._apply_ui_scale()
    
    def _apply_ui_scale(self):
        """Apply the current UI scale to all elements"""
        scale = self.ui_scale
        self.zoom_label.setText(f"{int(scale * 100)}%")
        
        # Scale base font sizes
        base_font = int(11 * scale)
        large_font = int(14 * scale)
        caption_font = int(self.caption_settings.get('font_size', 18) * scale)
        
        # Scale window size
        base_width = 900
        base_height = 300
        new_width = int(base_width * scale)
        new_height = int(base_height * scale)
        self.resize(new_width, new_height)
        self.setMinimumSize(int(500 * scale), int(220 * scale))
        
        # Update title label
        self.title_label.setStyleSheet(f"color: #f8fafc; font-size: {int(16 * scale)}px; font-weight: bold;")
        
        # Update labels
        for label in [self.source_label, self.lang_label, self.translate_emoji_label]:
            label.setStyleSheet(f"font-size: {int(16 * scale)}px;")
        
        # Update status label
        self.status_label.setStyleSheet(f"color: #94a3b8; font-size: {base_font}px;")
        
        # Update checkboxes
        checkbox_style = f"""
            QCheckBox {{
                color: #94a3b8;
                font-size: {base_font}px;
            }}
            QCheckBox::indicator {{
                width: {int(14 * scale)}px;
                height: {int(14 * scale)}px;
                border-radius: 3px;
                border: 1px solid #475569;
                background-color: #334155;
            }}
        """
        self.offline_checkbox.setStyleSheet(checkbox_style + """
            QCheckBox::indicator:checked {
                background-color: #10b981;
                border-color: #10b981;
            }
        """)
        self.translate_checkbox.setStyleSheet(checkbox_style + """
            QCheckBox::indicator:checked {
                background-color: #f59e0b;
                border-color: #f59e0b;
            }
        """)
        self.show_original_checkbox.setStyleSheet(checkbox_style + """
            QCheckBox::indicator:checked {
                background-color: #8b5cf6;
                border-color: #8b5cf6;
            }
        """)
        
        # Update combo boxes
        combo_style = f"""
            QComboBox {{
                background-color: #334155; color: white;
                border: 1px solid #475569; border-radius: 6px;
                padding: {int(5 * scale)}px {int(10 * scale)}px; 
                min-width: {int(100 * scale)}px;
                font-size: {base_font}px;
            }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox QAbstractItemView {{
                background-color: #1e293b; color: white;
                selection-background-color: #6366f1;
            }}
        """
        self.source_combo.setStyleSheet(combo_style)
        self.lang_combo.setStyleSheet(combo_style)
        self.translate_lang_combo.setStyleSheet(combo_style.replace("#6366f1", "#f59e0b"))
        
        # Special gradient style for interface language selector
        interface_lang_style = f"""
            QComboBox {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #4f46e5, stop:0.5 #7c3aed, stop:1 #a855f7);
                color: white;
                border: 2px solid #818cf8;
                border-radius: {int(8 * scale)}px;
                padding: {int(4 * scale)}px {int(10 * scale)}px;
                min-width: {int(85 * scale)}px;
                font-size: {int(12 * scale)}px;
                font-weight: bold;
            }}
            QComboBox:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #6366f1, stop:0.5 #8b5cf6, stop:1 #c084fc);
                border: 2px solid #a5b4fc;
            }}
            QComboBox::drop-down {{
                border: none;
                width: {int(20 * scale)}px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: {int(5 * scale)}px solid transparent;
                border-right: {int(5 * scale)}px solid transparent;
                border-top: {int(6 * scale)}px solid white;
                margin-right: {int(5 * scale)}px;
            }}
            QComboBox QAbstractItemView {{
                background-color: #1e1b4b;
                color: white;
                selection-background-color: #7c3aed;
                border: 2px solid #818cf8;
                border-radius: 6px;
                padding: {int(4 * scale)}px;
            }}
            QComboBox QAbstractItemView::item {{
                padding: {int(6 * scale)}px {int(10 * scale)}px;
                border-radius: 4px;
            }}
            QComboBox QAbstractItemView::item:hover {{
                background-color: #4c1d95;
            }}
        """
        self.interface_lang_combo.setStyleSheet(interface_lang_style)
        
        # Update buttons
        btn_size = int(30 * scale)
        self.start_btn.setFixedHeight(btn_size)
        self.start_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #10b981; color: white;
                border: none; border-radius: 6px;
                font-size: {base_font}px; font-weight: bold;
                padding: 0 {int(15 * scale)}px;
            }}
            QPushButton:hover {{ background-color: #34d399; }}
        """)
        
        for btn in [self.settings_btn, self.close_btn]:
            btn.setFixedSize(btn_size, btn_size)
        
        self.settings_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #475569; color: white;
                border: none; border-radius: 6px;
                font-size: {int(16 * scale)}px;
            }}
            QPushButton:hover {{ background-color: #64748b; }}
        """)
        
        self.close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #ef4444; color: white;
                border: none; border-radius: 6px;
                font-size: {int(14 * scale)}px;
            }}
            QPushButton:hover {{ background-color: #f87171; }}
        """)
        
        # Update zoom buttons
        zoom_btn_size = int(24 * scale)
        self.zoom_out_btn.setFixedSize(zoom_btn_size, zoom_btn_size)
        self.zoom_in_btn.setFixedSize(zoom_btn_size, zoom_btn_size)
        zoom_btn_style = f"""
            QPushButton {{
                background-color: #334155; color: white;
                border: 1px solid #475569; border-radius: 4px;
                font-size: {int(16 * scale)}px; font-weight: bold;
            }}
            QPushButton:hover {{ background-color: #475569; }}
        """
        self.zoom_out_btn.setStyleSheet(zoom_btn_style)
        self.zoom_in_btn.setStyleSheet(zoom_btn_style)
        self.zoom_label.setStyleSheet(f"color: #94a3b8; font-size: {int(10 * scale)}px; min-width: {int(35 * scale)}px;")
        
        # Update caption display font size based on zoom
        self._apply_caption_style()
    
    def _apply_caption_style(self):
        """Apply caption display styling with current zoom scale"""
        scale = self.ui_scale
        base_caption_size = self.caption_settings.get('font_size', 18)
        scaled_caption_size = int(base_caption_size * scale)
        
        font_family = self.caption_settings.get('font_family', 'Segoe UI')
        text_color = self.caption_settings.get('text_color', '#f8fafc')
        bg_color = self.caption_settings.get('bg_color', '#1e293b')
        bg_opacity = self.caption_settings.get('bg_opacity', 80) / 100.0
        
        # Convert hex to rgba
        bg_r = int(bg_color[1:3], 16)
        bg_g = int(bg_color[3:5], 16)
        bg_b = int(bg_color[5:7], 16)
        
        caption_style = f"""
            QTextEdit {{
                background-color: rgba({bg_r}, {bg_g}, {bg_b}, {bg_opacity});
                color: {text_color};
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 10px;
                font-family: '{font_family}';
                font-size: {scaled_caption_size}px;
            }}
        """
        
        self.caption_display.setStyleSheet(caption_style)
        self.translation_display.setStyleSheet(caption_style)
        self.translation_display_2.setStyleSheet(caption_style)
        
        # Update single-line ticker labels if they exist
        if hasattr(self, 'ticker_label') and self.ticker_label:
            ticker_style = f"color: {text_color}; font-family: '{font_family}'; font-size: {scaled_caption_size}px;"
            self.ticker_label.setStyleSheet(ticker_style)
        if hasattr(self, 'translation_ticker') and self.translation_ticker:
            self.translation_ticker.setStyleSheet(f"color: {text_color}; font-family: '{font_family}'; font-size: {scaled_caption_size}px;")
        if hasattr(self, 'translation_ticker_2') and self.translation_ticker_2:
            self.translation_ticker_2.setStyleSheet(f"color: {text_color}; font-family: '{font_family}'; font-size: {scaled_caption_size}px;")
    
    def _update_dual_display_visibility(self):
        """Update translation display visibility based on translation enabled + show original + dual captioning
        
        Display modes:
        - Dual Mode ON: Two translation boxes (1st lang in caption_display, 2nd lang in translation_display)
                        Original text is NOT shown, both boxes show translations
        - Show Original ON (Dual OFF): Original text + Translation (caption_display + translation_display)
        - Both OFF: Only translation shown (replaces original in caption_display)
        """
        # Safety check - ensure UI is initialized
        if not hasattr(self, 'translation_display_2'):
            return
        
        is_single_line = self.caption_settings.get('caption_mode', 'multi') == 'single'
        translation_active = self.translation_enabled and self.translate_checkbox.isChecked()
        show_original = self.show_original_checkbox.isChecked()
        dual_mode = self.dual_captioning_enabled and translation_active
        
        if is_single_line:
            # Single-line mode
            if dual_mode:
                # Dual mode: ticker_label = 1st translation, translation_ticker = 2nd translation
                self.ticker_label.show()
                self.translation_ticker.show()
                self.translation_ticker_2.hide()  # Not used in dual mode
            elif translation_active and show_original:
                # Show original mode: ticker_label = original, translation_ticker = translation
                self.ticker_label.show()
                self.translation_ticker.show()
                self.translation_ticker_2.hide()
            elif translation_active and not show_original:
                # Translation only mode: hide original, show only translation in translation_ticker
                self.ticker_label.hide()
                self.translation_ticker.show()
                self.translation_ticker_2.hide()
            else:
                # No translation - show only original
                self.ticker_label.show()
                self.translation_ticker.hide()
                self.translation_ticker_2.hide()
        else:
            # Multi-line mode
            if dual_mode:
                # Dual mode: caption_display = 1st translation, translation_display = 2nd translation
                self.caption_display.show()
                self.translation_display.show()
                self.translation_display_2.hide()  # Not used - we use caption_display for 1st translation
                # Update placeholder texts for dual mode
                self.caption_display.setPlaceholderText(f"🌐 Translation 1 ({self.translation_target_lang})...")
                self.translation_display.setPlaceholderText(f"🌐 Translation 2 ({self.translation_target_lang_2})...")
                # Resize window
                current_geo = self.geometry()
                min_height = 400
                self.setMinimumSize(600, 350)
                if current_geo.height() < min_height:
                    self.resize(max(current_geo.width(), 950), min_height)
            elif translation_active and show_original:
                # Show original mode: caption_display = original, translation_display = translation
                self.caption_display.show()
                self.translation_display.show()
                self.translation_display_2.hide()
                # Reset placeholder texts
                self.caption_display.setPlaceholderText("Original captions will appear here...")
                self.translation_display.setPlaceholderText("🌐 Translations will appear here...")
                # Resize window
                current_geo = self.geometry()
                min_height = 400
                self.setMinimumSize(600, 350)
                if current_geo.height() < min_height:
                    self.resize(max(current_geo.width(), 950), min_height)
            elif translation_active and not show_original:
                # Translation only mode: hide original, show only translation_display
                self.caption_display.hide()
                self.translation_display.show()
                self.translation_display_2.hide()
                # Update placeholder
                self.translation_display.setPlaceholderText("🌐 Translations will appear here...")
                self.setMinimumSize(500, 220)
            else:
                # No translation - show only original caption display
                self.caption_display.show()
                self.translation_display.hide()
                self.translation_display_2.hide()
                # Reset placeholder
                self.caption_display.setPlaceholderText("Captions will appear here...")
                self.setMinimumSize(500, 220)
    
    def on_translate_lang_changed(self, index):
        """Handle translation target language change"""
        target_lang = self.translate_lang_combo.currentData()
        self.translation_target_lang = target_lang
        
        # Update the worker's target language if it's running
        if self.translation_worker is not None and self.translation_worker.isRunning():
            self.translation_worker.set_target_language(target_lang)
            # Clear pending translations since language changed
            self.translation_worker.clear_queue()
            print(f"[Translation] Target language changed to: {target_lang}")
    
    def _start_translation_worker(self, target_lang):
        """Start the translation worker with the specified target language"""
        # Determine if we should use online or offline translation
        # Use online (Reverie API) when STT is online, offline (IndicTrans2) when STT is offline
        use_online_translation = not self.offline_checkbox.isChecked()
        
        if self.translation_worker is not None and self.translation_worker.isRunning():
            # Worker already running, just update target language and mode
            self.translation_worker.set_target_language(target_lang)
            self.translation_worker.set_online_mode(use_online_translation)
            return
        
        # Clean up old worker if exists but not running
        if self.translation_worker is not None:
            self.translation_worker = None
        
        mode_str = "online (Reverie API)" if use_online_translation else "offline (IndicTrans2)"
        print(f"[Translation] Starting worker with target: {target_lang}, mode: {mode_str}")
        
        if use_online_translation:
            self.status_label.setText("🔄 Starting online translation...")
        else:
            self.status_label.setText("🔄 Loading offline translation model...")
        
        self.translation_worker = TranslationWorker(
            tgt_lang=target_lang,
            device="cpu",
            use_online=use_online_translation
        )
        self.translation_worker.translation_ready.connect(self.on_translation_ready)
        self.translation_worker.model_loaded.connect(self.on_translation_model_loaded)
        self.translation_worker.error_signal.connect(self.on_translation_error)
        self.translation_worker.loading_started.connect(self.on_translation_loading_started)
        self.translation_worker.start()
    
    def _update_translation_mode(self):
        """Update translation worker mode based on offline checkbox"""
        if self.translation_worker is not None and self.translation_worker.isRunning():
            use_online = not self.offline_checkbox.isChecked()
            self.translation_worker.set_online_mode(use_online)
            mode_str = "online (Reverie API)" if use_online else "offline (IndicTrans2)"
            print(f"[Translation] Mode updated to: {mode_str}")
        
        # Also update second translation worker if running
        if self.translation_worker_2 is not None and self.translation_worker_2.isRunning():
            use_online = not self.offline_checkbox.isChecked()
            self.translation_worker_2.set_online_mode(use_online)
            print(f"[Translation 2] Mode updated to: {mode_str}")
    
    def on_translation_loading_started(self):
        """Handle translation model loading started"""
        self.status_label.setText("🔄 Loading translation models (this may take a moment)...")
        QApplication.processEvents()
    
    def _stop_translation_worker(self):
        """Stop the translation worker"""
        if self.translation_worker is not None:
            self.translation_worker.stop()
            if not self.translation_worker.wait(1000):
                self.translation_worker.terminate()
            self.translation_worker = None
            print("[Translation] Worker stopped")
    
    def _start_translation_worker_2(self, target_lang):
        """Start the second translation worker for dual captioning"""
        use_online_translation = not self.offline_checkbox.isChecked()
        
        if self.translation_worker_2 is not None and self.translation_worker_2.isRunning():
            self.translation_worker_2.set_target_language(target_lang)
            self.translation_worker_2.set_online_mode(use_online_translation)
            return
        
        if self.translation_worker_2 is not None:
            self.translation_worker_2 = None
        
        mode_str = "online (Reverie API)" if use_online_translation else "offline (IndicTrans2)"
        print(f"[Translation 2] Starting worker with target: {target_lang}, mode: {mode_str}")
        
        self.translation_worker_2 = TranslationWorker(
            tgt_lang=target_lang,
            device="cpu",
            use_online=use_online_translation
        )
        self.translation_worker_2.translation_ready.connect(self.on_translation_ready_2)
        self.translation_worker_2.model_loaded.connect(lambda m: print(f"[Translation 2] Model loaded: {m}"))
        self.translation_worker_2.error_signal.connect(lambda e: print(f"[Translation 2] Error: {e}"))
        self.translation_worker_2.start()
    
    def _stop_translation_worker_2(self):
        """Stop the second translation worker"""
        if self.translation_worker_2 is not None:
            self.translation_worker_2.stop()
            if not self.translation_worker_2.wait(1000):
                self.translation_worker_2.terminate()
            self.translation_worker_2 = None
            print("[Translation 2] Worker stopped")
    
    def on_translation_model_loaded(self, model_name):
        """Handle translation model loaded signal"""
        print(f"[Translation] Model loaded: {model_name}")
        if self.is_recording:
            self.status_label.setText("🎙️ Recording (translation ready)")
        else:
            self.status_label.setText("Translation ready")
    
    def on_translation_ready(self, original_text, translated_text):
        """Handle translated text - COMPLETE SENTENCES ONLY
        
        BREAKTHROUGH STRATEGY:
        - NEVER show partial/in-progress translations
        - Only display when we're confident the sentence/phrase is complete
        - Keep previous translation visible until new complete one arrives
        - This ensures users always have stable, readable text
        """
        print(f"[Translation Result] '{original_text[:30]}...' → '{translated_text[:30]}...'")
        
        # CRITICAL: Detect failed translations (API timeout returns original text)
        # If translated == original (and languages are different), translation failed
        # Don't update pending - keep the last successful translation
        if original_text.strip().lower() == translated_text.strip().lower():
            # Check if this is a same-language pass-through (which is OK)
            # Use current_detected_lang if available, otherwise fall back
            if self.auto_language_mode and self.current_detected_lang:
                src_lang = self.current_detected_lang
            else:
                src_lang = self.lang_combo.currentData() if hasattr(self, 'lang_combo') else 'en'
                if src_lang == 'auto':
                    src_lang = 'en'
            tgt_lang = self.translate_lang_combo.currentData() if hasattr(self, 'translate_lang_combo') else 'hi'
            
            if src_lang != tgt_lang:
                print(f"[Display] ⚠️ Translation failed (returned original text) - keeping previous")
                return  # Don't update with failed translation
        
        # Update tracking state
        if hasattr(self, '_trans_state'):
            self._trans_state['translated_prefix'] = original_text
            self._trans_state['translated_result'] = translated_text
        
        # Initialize complete-sentence display state
        if not hasattr(self, '_sentence_display'):
            self._sentence_display = {
                'pending_translation': "",       # Latest translation waiting to be shown
                'pending_original': "",          # Original text for pending
                'displayed_translation': "",     # Currently shown (stable) translation
                'is_complete': False,            # Whether pending is a complete sentence
                'silence_detected': False,       # Flag set by ASR when silence detected
            }
        
        # Store as pending (not displayed yet)
        self._sentence_display['pending_translation'] = translated_text
        self._sentence_display['pending_original'] = original_text
        
        # Check if this translation looks COMPLETE
        # A complete translation ends with sentence-ending punctuation
        # Check BOTH translated AND original text (source may have period but translation might not)
        sentence_enders = ['.', '!', '?', '।', '؟', '。', '？', '！', '॥', '؛', '۔']
        translated_ends = any(translated_text.rstrip().endswith(p) for p in sentence_enders)
        original_ends = any(original_text.rstrip().endswith(p) for p in sentence_enders)
        is_complete = translated_ends or original_ends
        
        if is_complete:
            print(f"[Display] Sentence ender detected: translated={translated_ends}, original={original_ends}")
        
        # Also check if silence was detected (natural pause = complete thought)
        if self._sentence_display.get('silence_detected', False):
            is_complete = True
            self._sentence_display['silence_detected'] = False  # Reset flag
        
        self._sentence_display['is_complete'] = is_complete
        
        if is_complete:
            # COMPLETE sentence - display immediately
            print(f"[Display] ✓ Complete sentence detected, updating display")
            self._show_complete_translation()
        else:
            # INCOMPLETE - don't update display, keep previous stable text
            # But start a fallback timer in case sentence never "completes"
            print(f"[Display] ⏳ Incomplete, keeping previous translation visible")
            self._schedule_fallback_display(2500)  # 2.5 second fallback
    
    def on_translation_ready_2(self, original_text, translated_text):
        """Handle translated text from second translation worker (dual captioning)"""
        print(f"[Translation 2 Result] '{original_text[:30]}...' → '{translated_text[:30]}...'")
        
        # Check for failed translation
        if original_text.strip().lower() == translated_text.strip().lower():
            if self.auto_language_mode and self.current_detected_lang:
                src_lang = self.current_detected_lang
            else:
                src_lang = self.lang_combo.currentData() if hasattr(self, 'lang_combo') else 'en'
                if src_lang == 'auto':
                    src_lang = 'en'
            tgt_lang = self.translation_target_lang_2  # Use the stored setting
            
            if src_lang != tgt_lang:
                print(f"[Display 2] ⚠️ Translation failed - keeping previous")
                return
        
        # Initialize state for second translation display
        if not hasattr(self, '_sentence_display_2'):
            self._sentence_display_2 = {
                'pending_translation': "",
                'pending_original': "",
                'displayed_translation': "",
            }
        
        self._sentence_display_2['pending_translation'] = translated_text
        self._sentence_display_2['pending_original'] = original_text
        
        # Check if complete (same logic as primary)
        sentence_enders = ['.', '!', '?', '।', '؟', '。', '？', '！', '॥', '؛', '۔']
        translated_ends = any(translated_text.rstrip().endswith(p) for p in sentence_enders)
        original_ends = any(original_text.rstrip().endswith(p) for p in sentence_enders)
        is_complete = translated_ends or original_ends
        
        if is_complete:
            self._show_complete_translation_2()
        else:
            self._schedule_fallback_display_2(2500)
    
    def _schedule_fallback_display_2(self, delay_ms):
        """Fallback timer for second translation"""
        if not hasattr(self, '_fallback_timer_2'):
            self._fallback_timer_2 = QTimer()
            self._fallback_timer_2.setSingleShot(True)
            self._fallback_timer_2.timeout.connect(self._fallback_show_translation_2)
        
        if not self._fallback_timer_2.isActive():
            self._fallback_timer_2.start(delay_ms)
    
    def _fallback_show_translation_2(self):
        """Fallback: show pending second translation after timeout"""
        # Safety check - don't execute if we're not recording anymore
        if not self.is_recording:
            return
        if not hasattr(self, '_sentence_display_2'):
            return
        
        pending = self._sentence_display_2.get('pending_translation', "")
        displayed = self._sentence_display_2.get('displayed_translation', "")
        
        if pending and pending != displayed:
            self._show_complete_translation_2()
    
    def _show_complete_translation_2(self):
        """Display the pending second translation"""
        if not hasattr(self, '_sentence_display_2'):
            return
        
        if hasattr(self, '_fallback_timer_2'):
            self._fallback_timer_2.stop()
        
        pending = self._sentence_display_2.get('pending_translation', "")
        if not pending:
            return
        
        self._sentence_display_2['displayed_translation'] = pending
        
        is_single_line = self.caption_settings.get('caption_mode', 'multi') == 'single'
        original_text = self._sentence_display_2.get('pending_original', pending)
        
        self._display_translation_2(original_text, pending, is_single_line)
    
    def _display_translation_2(self, original_text, translated_text, is_single_line):
        """Display second translation in translation_display (for dual mode)"""
        if is_single_line:
            # In dual mode, use translation_ticker for 2nd language
            self.translation_ticker.setText(translated_text)
        else:
            # Multi-line: Use translation_display for 2nd language in dual mode
            if not hasattr(self, '_translation_lines_2'):
                self._translation_lines_2 = []
            
            MAX_LINES = 3
            
            # Avoid duplicates
            if self._translation_lines_2 and translated_text.strip() == self._translation_lines_2[-1].strip():
                return
            
            self._translation_lines_2.append(translated_text)
            while len(self._translation_lines_2) > MAX_LINES:
                self._translation_lines_2.pop(0)
            
            display_text = '\n'.join(self._translation_lines_2)
            # Use translation_display (not translation_display_2) for dual mode
            self.translation_display.setPlainText(display_text)
    
    def _schedule_fallback_display(self, delay_ms):
        """Fallback: show translation after delay even if not complete"""
        if not hasattr(self, '_fallback_timer'):
            from PyQt5.QtCore import QTimer
            self._fallback_timer = QTimer()
            self._fallback_timer.setSingleShot(True)
            self._fallback_timer.timeout.connect(self._fallback_show_translation)
        
        # Only restart if not already waiting
        if not self._fallback_timer.isActive():
            self._fallback_timer.start(delay_ms)
    
    def _fallback_show_translation(self):
        """Fallback: show pending translation after timeout"""
        # Safety check - don't execute if we're not recording anymore
        if not self.is_recording:
            return
        if not hasattr(self, '_sentence_display'):
            return
        
        pending = self._sentence_display.get('pending_translation', "")
        displayed = self._sentence_display.get('displayed_translation', "")
        
        # Only update if pending is significantly different/longer
        if pending and pending != displayed:
            print(f"[Display] ⏰ Fallback timeout - showing pending translation")
            self._show_complete_translation()
    
    def _show_complete_translation(self):
        """Display the pending translation (called when we know it's complete)"""
        if not hasattr(self, '_sentence_display'):
            return
        
        # Stop fallback timer
        if hasattr(self, '_fallback_timer'):
            self._fallback_timer.stop()
        
        pending = self._sentence_display.get('pending_translation', "")
        if not pending:
            return
        
        # Update displayed
        self._sentence_display['displayed_translation'] = pending
        
        # Determine display mode
        show_original = self.show_original_checkbox.isChecked()
        is_single_line = self.caption_settings.get('caption_mode', 'multi') == 'single'
        original_text = self._sentence_display.get('pending_original', pending)
        dual_mode = self.dual_captioning_enabled and self.translation_enabled
        
        if dual_mode:
            # Dual mode: 1st translation goes to caption_display (not original text)
            self._display_translation_in_caption_box(pending, is_single_line)
        elif show_original:
            # Show original mode: original in caption_display, translation in translation_display
            self._display_translation_dual(original_text, pending, is_single_line)
        else:
            # Replace mode: translation replaces original in caption_display
            self._display_translation_replace(original_text, pending, is_single_line)
    
    def _display_translation_in_caption_box(self, translated_text, is_single_line):
        """Display first translation in caption_display (for dual mode)"""
        if is_single_line:
            self.ticker_label.setText(translated_text)
        else:
            # Multi-line: Show in caption_display
            if not hasattr(self, '_dual_trans_lines_1'):
                self._dual_trans_lines_1 = []
            
            MAX_LINES = 3
            
            # Avoid duplicates
            if self._dual_trans_lines_1 and translated_text.strip() == self._dual_trans_lines_1[-1].strip():
                return
            
            self._dual_trans_lines_1.append(translated_text)
            while len(self._dual_trans_lines_1) > MAX_LINES:
                self._dual_trans_lines_1.pop(0)
            
            display_text = '\n'.join(self._dual_trans_lines_1)
            self.caption_display.setPlainText(display_text)
            
            # Store in history
            if translated_text.strip():
                self.translation_history.append(translated_text.strip())
    
    def _mark_silence_detected(self):
        """Called when ASR detects silence - marks pending translation as complete"""
        if hasattr(self, '_sentence_display'):
            self._sentence_display['silence_detected'] = True
            # If we have pending translation, show it now
            if self._sentence_display.get('pending_translation'):
                print(f"[Display] 🔇 Silence detected - showing pending translation")
                self._show_complete_translation()
    
    def _display_translation_dual(self, original_text, translated_text, is_single_line):
        """Display translation in separate translation display area (dual caption mode)
        
        STABLE CAPTION STYLE:
        - New translations appear at the bottom
        - Previous translations stay in place
        - When too many lines, oldest is removed
        - No scrolling animation - stable display
        """
        if is_single_line:
            # Single-line: Show only translated text in translation ticker
            self.translation_ticker.setText(translated_text)
            # Store in translation history
            if translated_text.strip() and (not self.translation_history or self.translation_history[-1] != translated_text.strip()):
                self.translation_history.append(translated_text.strip())
        else:
            # Multi-line: Stable translation display
            # Initialize translation history if needed
            if not hasattr(self, '_translation_lines'):
                self._translation_lines = []  # List of completed translations
            
            MAX_LINES = 3  # Maximum lines to show
            
            # Check if this is updating current segment or new segment
            is_same_segment = False
            if hasattr(self, '_last_trans_original') and self._last_trans_original:
                # If new original is extension of old, it's the same segment
                if original_text.startswith(self._last_trans_original) or \
                   self._last_trans_original.startswith(original_text[:min(len(original_text), 20)]):
                    is_same_segment = True
            
            # Also check if translation is same as last one (avoid duplicates)
            if self._translation_lines and translated_text.strip() == self._translation_lines[-1].strip():
                # Same translation - don't add or update
                return
            
            if is_same_segment:
                # Update the current (last) line in place
                if self._translation_lines:
                    self._translation_lines[-1] = translated_text
                else:
                    self._translation_lines.append(translated_text)
                    # Store new translation in history
                    if translated_text.strip():
                        self.translation_history.append(translated_text.strip())
            else:
                # New segment - add as new line at bottom
                self._translation_lines.append(translated_text)
                # Store new translation in history
                if translated_text.strip() and (not self.translation_history or self.translation_history[-1] != translated_text.strip()):
                    self.translation_history.append(translated_text.strip())
                
                # Remove oldest lines if we exceed max
                while len(self._translation_lines) > MAX_LINES:
                    self._translation_lines.pop(0)
            
            self._last_trans_original = original_text
            
            # Build display content - each translation on its own line
            display_text = '\n'.join(self._translation_lines)
            self.translation_display.setPlainText(display_text)
    
    def _display_translation_replace(self, original_text, translated_text, is_single_line):
        """Replace pending original with translation (Show Original OFF)"""
        if is_single_line:
            # Single-line: Just show translation (discards old)
            self.ticker_label.setText(translated_text)
            self.single_line_text = translated_text
        else:
            # Multi-line: ACCUMULATE - replace pending markers, keep translated history
            current_text = self.caption_display.toPlainText()
            lines = current_text.split('\n')
            
            new_lines = []
            replaced = False
            
            for line in lines:
                if line.endswith(' ⏳') and not replaced:
                    # Replace first pending line with translation
                    new_lines.append(translated_text)
                    replaced = True
                elif line.endswith(' ⏳'):
                    # Skip additional pending lines
                    continue
                elif line.strip():
                    new_lines.append(line)
            
            # If no pending found, just append translation
            if not replaced:
                new_lines.append(translated_text)
            
            new_content = '\n'.join(new_lines)
            self.caption_display.setPlainText(new_content)
            self.partial_text = ""
            
            scrollbar = self.caption_display.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
    
    def on_translation_error(self, error_msg):
        """Handle translation error"""
        print(f"[Translation Error] {error_msg}")
        self.status_label.setText(f"⚠️ Translation: {error_msg[:30]}...")
        
    def stop_recording(self):
        """Stop recording - use non-blocking cleanup to prevent UI freeze"""
        print("[DEBUG] stop_recording called")
        
        # Set recording lock to prevent race conditions
        self._recording_lock = True
        
        # Stop fallback timers to prevent callbacks during cleanup
        if hasattr(self, '_fallback_timer') and self._fallback_timer:
            self._fallback_timer.stop()
        if hasattr(self, '_fallback_timer_2') and self._fallback_timer_2:
            self._fallback_timer_2.stop()
        
        # Stop response watchdog
        self._stop_response_watchdog()
        
        # Stop online retry timer
        if self.online_retry_timer:
            self.online_retry_timer.stop()
        
        # Reset auto-switch state
        self.auto_switched_offline = False
        
        # Signal all workers to stop first (non-blocking)
        if self.audio_capture:
            self.audio_capture.stop()
            
        if self.stt_worker:
            self.stt_worker.stop()
        
        if self.whisper_worker:
            self.whisper_worker.stop()
        
        if self.language_detector:
            self.language_detector.stop()
        
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
        # offline_checkbox stays enabled and preserves its state
        self.status_label.setText("Stopped")
        self.audio_level_fill.setGeometry(0, 0, 0, 6)
        
        # Clean up threads in background using QTimer
        QTimer.singleShot(100, self._cleanup_workers)
        
    def _cleanup_workers_sync(self):
        """Synchronously clean up all workers before starting new recording"""
        print("[DEBUG] Synchronous cleanup starting...")
        
        # Stop all workers first (EXCEPT translation worker - keep it running for persistence)
        if self.audio_capture:
            self.audio_capture.stop()
        if self.stt_worker:
            self.stt_worker.stop()
        if self.whisper_worker:
            self.whisper_worker.stop()
        if self.language_detector:
            self.language_detector.stop()
        # NOTE: Translation worker is NOT stopped here - models take too long to reload
        
        # Stop timers
        self._stop_response_watchdog()
        self._stop_online_retry_timer()
        
        # Wait for threads to finish
        if self.audio_capture:
            self.audio_capture.wait(300)
            self.audio_capture = None
            
        if self.stt_worker:
            self.stt_worker.wait(300)
            self.stt_worker = None
        
        if self.whisper_worker:
            self.whisper_worker.wait(500)
            self.whisper_worker = None
        
        if self.language_detector:
            self.language_detector.wait(300)
            self.language_detector = None
        
        # Translation worker is preserved across recordings
        
        # Reset state
        self.auto_switched_offline = False
        self.api_failed = False
        
        print("[DEBUG] Synchronous cleanup complete")
        
    def _cleanup_workers(self):
        """Clean up worker threads (called after UI update via timer)"""
        # Skip if already cleaned up
        if not self.audio_capture and not self.stt_worker and not self.whisper_worker and not self.language_detector:
            print("[DEBUG] Workers already cleaned up")
            return
            
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
        
        if self.language_detector:
            self.language_detector.stop()
            if not self.language_detector.wait(200):
                self.language_detector.terminate()
            self.language_detector = None
        
        # Note: Translation worker is NOT cleaned up here - it persists across recordings
        
        # Stop retry timer
        self._stop_online_retry_timer()
            
        # Reset auto language state
        if self.auto_language_mode:
            # Reset the combo text back to just "Auto"
            try:
                self.lang_combo.setItemText(0, "🔄 Auto")
            except:
                pass
            
        # Release the recording lock
        self._recording_lock = False
        print("[DEBUG] Workers cleaned up, lock released")
        
    def on_audio_data(self, data):
        """Send audio to the appropriate worker (online or offline)"""
        import time
        self.last_audio_sent_time = time.time()
        
        # Send to language detector if auto mode is active
        if self.auto_language_mode and self.language_detector:
            self.language_detector.add_audio(data)
        
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
    
    def on_language_detected(self, lang_code, confidence):
        """Handle detected language from auto-detection"""
        # Check if we support this language
        if lang_code not in LANGUAGES:
            print(f"[LangDetect] Detected unsupported language: {lang_code}")
            return
        
        # Check if this is the initial detection or a change
        is_initial = (self.current_detected_lang is None)
        is_same = (lang_code == self.current_detected_lang)
        
        if is_same and not is_initial:
            # Same language detected again, no action needed
            return
        
        print(f"[LangDetect] {'Initial' if is_initial else 'Switching'}: {self.current_detected_lang} → {lang_code} ({confidence:.0%})")
        self.current_detected_lang = lang_code
        
        # Update the language combo to show detected language (visual feedback)
        lang_index = self.lang_combo.findData(lang_code)
        if lang_index >= 0:
            # Block signals to prevent triggering other handlers
            self.lang_combo.blockSignals(True)
            # Show the detected language name with auto prefix
            lang_name = LANGUAGES.get(lang_code, lang_code)
            self.lang_combo.setItemText(0, f"🔄 Auto ({lang_name})")
            self.lang_combo.blockSignals(False)
        
        # Update the STT worker with new language
        if self.use_offline_mode and self.whisper_worker:
            # For Whisper, we can update the language setting directly
            self.whisper_worker.language = lang_code
            print(f"[LangDetect] Updated Whisper language to: {lang_code}")
        elif self.stt_worker:
            # For online API, we need to RECONNECT with new language
            # because the language is part of the WebSocket URL
            old_lang = self.stt_worker.language
            if old_lang != lang_code:
                print(f"[LangDetect] Reconnecting API with new language: {old_lang} → {lang_code}")
                self._reconnect_stt_with_language(lang_code)
        
        # Show notification
        self.status_label.setText(f"🌐 Detected: {LANGUAGES.get(lang_code, lang_code)}")
    
    def _reconnect_stt_with_language(self, new_lang_code):
        """Reconnect the online STT worker with a new language"""
        if not self.stt_worker:
            return
        
        # Stop the current STT worker
        print(f"[LangDetect] Stopping current STT worker...")
        old_worker = self.stt_worker
        old_worker.stop()
        
        # Create new STT worker with the detected language
        from .config import config
        self.stt_worker = STTWorker(
            api_key=config['api_key'],
            app_id=config['app_id'],
            language=new_lang_code,
            domain=config.get('default_domain', 'generic')
        )
        self.stt_worker.transcription.connect(self.on_transcription)
        self.stt_worker.status_changed.connect(self.on_status_changed)
        self.stt_worker.error_signal.connect(self.on_stt_error)
        self.stt_worker.start()
        
        print(f"[LangDetect] New STT worker started with language: {new_lang_code}")
        
        # Wait for old worker to finish in background
        QTimer.singleShot(500, lambda: self._cleanup_old_stt_worker(old_worker))
    
    def _cleanup_old_stt_worker(self, worker):
        """Clean up an old STT worker after it stops"""
        if worker:
            worker.wait(500)
            print("[LangDetect] Old STT worker cleaned up")
    
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
        
        # Track response time for online mode watchdog
        if not self.use_offline_mode and data.get('source') != 'offline':
            import time
            self.last_online_response_time = time.time()
        
        cause = data.get('cause', '')
        if cause == 'ready':
            return
            
        text = data.get('display_text') or data.get('text', '')
        is_final = data.get('final', False)
        
        # Debug output
        print(f"[Caption] final={is_final}, cause={cause}, text={text[:50] if text else 'empty'}")
        
        if not text or not text.strip():
            return
        
        # Determine display strategy based on translation settings
        show_original = self.show_original_checkbox.isChecked() if hasattr(self, 'show_original_checkbox') else True
        translation_active = self.translation_enabled and self.translation_worker and self.translation_worker.is_ready()
        
        # Display logic:
        # - If translation OFF: Always show original
        # - If translation ON + Show Original ON: Show original (translation appears below)
        # - If translation ON + Show Original OFF: Show original temporarily until translation arrives
        
        if translation_active and not show_original:
            # Translation will replace this - show in "pending" state
            self._display_transcription_pending(text, is_final, cause)
        else:
            # Show original text normally
            self._display_transcription(text, is_final, cause)
        
        # Mark silence detected for translation display (triggers showing pending translation)
        if (is_final or cause == 'silence detected'):
            self._mark_silence_detected()
        
        # Translation logic - translate complete sentences OR long partial text
        if not self.translation_enabled:
            return
            
        if not self.translation_worker or not self.translation_worker.is_ready():
            if self.translation_worker and not self.translation_worker.is_ready():
                self.status_label.setText("🔄 Translation loading...")
            return
        
        # Determine source language
        if self.auto_language_mode and self.current_detected_lang:
            src_lang = self.current_detected_lang
        else:
            src_lang = self.lang_combo.currentData()
            if src_lang == "auto":
                # Auto mode but no detection yet - assume English for now
                # Translation will still work, just might not be optimal
                src_lang = "en"
        
        # Get target language
        tgt_lang = self.translate_lang_combo.currentData() if hasattr(self, 'translate_lang_combo') else "en"
        
        # Skip translation if source and target languages are the same
        # BUT only if we're confident about source language (not auto-guessing)
        if src_lang == tgt_lang:
            # Check if we're just guessing source language
            is_guessing = (self.auto_language_mode and not self.current_detected_lang)
            
            if not is_guessing:
                # Same language confirmed - no translation needed, display ASR text as "translation"
                show_original = self.show_original_checkbox.isChecked() if hasattr(self, 'show_original_checkbox') else True
                is_single_line = self.caption_settings.get('caption_mode', 'multi') == 'single'
                if show_original:
                    self._display_translation_dual(text, text, is_single_line)
                else:
                    self._display_translation_replace(text, text, is_single_line)
                return
            # If guessing, still proceed with translation attempt
        
        import time
        current_time = time.time()
        
        # Initialize translation state if needed
        if not hasattr(self, '_trans_state'):
            self._trans_state = {
                'last_text': "",
                'last_trans_time': 0,
                'last_chunk_time': 0,
                # NEW: Sentence-based translation tracking
                'completed_sentences': "",      # Already translated & finalized sentences
                'completed_translations': "",   # Their translations
                'current_sentence': "",         # Current sentence being translated
                'current_translation': "",      # Its translation (updated progressively)
            }
        
        stripped = text.strip()
        
        # ============================================================
        # SENTENCE-BASED INCREMENTAL TRANSLATION
        # ============================================================
        # Problem: Re-translating entire text wastes API calls & time
        # 
        # Solution:
        # 1. Track completed sentences separately from current sentence
        # 2. Only translate the CURRENT (incomplete) sentence
        # 3. When sentence completes, move to "completed" and start fresh
        # 4. Display: completed_translations + current_translation
        # ============================================================
        
        from .constants import is_sentence_complete, is_clause_complete, SENTENCE_TERMINATORS
        
        should_translate = False
        reason = ""
        
        # Get sentence terminators for source language (tgt_lang already set above)
        src_terminators = SENTENCE_TERMINATORS.get(src_lang, SENTENCE_TERMINATORS.get("en", ['.', '!', '?']))
        
        # Calculate time since last chunk was sent
        time_since_chunk = current_time - self._trans_state['last_chunk_time']
        
        # ============================================================
        # EXTRACT CURRENT SENTENCE TO TRANSLATE
        # ============================================================
        # Find where completed sentences end and current sentence begins
        completed_len = len(self._trans_state['completed_sentences'])
        
        # Current sentence is everything after completed sentences
        current_sentence = stripped[completed_len:].strip() if len(stripped) > completed_len else stripped
        
        # Check if the current text contains a NEW sentence terminator
        # (after what we've already completed)
        new_sentence_ended = False
        sentence_boundary_pos = -1
        
        for i, char in enumerate(current_sentence):
            if char in src_terminators:
                sentence_boundary_pos = i
                new_sentence_ended = True
                # Don't break - find the LAST terminator in current text
        
        # ============================================================
        # FAST TRANSLATION - SIMPLER APPROACH
        # ============================================================
        # Instead of complex incremental chunking, just:
        # 1. Send current sentence as-is (limited to reasonable size)
        # 2. Replace display with latest translation
        # 3. Trust API to handle moderate text lengths (<150 chars) quickly
        # ============================================================
        CHUNK_INTERVAL = 0.5   # 500ms between updates (slower = more stable display)
        MIN_CHARS = 8          # Need at least 8 chars to trigger (avoid single words)
        MAX_CHARS = 200        # Max chars to send
        
        # Skip if too short
        if not current_sentence or len(current_sentence) < MIN_CHARS:
            return
        
        # Check if current sentence changed since last translation
        last_sent = self._trans_state.get('last_sent_text', "")
        
        # TRIGGER 1: Sentence completed - translate immediately (HIGH PRIORITY)
        if new_sentence_ended and sentence_boundary_pos >= 0:
            completed = current_sentence[:sentence_boundary_pos + 1].strip()
            if completed and completed != last_sent:
                # Limit size if needed
                text_to_send = completed[-MAX_CHARS:] if len(completed) > MAX_CHARS else completed
                self._trans_state['last_sent_text'] = completed
                self._trans_state['last_chunk_time'] = current_time
                self.translation_worker.clear_queue()
                self.translation_worker.add_text(text_to_send, src_lang)
                # Also send to second translation worker if dual captioning is enabled
                if self.dual_captioning_enabled and self.translation_worker_2 is not None and self.translation_worker_2.isRunning():
                    self.translation_worker_2.clear_queue()
                    self.translation_worker_2.add_text(text_to_send, src_lang)
                print(f"[Trans] (sentence✓) '{text_to_send[:30]}...' ({len(text_to_send)} chars)")
            return
        
        # TRIGGER 2: Silence/final - translate what we have (HIGH PRIORITY)
        if is_final or cause == 'silence detected' or cause == 'whisper_offline':
            if current_sentence and current_sentence != last_sent:
                text_to_send = current_sentence[-MAX_CHARS:] if len(current_sentence) > MAX_CHARS else current_sentence
                self._trans_state['last_sent_text'] = current_sentence
                self._trans_state['last_chunk_time'] = current_time
                self.translation_worker.clear_queue()
                self.translation_worker.add_text(text_to_send, src_lang)
                # Also send to second translation worker if dual captioning is enabled
                if self.dual_captioning_enabled and self.translation_worker_2 is not None and self.translation_worker_2.isRunning():
                    self.translation_worker_2.clear_queue()
                    self.translation_worker_2.add_text(text_to_send, src_lang)
                print(f"[Trans] (end) '{text_to_send[:30]}...' ({len(text_to_send)} chars)")
            return
        
        # TRIGGER 3: Progressive - every CHUNK_INTERVAL (LOW PRIORITY - for background updates)
        # These won't be shown immediately, just buffered until sentence completes
        if time_since_chunk >= CHUNK_INTERVAL:
            # Send if:
            # 1. First translation (last_sent is empty), OR
            # 2. Text grew by at least 10 chars, OR  
            # 3. Text is significantly different from last sent
            should_send = False
            
            if not last_sent:
                should_send = True  # First chunk
            elif len(current_sentence) >= len(last_sent) + 10:
                should_send = True  # Text grew significantly
            elif current_sentence != last_sent and not current_sentence.startswith(last_sent[:min(20, len(last_sent))]):
                should_send = True  # Text is significantly different (e.g., new segment)
            
            if should_send:
                text_to_send = current_sentence[-MAX_CHARS:] if len(current_sentence) > MAX_CHARS else current_sentence
                self._trans_state['last_sent_text'] = current_sentence
                self._trans_state['last_chunk_time'] = current_time
                self.translation_worker.clear_queue()
                self.translation_worker.add_text(text_to_send, src_lang)
                # Also send to second translation worker if dual captioning is enabled
                if self.dual_captioning_enabled and self.translation_worker_2 is not None and self.translation_worker_2.isRunning():
                    self.translation_worker_2.clear_queue()
                    self.translation_worker_2.add_text(text_to_send, src_lang)
                print(f"[Trans] (chunk) '{text_to_send[:30]}...' ({len(text_to_send)} chars)")
                pass  # Not enough new chars

    
    def _display_transcription_pending(self, text, is_final, cause=""):
        """Display transcription as pending translation (will be replaced by translation)"""
        is_single_line = self.caption_settings.get('caption_mode', 'multi') == 'single'
        
        # Track pending text for replacement
        if not hasattr(self, '_pending_original'):
            self._pending_original = ""
        self._pending_original = text.strip()
        
        if is_single_line:
            # Show with translating indicator
            clean_text = text.strip().replace('\n', ' ').replace('  ', ' ')
            self.ticker_label.setText(clean_text + " ⏳")
        else:
            # Multi-line: Show original with subtle indicator
            current_text = self.caption_display.toPlainText()
            
            # Remove old pending text
            if self.partial_text and current_text.endswith(self.partial_text):
                current_text = current_text[:-len(self.partial_text)].rstrip()
            
            # Remove old pending marker lines
            lines = current_text.split('\n')
            lines = [l for l in lines if not l.endswith(' ⏳')]
            current_text = '\n'.join(lines)
            
            if is_final:
                # Final result - show as pending translation
                if current_text.strip():
                    new_content = current_text.rstrip() + '\n' + text.strip() + ' ⏳'
                else:
                    new_content = text.strip() + ' ⏳'
                self.partial_text = ""
            else:
                # Partial - show at end
                if current_text.strip():
                    new_content = current_text + text.strip() + ' ⏳'
                else:
                    new_content = text.strip() + ' ⏳'
                self.partial_text = text.strip() + ' ⏳'
            
            self.caption_display.setPlainText(new_content)
            scrollbar = self.caption_display.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
    
    def _display_transcription(self, text, is_final, cause=""):
        """Display transcription text in the appropriate mode"""
        is_single_line = self.caption_settings.get('caption_mode', 'multi') == 'single'
        
        # Store final captions in history
        if is_final and text.strip():
            self.caption_history.append(text.strip())
        
        if is_single_line:
            # Ticker-style single line mode - show latest text cleanly
            clean_text = text.strip().replace('\n', ' ').replace('  ', ' ')
            
            if is_final:
                # Final result - this becomes the new "latest" complete text
                # Add to running text with separator
                if self.single_line_text:
                    self.single_line_text = self.single_line_text + "  ·  " + clean_text
                else:
                    self.single_line_text = clean_text
                self.partial_text = ""
                self._update_ticker_display(self.single_line_text)
            else:
                # Partial (live) result - show stored text + current partial
                if self.single_line_text:
                    display = self.single_line_text + "  ·  " + clean_text
                else:
                    display = clean_text
                self.partial_text = clean_text
                self._update_ticker_display(display)
        else:
            # Multi-line mode - accumulate all text persistently
            if is_final:
                # Final result - append to history permanently
                current_text = self.caption_display.toPlainText()
                
                # Remove any trailing partial text we added before
                if self.partial_text and current_text.endswith(self.partial_text):
                    current_text = current_text[:-len(self.partial_text)].rstrip()
                
                # Build new content: existing text + new final text
                if current_text:
                    new_content = current_text + "\n" + text.strip()
                else:
                    new_content = text.strip()
                
                self.caption_display.setPlainText(new_content)
                self.partial_text = ""
                
                # Auto-scroll to bottom
                scrollbar = self.caption_display.verticalScrollBar()
                scrollbar.setValue(scrollbar.maximum())
                
            elif cause != 'silence detected':
                # Partial result - show temporarily at the end
                current_text = self.caption_display.toPlainText()
                
                # Remove old partial text if it exists at the end
                if self.partial_text and current_text.endswith(self.partial_text):
                    current_text = current_text[:-len(self.partial_text)]
                
                # Show current content + new partial (partial will be replaced next time)
                display_text = current_text + text.strip()
                self.caption_display.setPlainText(display_text)
                self.partial_text = text.strip()
                
                # Update status
                self.status_label.setText(f"🎤 Listening...")
                
                # Auto-scroll
                scrollbar = self.caption_display.verticalScrollBar()
                scrollbar.setValue(scrollbar.maximum())
    
    def _update_ticker_display(self, full_text):
        """Update ticker display - shows the end portion of text that fits"""
        if not self.ticker_label:
            return
        
        # Get available width for text (label width minus padding)
        available_width = self.ticker_label.width() - 30  # Account for padding
        if available_width <= 0:
            available_width = self.width() - 60  # Fallback
        
        # Get font metrics to measure text
        font = self.ticker_label.font()
        metrics = QFontMetrics(font)
        
        # If text fits, just show it
        text_width = metrics.horizontalAdvance(full_text)
        if text_width <= available_width:
            self.ticker_label.setText(full_text)
        else:
            # Text is too long - show the END portion (newest text)
            # Add ellipsis at start to indicate there's more
            ellipsis = "... "
            ellipsis_width = metrics.horizontalAdvance(ellipsis)
            target_width = available_width - ellipsis_width
            
            # Find how much of the end we can show
            # Start from the end and work backwards
            truncated = full_text
            while metrics.horizontalAdvance(truncated) > target_width and len(truncated) > 1:
                truncated = truncated[1:]
            
            # Clean up - don't start mid-word if possible
            space_idx = truncated.find(' ')
            if space_idx > 0 and space_idx < 20:  # Only if space is near start
                truncated = truncated[space_idx + 1:]
            
            self.ticker_label.setText(ellipsis + truncated)
            
    def on_status_changed(self, status, message):
        if status == "connected":
            self.status_label.setStyleSheet("color: #10b981; font-size: 11px;")
            # If we were auto-switched to offline, switch back to online now
            if self.auto_switched_offline and self.is_recording:
                print("[Auto] API connected - switching back to online mode")
                self._switch_back_to_online()
        elif status == "error":
            self.status_label.setStyleSheet("color: #ef4444; font-size: 11px;")
        elif status == "loading":
            self.status_label.setStyleSheet("color: #f59e0b; font-size: 11px;")
        elif status == "ready":
            self.status_label.setStyleSheet("color: #10b981; font-size: 11px;")
        else:
            self.status_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.status_label.setText(message)
    
    def _switch_back_to_online(self):
        """Switch back to online mode after auto-offline (when API reconnects)"""
        try:
            if not self.auto_switched_offline or not self.is_recording:
                return
            
            print("[Auto] API reconnected - switching back to online mode...")
            
            # Stop the retry timer
            self._stop_online_retry_timer()
            
            # Stop Whisper worker safely
            if self.whisper_worker:
                try:
                    self.whisper_worker.stop()
                    # Don't wait - let it clean up in background
                except Exception as e:
                    print(f"[Auto] Error stopping whisper worker: {e}")
                self.whisper_worker = None
            
            # Update UI
            self.offline_checkbox.blockSignals(True)
            self.offline_checkbox.setChecked(False)
            self.offline_checkbox.blockSignals(False)
            
            self.use_offline_mode = False
            self.api_failed = False
            self.auto_switched_offline = False
            
            # Start the watchdog for ongoing monitoring
            self._start_response_watchdog()
            
            self.status_label.setText("🟢 Back online")
            print("[Auto] Now back in online mode")
        except Exception as e:
            print(f"[Auto] Error switching back to online: {e}")
            import traceback
            traceback.print_exc()
    
    def _start_online_retry_timer(self):
        """Start timer to periodically try reconnecting to online API"""
        if self.online_retry_timer is None:
            self.online_retry_timer = QTimer()
            self.online_retry_timer.timeout.connect(self._try_reconnect_online)
        # Try every 30 seconds
        self.online_retry_timer.start(30000)
        print("[Retry] Started online retry timer (30s interval)")
    
    def _stop_online_retry_timer(self):
        """Stop the online retry timer"""
        if self.online_retry_timer:
            self.online_retry_timer.stop()
            print("[Retry] Stopped online retry timer")
    
    def _try_reconnect_online(self):
        """Try to reconnect to online API"""
        try:
            if not self.auto_switched_offline or not self.is_recording:
                self._stop_online_retry_timer()
                return
            
            # Don't create new STT worker if one already exists
            if self.stt_worker is not None:
                return
            
            print("[Retry] Attempting to reconnect to online API...")
            
            # Get language
            if self.auto_language_mode and self.current_detected_lang:
                lang = self.current_detected_lang
            else:
                lang = self.lang_combo.currentData()
                if lang == "auto":
                    lang = "en"
            
            # Create a new STT worker to try connection
            self.stt_worker = STTWorker(
                api_key=config['api_key'],
                app_id=config['app_id'],
                language=lang,
                domain=config.get('default_domain', 'generic')
            )
            self.stt_worker.transcription.connect(self.on_transcription)
            self.stt_worker.status_changed.connect(self.on_status_changed)
            self.stt_worker.error_signal.connect(self._on_retry_error)
            self.stt_worker.start()
        except Exception as e:
            print(f"[Retry] Error during reconnect: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_retry_error(self, error):
        """Handle error during online retry - just log and keep trying"""
        try:
            print(f"[Retry] Connection failed: {error}")
            if self.stt_worker:
                try:
                    self.stt_worker.stop()
                except:
                    pass
                self.stt_worker = None
            # Timer will try again in 30 seconds
        except Exception as e:
            print(f"[Retry] Error handling retry error: {e}")
        
    def on_stt_error(self, error):
        """Handle online API errors - automatically fallback to offline"""
        try:
            print(f"[STT] Error: {error}")
            print(f"[STT] State: is_recording={self.is_recording}, whisper_worker={self.whisper_worker is not None}, WHISPER_AVAILABLE={WHISPER_AVAILABLE}")
            
            # Stop watchdog since API failed
            self._stop_response_watchdog()
            
            # Check if we should fallback to offline mode
            # Use self.is_recording check but also allow if we just started
            can_fallback = WHISPER_AVAILABLE and self.whisper_worker is None
            
            if can_fallback and self.is_recording:
                self.api_failed = True
                self.auto_switched_offline = True
                self.status_label.setText(f"⚠️ Offline mode (no connection)")
                
                # Update checkbox without triggering the toggle handler
                self.offline_checkbox.blockSignals(True)
                self.offline_checkbox.setChecked(True)
                self.offline_checkbox.blockSignals(False)
                QApplication.processEvents()
                
                # Pre-load model in main thread
                model = WhisperOfflineWorker.preload_model(model_size="tiny", device="cpu")
                if model is None:
                    self.status_label.setText(f"Error: {error} (offline fallback failed)")
                    self.stop_recording()
                    return
                
                # Stop the failed STT worker
                if self.stt_worker:
                    try:
                        self.stt_worker.stop()
                    except:
                        pass
                    self.stt_worker = None
                
                # Get language - handle auto mode
                if self.auto_language_mode and self.current_detected_lang:
                    lang = self.current_detected_lang
                else:
                    lang = self.lang_combo.currentData()
                    if lang == "auto":
                        lang = "en"  # Fallback
                
                self.use_offline_mode = True
                
                self.whisper_worker = WhisperOfflineWorker(
                    model_size="tiny",
                    language=lang,
                    device="cpu",
                    model=model
                )
                self.whisper_worker.transcription.connect(self.on_transcription)
                self.whisper_worker.status_changed.connect(self.on_status_changed)
                self.whisper_worker.error_signal.connect(self.on_whisper_error)
                self.whisper_worker.start()
                
                # Start retry timer to periodically check if online is available
                self._start_online_retry_timer()
                
                print(f"[STT] Switched to offline fallback with lang={lang}")
            else:
                self.status_label.setText(f"Error: {error}")
                self.stop_recording()
        except Exception as e:
            print(f"[STT] Error in error handler: {e}")
            import traceback
            traceback.print_exc()
    
    def _start_response_watchdog(self):
        """Start watchdog timer to detect if online API stops responding"""
        if self.response_watchdog_timer is None:
            self.response_watchdog_timer = QTimer()
            self.response_watchdog_timer.timeout.connect(self._check_response_timeout)
        # Check every second
        self.response_watchdog_timer.start(1000)
        import time
        self.last_online_response_time = time.time()
        print("[Watchdog] Started response watchdog")
    
    def _stop_response_watchdog(self):
        """Stop the response watchdog timer"""
        if self.response_watchdog_timer:
            self.response_watchdog_timer.stop()
            print("[Watchdog] Stopped response watchdog")
    
    def _check_response_timeout(self):
        """Check if we've timed out waiting for online API response"""
        try:
            if not self.is_recording:
                self._stop_response_watchdog()
                return
            
            # Only check if we're in online mode
            if self.use_offline_mode:
                self._stop_response_watchdog()
                return
            
            import time
            current_time = time.time()
            time_since_response = current_time - self.last_online_response_time
            time_since_audio = current_time - self.last_audio_sent_time
            
            # Only trigger offline switch if:
            # 1. We've been sending audio recently (within last 2 seconds) - means audio is active
            # 2. But haven't received a response for watchdog_timeout seconds
            # This prevents switching to offline when video is paused (no audio being sent)
            if time_since_audio < 2.0 and time_since_response > self.watchdog_timeout:
                print(f"[Watchdog] No response for {time_since_response:.1f}s while audio active - switching to offline")
                self._stop_response_watchdog()
                self._auto_switch_to_offline()
            elif time_since_audio >= 2.0:
                # No audio being sent (silence/paused) - reset the response timer
                # so we don't immediately switch to offline when audio resumes
                self.last_online_response_time = current_time
        except Exception as e:
            print(f"[Watchdog] Error in timeout check: {e}")
    
    def _auto_switch_to_offline(self):
        """Automatically switch to offline mode due to no response"""
        try:
            if not WHISPER_AVAILABLE or self.whisper_worker:
                return
            
            self.api_failed = True
            self.auto_switched_offline = True
            self.status_label.setText(f"⚠️ Offline mode (no response)")
            
            # Update checkbox without triggering the toggle handler
            self.offline_checkbox.blockSignals(True)
            self.offline_checkbox.setChecked(True)
            self.offline_checkbox.blockSignals(False)
            QApplication.processEvents()
            
            # Pre-load model in main thread
            model = WhisperOfflineWorker.preload_model(model_size="tiny", device="cpu")
            if model is None:
                self.status_label.setText("Error: offline fallback failed")
                self.stop_recording()
                return
            
            # Stop the online STT worker
            if self.stt_worker:
                try:
                    self.stt_worker.stop()
                except:
                    pass
                self.stt_worker = None
            
            # Get language - handle auto mode
            if self.auto_language_mode and self.current_detected_lang:
                lang = self.current_detected_lang
            else:
                lang = self.lang_combo.currentData()
                if lang == "auto":
                    lang = "en"  # Fallback
            
            self.use_offline_mode = True
            
            self.whisper_worker = WhisperOfflineWorker(
                model_size="tiny",
                language=lang,
                device="cpu",
                model=model
            )
            self.whisper_worker.transcription.connect(self.on_transcription)
            self.whisper_worker.status_changed.connect(self.on_status_changed)
            self.whisper_worker.error_signal.connect(self.on_whisper_error)
            self.whisper_worker.start()
            
            # Start retry timer to periodically check if online is available
            self._start_online_retry_timer()
            
            print(f"[Watchdog] Switched to offline mode with lang={lang}")
        except Exception as e:
            print(f"[Watchdog] Error switching to offline: {e}")
            import traceback
            traceback.print_exc()
    
    def clear_all_captions(self):
        """Clear all captions from both multi-line and ticker displays"""
        self.caption_display.clear()
        self.translation_display.clear()
        self.translation_display_2.clear()  # Clear second translation display
        self.single_line_text = ""
        self.partial_text = ""
        if self.ticker_label:
            self.ticker_label.setText("")
        if self.translation_ticker:
            self.translation_ticker.setText("")
        if self.translation_ticker_2:
            self.translation_ticker_2.setText("")
        
        # Clear caption and translation history
        self.caption_history = []
        self.translation_history = []
        
        # Reset translation tracking (sentence-based state)
        if hasattr(self, '_trans_state'):
            self._trans_state['last_text'] = ""
            self._trans_state['last_trans_time'] = 0
            self._trans_state['last_chunk_time'] = 0
            self._trans_state['completed_sentences'] = ""
            self._trans_state['completed_translations'] = ""
            self._trans_state['current_sentence'] = ""
            self._trans_state['current_translation'] = ""
            self._trans_state['last_sent_text'] = ""  # CRITICAL: Reset to allow new translations
        
        # Reset translation display tracking
        if hasattr(self, '_last_trans_original'):
            self._last_trans_original = ""
        if hasattr(self, '_last_trans_original_2'):
            self._last_trans_original_2 = ""
        if hasattr(self, '_trans_in_progress'):
            self._trans_in_progress = ""
        # Reset scrolling translation lines
        if hasattr(self, '_translation_lines'):
            self._translation_lines = []
        if hasattr(self, '_translation_lines_2'):
            self._translation_lines_2 = []
        # Reset dual mode translation lines
        if hasattr(self, '_dual_trans_lines_1'):
            self._dual_trans_lines_1 = []

            
    def copy_captions(self):
        """Copy all captions and translations to clipboard"""
        # Build full text from history
        output_parts = []
        
        # Add caption history
        if self.caption_history:
            output_parts.append("=== CAPTIONS ===")
            output_parts.append('\n'.join(self.caption_history))
        
        # Add translation history if available
        if self.translation_history:
            if output_parts:
                output_parts.append("\n")
            output_parts.append("=== TRANSLATIONS ===")
            output_parts.append('\n'.join(self.translation_history))
        
        # If no history yet, copy current display
        if not output_parts:
            if self.caption_settings.get('caption_mode', 'multi') == 'single':
                text = self.single_line_text or (self.ticker_label.text() if self.ticker_label else "")
            else:
                text = self.caption_display.toPlainText()
            output_parts.append(text)
        
        full_text = '\n'.join(output_parts)
        QApplication.clipboard().setText(full_text)
        self.status_label.setText("Copied!")
        QTimer.singleShot(1500, lambda: self.status_label.setText("🟢 Connected") if self.is_recording else None)
    
    def nativeEvent(self, eventType, message):
        """Handle Windows native events for resize cursors"""
        try:
            if eventType == b"windows_generic_MSG":
                import ctypes
                from ctypes import wintypes
                
                msg = ctypes.wintypes.MSG.from_address(int(message))
                
                # WM_NCHITTEST = 0x0084
                if msg.message == 0x0084:
                    # Get cursor position
                    x = msg.lParam & 0xFFFF
                    y = (msg.lParam >> 16) & 0xFFFF
                    
                    # Handle signed coordinates
                    if x > 32767:
                        x -= 65536
                    if y > 32767:
                        y -= 65536
                    
                    geo = self.frameGeometry()
                    margin = self.resize_margin
                    
                    left = x - geo.left() < margin
                    right = geo.right() - x < margin
                    top = y - geo.top() < margin
                    bottom = geo.bottom() - y < margin
                    
                    # Return hit test result
                    # HTCLIENT=1, HTCAPTION=2, HTLEFT=10, HTRIGHT=11, HTTOP=12, HTTOPLEFT=13, 
                    # HTTOPRIGHT=14, HTBOTTOM=15, HTBOTTOMLEFT=16, HTBOTTOMRIGHT=17
                    if top and left:
                        return True, 13  # HTTOPLEFT
                    elif top and right:
                        return True, 14  # HTTOPRIGHT
                    elif bottom and left:
                        return True, 16  # HTBOTTOMLEFT
                    elif bottom and right:
                        return True, 17  # HTBOTTOMRIGHT
                    elif left:
                        return True, 10  # HTLEFT
                    elif right:
                        return True, 11  # HTRIGHT
                    elif top:
                        return True, 12  # HTTOP
                    elif bottom:
                        return True, 15  # HTBOTTOM
        except:
            pass
        
        return super().nativeEvent(eventType, message)
    
    def _get_resize_edge(self, pos):
        """Determine which edge/corner the position is near"""
        geo = self.rect()
        margin = self.resize_margin
        
        near_left = pos.x() < margin
        near_right = pos.x() > geo.width() - margin
        near_top = pos.y() < margin
        near_bottom = pos.y() > geo.height() - margin
        
        if near_top and near_left:
            return 'top-left'
        elif near_top and near_right:
            return 'top-right'
        elif near_bottom and near_left:
            return 'bottom-left'
        elif near_bottom and near_right:
            return 'bottom-right'
        elif near_left:
            return 'left'
        elif near_right:
            return 'right'
        elif near_top:
            return 'top'
        elif near_bottom:
            return 'bottom'
        return None
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            edge = self._get_resize_edge(pos)
            
            if edge:
                # Start resizing
                self.resizing = True
                self.resize_edge = edge
                self.resize_start_pos = event.globalPos()
                self.resize_start_geo = self.geometry()
                event.accept()
            else:
                # Start dragging
                self.resizing = False
                self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
                event.accept()
            
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            if self.resizing and self.resize_edge:
                # Perform resize
                delta = event.globalPos() - self.resize_start_pos
                geo = self.resize_start_geo
                new_geo = QRect(geo)
                min_w, min_h = self.minimumWidth(), self.minimumHeight()
                
                edge = self.resize_edge
                if 'left' in edge:
                    new_left = geo.left() + delta.x()
                    new_width = geo.right() - new_left + 1
                    if new_width >= min_w:
                        new_geo.setLeft(new_left)
                if 'right' in edge:
                    new_width = geo.width() + delta.x()
                    if new_width >= min_w:
                        new_geo.setWidth(new_width)
                if 'top' in edge:
                    new_top = geo.top() + delta.y()
                    new_height = geo.bottom() - new_top + 1
                    if new_height >= min_h:
                        new_geo.setTop(new_top)
                if 'bottom' in edge:
                    new_height = geo.height() + delta.y()
                    if new_height >= min_h:
                        new_geo.setHeight(new_height)
                
                self.setGeometry(new_geo)
                event.accept()
            elif self.drag_position:
                # Perform drag
                self.move(event.globalPos() - self.drag_position)
                event.accept()
        else:
            # Update cursor based on position
            edge = self._get_resize_edge(event.pos())
            
            if edge in ('top-left', 'bottom-right'):
                self.setCursor(Qt.SizeFDiagCursor)
            elif edge in ('top-right', 'bottom-left'):
                self.setCursor(Qt.SizeBDiagCursor)
            elif edge in ('left', 'right'):
                self.setCursor(Qt.SizeHorCursor)
            elif edge in ('top', 'bottom'):
                self.setCursor(Qt.SizeVerCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            event.accept()
            
    def mouseReleaseEvent(self, event):
        self.drag_position = None
        self.resizing = False
        self.resize_edge = None
        self.setCursor(Qt.ArrowCursor)
        event.accept()
        
    def closeEvent(self, event):
        self._stop_response_watchdog()
        self.stop_recording()
        # Stop translation worker on app close
        self._stop_translation_worker()
        event.accept()
        # Quit the application completely
        QApplication.quit()
