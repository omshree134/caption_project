"""
Main overlay window for displaying captions
"""

import sys
import socket
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSlider, QFrame,
    QTextEdit, QApplication, QCheckBox, QMessageBox,
    QDialog, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer
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
        
        # Resize handling - uses Windows native hit testing
        self.resize_margin = 8  # Pixels from edge to trigger resize
        
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
        
        translation_row.addWidget(QLabel("📝"))
        
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
        """)
        self.translate_lang_combo.setToolTip("Translate to this language")
        # Add all translation languages
        for code, name in TRANSLATION_LANGUAGES.items():
            self.translate_lang_combo.addItem(name, code)
        # Set English as default target
        en_idx = self.translate_lang_combo.findData("en")
        if en_idx >= 0:
            self.translate_lang_combo.setCurrentIndex(en_idx)
        self.translate_lang_combo.setEnabled(False)  # Disabled until translation is enabled
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
        
        translation_row.addStretch()
        
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
        
        # Caption display area (multi-line mode)
        self.caption_display = QTextEdit()
        self.caption_display.setReadOnly(True)
        self.caption_display.setPlaceholderText("Captions will appear here...")
        container_layout.addWidget(self.caption_display)
        
        # Ticker-style caption (single-line mode) - hidden by default
        # Simple label that shows the latest caption text cleanly
        self.ticker_label = QLabel("")
        self.ticker_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.ticker_label.setWordWrap(False)  # Single line only
        self.ticker_label.setTextFormat(Qt.PlainText)
        self.ticker_label.hide()  # Hidden by default
        container_layout.addWidget(self.ticker_label)
        
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
            self.clear_btn, self.copy_btn
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
        dialog = CaptionSettingsDialog(self, self.caption_settings)
        if dialog.exec_() == QDialog.Accepted:
            self.caption_settings = dialog.get_settings()
            self.apply_caption_settings()
    
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
        
        # Handle single-line vs multi-line mode
        is_single_line = s.get('caption_mode', 'multi') == 'single'
        if is_single_line:
            # Save current geometry before switching to single-line
            if self.multi_line_geometry is None:
                self.multi_line_geometry = self.geometry()
            
            # Hide multi-line display, show ticker
            self.caption_display.hide()
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
            
            # Calculate compact height for single line
            line_height = s['font_size'] + 35  # font size + padding + border
            self.ticker_label.setFixedHeight(line_height)
            
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
            self.caption_display.show()
            
            # Reset height constraints
            self.caption_display.setMaximumHeight(16777215)  # Qt's default max
            self.caption_display.setMinimumHeight(60)
            self.caption_display.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            
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
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
            
    def start_recording(self):
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
        
    def on_offline_toggled(self, checked):
        """Handle offline checkbox toggle - allows live switching during recording"""
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
        self.translation_enabled = checked
        self.translate_lang_combo.setEnabled(checked)
        self.show_original_checkbox.setEnabled(checked)
        
        if checked:
            # Initialize translation worker if not already running
            if self.translation_worker is None or not self.translation_worker.isRunning():
                target_lang = self.translate_lang_combo.currentData()
                self._start_translation_worker(target_lang)
            elif self.translation_worker and not self.translation_worker.is_ready():
                # Worker is running but models not loaded yet - just wait
                self.status_label.setText("🔄 Translation models loading...")
        else:
            # Don't stop the worker - just disable adding to queue
            # This way, if user toggles on again, models are already loaded
            pass
    
    def on_show_original_toggled(self, checked):
        """Handle show original text checkbox toggle"""
        # Update config
        config['show_original_text'] = checked
    
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
        if self.translation_worker is not None and self.translation_worker.isRunning():
            # Worker already running, just update target language
            self.translation_worker.set_target_language(target_lang)
            return
        
        # Clean up old worker if exists but not running
        if self.translation_worker is not None:
            self.translation_worker = None
        
        print(f"[Translation] Starting worker with target: {target_lang}")
        self.status_label.setText("🔄 Loading translation model...")
        
        self.translation_worker = TranslationWorker(
            tgt_lang=target_lang,
            device="cpu"
        )
        self.translation_worker.translation_ready.connect(self.on_translation_ready)
        self.translation_worker.model_loaded.connect(self.on_translation_model_loaded)
        self.translation_worker.error_signal.connect(self.on_translation_error)
        self.translation_worker.loading_started.connect(self.on_translation_loading_started)
        self.translation_worker.start()
    
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
    
    def on_translation_model_loaded(self, model_name):
        """Handle translation model loaded signal"""
        print(f"[Translation] Model loaded: {model_name}")
        if self.is_recording:
            self.status_label.setText("🎙️ Recording (translation ready)")
        else:
            self.status_label.setText("Translation ready")
    
    def on_translation_ready(self, original_text, translated_text):
        """Handle translated text from translation worker"""
        print(f"[Translation] {original_text} → {translated_text}")
        
        # Determine what to display based on settings
        show_original = config.get('show_original_text', False)
        
        if show_original:
            display_text = f"{original_text}\n→ {translated_text}"
        else:
            display_text = translated_text
        
        # Use the same display logic as regular transcription
        self._display_transcription(display_text, is_final=True, cause="")
    
    def on_translation_error(self, error_msg):
        """Handle translation error"""
        print(f"[Translation Error] {error_msg}")
        self.status_label.setText(f"⚠️ Translation: {error_msg[:30]}...")
        
    def stop_recording(self):
        """Stop recording - use non-blocking cleanup to prevent UI freeze"""
        print("[DEBUG] stop_recording called")
        
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
            
        print("[DEBUG] Workers cleaned up")
        
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
        
        # Display original text first (before translation)
        # This gives immediate visual feedback while translation processes
        if not self.translation_enabled:
            # No translation - display directly
            self._display_transcription(text, is_final, cause)
            return
        
        # Translation is enabled
        if self.translation_worker and self.translation_worker.is_ready():
            # Determine source language from detection or selection
            if self.auto_language_mode and self.current_detected_lang:
                src_lang = self.current_detected_lang
            else:
                src_lang = self.lang_combo.currentData()
                if src_lang == "auto":
                    src_lang = "en"  # Default fallback
            
            # Optimized translation throttling:
            # - Final results: always translate
            # - Partial results: throttle by time AND minimum chunk size
            should_translate = False
            import time
            current_time = time.time()
            
            if not hasattr(self, '_last_translation_time'):
                self._last_translation_time = 0
                self._last_translation_text = ""
                self._translation_char_buffer = ""
            
            if is_final:
                # Final results always get translated
                should_translate = True
                self._translation_char_buffer = ""  # Reset buffer
            else:
                # For partials: translate more frequently for responsive feel
                text_len = len(text.strip())
                last_len = len(self._last_translation_text)
                time_since_last = current_time - self._last_translation_time
                
                # Translate partials when:
                # 1. New text (15+ chars since last translation) OR
                # 2. Time threshold (1.5 seconds) passed with any new content OR
                # 3. First partial (no previous translation)
                new_chars = text_len - last_len
                
                if self._last_translation_time == 0:
                    # First translation - start quickly
                    should_translate = text_len >= 5
                elif new_chars >= 15:
                    # Moderate new content
                    should_translate = True
                elif time_since_last >= 1.5 and new_chars >= 3:
                    # Time threshold with some new content
                    should_translate = True
            
            if should_translate:
                self._last_translation_time = current_time
                self._last_translation_text = text
                # Clear older queue items since we have newer text
                self.translation_worker.clear_queue()
                # Send to translation worker with source language
                self.translation_worker.add_text(text.strip(), src_lang)
            
            # Always show original text while waiting for translation
            if config.get('show_original_text', False):
                self._display_transcription(text, is_final, cause)
        else:
            # Translation enabled but worker not ready yet
            # Show original text with indicator
            self._display_transcription(text, is_final, cause)
            if self.translation_worker and not self.translation_worker.is_ready():
                self.status_label.setText("🔄 Translation loading...")
    
    def _display_transcription(self, text, is_final, cause=""):
        """Display transcription text in the appropriate mode"""
        is_single_line = self.caption_settings.get('caption_mode', 'multi') == 'single'
        
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
        if not self.auto_switched_offline or not self.is_recording:
            return
        
        print("[Auto] API reconnected - switching back to online mode...")
        
        # Stop the retry timer
        self._stop_online_retry_timer()
        
        # Stop Whisper worker
        if self.whisper_worker:
            self.whisper_worker.stop()
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
        if not self.auto_switched_offline or not self.is_recording:
            self._stop_online_retry_timer()
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
    
    def _on_retry_error(self, error):
        """Handle error during online retry - just log and keep trying"""
        print(f"[Retry] Connection failed: {error}")
        if self.stt_worker:
            self.stt_worker.stop()
            self.stt_worker = None
        # Timer will try again in 30 seconds
        
    def on_stt_error(self, error):
        """Handle online API errors - automatically fallback to offline"""
        print(f"[STT] Error: {error}")
        
        # Stop watchdog since API failed
        self._stop_response_watchdog()
        
        # Check if we should fallback to offline mode
        if WHISPER_AVAILABLE and not self.whisper_worker and self.is_recording:
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
                self.stt_worker.stop()
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
    
    def _auto_switch_to_offline(self):
        """Automatically switch to offline mode due to no response"""
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
            self.stt_worker.stop()
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
    
    def clear_all_captions(self):
        """Clear all captions from both multi-line and ticker displays"""
        self.caption_display.clear()
        self.single_line_text = ""
        self.partial_text = ""
        if self.ticker_label:
            self.ticker_label.setText("")
            
    def copy_captions(self):
        # Copy from whichever display is active
        if self.caption_settings.get('caption_mode', 'multi') == 'single':
            text = self.single_line_text or self.ticker_label.text() if self.ticker_label else ""
        else:
            text = self.caption_display.toPlainText()
        QApplication.clipboard().setText(text)
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
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
            
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.drag_position:
            self.move(event.globalPos() - self.drag_position)
            event.accept()
            
    def mouseReleaseEvent(self, event):
        self.drag_position = None
        event.accept()
        
    def closeEvent(self, event):
        self._stop_response_watchdog()
        self.stop_recording()
        # Stop translation worker on app close
        self._stop_translation_worker()
        event.accept()
        # Quit the application completely
        QApplication.quit()
