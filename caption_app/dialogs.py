"""
UI Dialogs for the Caption App
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QFontComboBox, QComboBox, QColorDialog, QGroupBox,
    QFormLayout, QSlider, QCheckBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont

from .constants import TRANSLATION_LANGUAGES


class CaptionSettingsDialog(QDialog):
    """Dialog for customizing caption appearance"""
    
    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.settings = settings or {}
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Caption Settings")
        self.setFixedWidth(400)
        self.setStyleSheet("""
            QDialog {
                background-color: #1e293b;
                color: #f8fafc;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #475569;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 10px;
                color: #818cf8;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLabel {
                color: #94a3b8;
            }
            QSpinBox, QFontComboBox, QComboBox {
                background-color: #334155;
                color: white;
                border: 1px solid #475569;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton {
                background-color: #475569;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #64748b;
            }
            QSlider::groove:horizontal {
                background: #475569;
                height: 4px;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #6366f1;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Caption Display Mode Group
        mode_group = QGroupBox("Caption Display Mode")
        mode_layout = QFormLayout(mode_group)
        
        self.caption_mode_combo = QComboBox()
        self.caption_mode_combo.addItem("Multi Line (Scrollable History)", "multi")
        self.caption_mode_combo.addItem("Single Line (Live Ticker)", "single")
        current_mode = self.settings.get('caption_mode', 'multi')
        mode_index = self.caption_mode_combo.findData(current_mode)
        if mode_index >= 0:
            self.caption_mode_combo.setCurrentIndex(mode_index)
        mode_layout.addRow("Display Mode:", self.caption_mode_combo)
        
        layout.addWidget(mode_group)
        
        # Dual Captioning Group
        dual_group = QGroupBox("Dual Language Translation")
        dual_layout = QFormLayout(dual_group)
        
        # Dual captioning checkbox
        self.dual_caption_checkbox = QCheckBox("Enable dual language translation")
        self.dual_caption_checkbox.setChecked(self.settings.get('dual_captioning_enabled', False))
        self.dual_caption_checkbox.setToolTip("Show translations in two languages simultaneously")
        self.dual_caption_checkbox.toggled.connect(self._on_dual_toggled)
        dual_layout.addRow(self.dual_caption_checkbox)
        
        # Second language selector
        self.dual_lang_combo = QComboBox()
        self.dual_lang_combo.setMaxVisibleItems(5)  # Scrollable dropdown
        for code, name in TRANSLATION_LANGUAGES.items():
            self.dual_lang_combo.addItem(name, code)
        # Set current second language
        current_lang_2 = self.settings.get('translation_target_lang_2', 'hi')
        lang_idx = self.dual_lang_combo.findData(current_lang_2)
        if lang_idx >= 0:
            self.dual_lang_combo.setCurrentIndex(lang_idx)
        self.dual_lang_combo.setEnabled(self.dual_caption_checkbox.isChecked())
        dual_layout.addRow("Second Language:", self.dual_lang_combo)
        
        layout.addWidget(dual_group)
        
        # Font Settings Group
        font_group = QGroupBox("Font Settings")
        font_layout = QFormLayout(font_group)
        
        # Font family
        self.font_combo = QFontComboBox()
        self.font_combo.setCurrentFont(QFont(self.settings.get('font_family', 'Segoe UI')))
        font_layout.addRow("Font:", self.font_combo)
        
        # Font size (base size - will be scaled by zoom level)
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(10, 72)
        self.font_size_spin.setValue(self.settings.get('font_size', 18))
        self.font_size_spin.setSuffix(" px")
        self.font_size_spin.setToolTip("Base font size (scales with zoom +/- buttons)")
        font_layout.addRow("Base Size:", self.font_size_spin)
        
        # Font weight
        self.font_weight_combo = QComboBox()
        self.font_weight_combo.addItems(["Normal", "Bold"])
        self.font_weight_combo.setCurrentText(self.settings.get('font_weight', 'Normal'))
        font_layout.addRow("Weight:", self.font_weight_combo)
        
        layout.addWidget(font_group)
        
        # Text Color Group
        color_group = QGroupBox("Text Color")
        color_layout = QFormLayout(color_group)
        
        # Text color picker
        text_color_row = QHBoxLayout()
        self.text_color = QColor(self.settings.get('text_color', '#f8fafc'))
        self.text_color_btn = QPushButton()
        self.text_color_btn.setFixedSize(60, 30)
        self.update_color_button(self.text_color_btn, self.text_color)
        self.text_color_btn.clicked.connect(self.pick_text_color)
        text_color_row.addWidget(self.text_color_btn)
        text_color_row.addStretch()
        color_layout.addRow("Color:", text_color_row)
        
        # Text opacity
        text_opacity_row = QHBoxLayout()
        self.text_opacity_slider = QSlider(Qt.Horizontal)
        self.text_opacity_slider.setRange(0, 100)
        self.text_opacity_slider.setValue(self.settings.get('text_opacity', 100))
        self.text_opacity_label = QLabel(f"{self.text_opacity_slider.value()}%")
        self.text_opacity_slider.valueChanged.connect(lambda v: self.text_opacity_label.setText(f"{v}%"))
        text_opacity_row.addWidget(self.text_opacity_slider)
        text_opacity_row.addWidget(self.text_opacity_label)
        color_layout.addRow("Opacity:", text_opacity_row)
        
        layout.addWidget(color_group)
        
        # Caption Background Group
        bg_group = QGroupBox("Caption Background")
        bg_layout = QFormLayout(bg_group)
        
        # Background color picker
        bg_color_row = QHBoxLayout()
        self.bg_color = QColor(self.settings.get('bg_color', '#1e293b'))
        self.bg_color_btn = QPushButton()
        self.bg_color_btn.setFixedSize(60, 30)
        self.update_color_button(self.bg_color_btn, self.bg_color)
        self.bg_color_btn.clicked.connect(self.pick_bg_color)
        bg_color_row.addWidget(self.bg_color_btn)
        bg_color_row.addStretch()
        bg_layout.addRow("Color:", bg_color_row)
        
        # Background opacity
        bg_opacity_row = QHBoxLayout()
        self.bg_opacity_slider = QSlider(Qt.Horizontal)
        self.bg_opacity_slider.setRange(0, 100)
        self.bg_opacity_slider.setValue(self.settings.get('bg_opacity', 80))
        self.bg_opacity_label = QLabel(f"{self.bg_opacity_slider.value()}%")
        self.bg_opacity_slider.valueChanged.connect(lambda v: self.bg_opacity_label.setText(f"{v}%"))
        bg_opacity_row.addWidget(self.bg_opacity_slider)
        bg_opacity_row.addWidget(self.bg_opacity_label)
        bg_layout.addRow("Opacity:", bg_opacity_row)
        
        layout.addWidget(bg_group)
        
        # Border Settings Group
        border_group = QGroupBox("Border")
        border_layout = QFormLayout(border_group)
        
        # Border color
        border_color_row = QHBoxLayout()
        self.border_color = QColor(self.settings.get('border_color', '#475569'))
        self.border_color_btn = QPushButton()
        self.border_color_btn.setFixedSize(60, 30)
        self.update_color_button(self.border_color_btn, self.border_color)
        self.border_color_btn.clicked.connect(self.pick_border_color)
        border_color_row.addWidget(self.border_color_btn)
        border_color_row.addStretch()
        border_layout.addRow("Color:", border_color_row)
        
        # Border width
        self.border_width_spin = QSpinBox()
        self.border_width_spin.setRange(0, 5)
        self.border_width_spin.setValue(self.settings.get('border_width', 1))
        self.border_width_spin.setSuffix(" px")
        border_layout.addRow("Width:", self.border_width_spin)
        
        layout.addWidget(border_group)
        
        # Preview
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_label = QLabel("This is how your captions will look...")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(60)
        self.preview_label.setWordWrap(True)
        self.update_preview()
        preview_layout.addWidget(self.preview_label)
        layout.addWidget(preview_group)
        
        # Connect all controls to preview update
        self.font_combo.currentFontChanged.connect(self.update_preview)
        self.font_size_spin.valueChanged.connect(self.update_preview)
        self.font_weight_combo.currentTextChanged.connect(self.update_preview)
        self.text_opacity_slider.valueChanged.connect(self.update_preview)
        self.bg_opacity_slider.valueChanged.connect(self.update_preview)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        reset_btn = QPushButton("Reset to Default")
        reset_btn.clicked.connect(self.reset_defaults)
        button_layout.addWidget(reset_btn)
        
        button_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        apply_btn = QPushButton("Apply")
        apply_btn.setStyleSheet("background-color: #6366f1;")
        apply_btn.clicked.connect(self.accept)
        button_layout.addWidget(apply_btn)
        
        layout.addLayout(button_layout)
    
    def update_color_button(self, btn, color):
        """Update button background to show selected color"""
        btn.setStyleSheet(f"background-color: {color.name()}; border: 2px solid #475569; border-radius: 4px;")
    
    def pick_text_color(self):
        color = QColorDialog.getColor(self.text_color, self, "Select Text Color")
        if color.isValid():
            self.text_color = color
            self.update_color_button(self.text_color_btn, color)
            self.update_preview()
    
    def pick_bg_color(self):
        color = QColorDialog.getColor(self.bg_color, self, "Select Background Color")
        if color.isValid():
            self.bg_color = color
            self.update_color_button(self.bg_color_btn, color)
            self.update_preview()
    
    def pick_border_color(self):
        color = QColorDialog.getColor(self.border_color, self, "Select Border Color")
        if color.isValid():
            self.border_color = color
            self.update_color_button(self.border_color_btn, color)
            self.update_preview()
    
    def update_preview(self):
        """Update the preview label with current settings"""
        font_family = self.font_combo.currentFont().family()
        font_size = self.font_size_spin.value()
        font_weight = "bold" if self.font_weight_combo.currentText() == "Bold" else "normal"
        text_opacity = self.text_opacity_slider.value() / 100.0
        bg_opacity = self.bg_opacity_slider.value() / 100.0
        
        text_r, text_g, text_b = self.text_color.red(), self.text_color.green(), self.text_color.blue()
        bg_r, bg_g, bg_b = self.bg_color.red(), self.bg_color.green(), self.bg_color.blue()
        border_color = self.border_color.name()
        border_width = self.border_width_spin.value()
        
        self.preview_label.setStyleSheet(f"""
            background-color: rgba({bg_r}, {bg_g}, {bg_b}, {bg_opacity});
            color: rgba({text_r}, {text_g}, {text_b}, {text_opacity});
            font-family: '{font_family}';
            font-size: {font_size}px;
            font-weight: {font_weight};
            border: {border_width}px solid {border_color};
            border-radius: 8px;
            padding: 10px;
        """)
    
    def reset_defaults(self):
        """Reset all settings to defaults"""
        self.caption_mode_combo.setCurrentIndex(0)  # Multi line
        self.dual_caption_checkbox.setChecked(False)
        self.dual_lang_combo.setCurrentIndex(self.dual_lang_combo.findData('hi'))
        self.font_combo.setCurrentFont(QFont('Segoe UI'))
        self.font_size_spin.setValue(18)
        self.font_weight_combo.setCurrentText('Normal')
        self.text_color = QColor('#f8fafc')
        self.update_color_button(self.text_color_btn, self.text_color)
        self.text_opacity_slider.setValue(100)
        self.bg_color = QColor('#1e293b')
        self.update_color_button(self.bg_color_btn, self.bg_color)
        self.bg_opacity_slider.setValue(80)
        self.border_color = QColor('#475569')
        self.update_color_button(self.border_color_btn, self.border_color)
        self.border_width_spin.setValue(1)
        self.update_preview()
    
    def _on_dual_toggled(self, checked):
        """Enable/disable second language combo based on dual checkbox"""
        self.dual_lang_combo.setEnabled(checked)
    
    def get_settings(self):
        """Return current settings as dictionary"""
        return {
            'caption_mode': self.caption_mode_combo.currentData(),
            'dual_captioning_enabled': self.dual_caption_checkbox.isChecked(),
            'translation_target_lang_2': self.dual_lang_combo.currentData(),
            'font_family': self.font_combo.currentFont().family(),
            'font_size': self.font_size_spin.value(),
            'font_weight': self.font_weight_combo.currentText(),
            'text_color': self.text_color.name(),
            'text_opacity': self.text_opacity_slider.value(),
            'bg_color': self.bg_color.name(),
            'bg_opacity': self.bg_opacity_slider.value(),
            'border_color': self.border_color.name(),
            'border_width': self.border_width_spin.value(),
        }
