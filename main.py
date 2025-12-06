#!/usr/bin/env python3
"""
Live Captions - Desktop Application
Real-time speech-to-text captioning with online and offline modes.

Usage:
    python main.py
"""

import sys
def exception_hook(exctype, value, tb):
    """Global exception hook to catch unhandled exceptions"""
    import traceback
    print("=" * 50)
    print("UNHANDLED EXCEPTION:")
    print("=" * 50)
    traceback.print_exception(exctype, value, tb)
    print("=" * 50)
    sys.__excepthook__(exctype, value, tb)


def main():
    # Set up global exception hook
    sys.excepthook = exception_hook
    
    # Import constants first (loads Whisper model BEFORE Qt)
    from caption_app.constants import WHISPER_AVAILABLE, get_whisper_model
    
    # Note: Whisper model is pre-loaded at module import time (before Qt)
    if WHISPER_AVAILABLE:
        model = get_whisper_model()
        if model is not None:
            print("[Whisper] Using pre-loaded model")
        else:
            print("[Whisper] Warning: Model not pre-loaded, offline mode may not work")
    
    # Now import PyQt5 (after Whisper model is loaded)
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtGui import QPalette, QColor
    
    # Import main window
    from caption_app.main_window import CaptionOverlay
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(15, 23, 42))
    palette.setColor(QPalette.WindowText, QColor(248, 250, 252))
    palette.setColor(QPalette.Base, QColor(30, 41, 59))
    palette.setColor(QPalette.Text, QColor(248, 250, 252))
    palette.setColor(QPalette.Button, QColor(51, 65, 85))
    palette.setColor(QPalette.ButtonText, QColor(248, 250, 252))
    palette.setColor(QPalette.Highlight, QColor(99, 102, 241))
    app.setPalette(palette)
    
    window = CaptionOverlay()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()