# Caption Overlay

A desktop application for real-time speech-to-text captions with support for both online (Reverie API) and offline (Whisper) transcription, plus translation to 22+ Indian languages.

## Features

- **Floating overlay** - Transparent, always-on-top caption window
- **Resizable window** - Drag from any edge to resize
- **Multiple audio sources** - Microphone, System Audio (Speaker), or Both
- **Online mode** - Real-time streaming via Reverie STT API
- **Offline mode** - Local transcription using Whisper (no internet required)
- **Multi-language support** - Hindi, English, and more
- **Translation** - Translate captions to 22+ Indian languages using IndicTrans2
- **Customizable appearance** - Font, colors, opacity settings

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/omshree134/caption_project
   cd caption-overlay
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install IndicTrans2 for translation** (optional)

   ```bash
   pip install git+https://github.com/VarunGumma/IndicTransToolkit.git
   ```

4. **Configure API keys** (for online mode)
   ```bash
   cp config.example.json config.json
   # Edit config.json with your Reverie API credentials
   ```

## Usage

```bash
python main.py
```

### Controls

- **Source**: Select audio input (Mic Only, System Audio, Mic + System Audio)
- **Language**: Choose transcription language
- **Offline**: Check to use local Whisper model (no internet needed)
- **Translate**: Enable translation of captions
- **Target Language**: Choose translation target language (22 Indian languages + English)
- **Settings (⚙)**: Customize caption appearance (font, colors, opacity)
- **Start/Stop**: Toggle captioning
- **Opacity slider**: Adjust background transparency

## Project Structure

```
caption-overlay/
├── main.py                    # Application entry point
├── caption_app/               # Main package
│   ├── __init__.py           # Package exports
│   ├── constants.py          # Global constants, Whisper pre-loading
│   ├── config.py             # Configuration loading
│   ├── audio.py              # Audio capture (mic, WASAPI loopback)
│   ├── stt_workers.py        # STT workers (online API, offline Whisper)
│   ├── translation.py        # Translation with IndicTrans2
│   ├── dialogs.py            # UI dialogs (settings)
│   └── main_window.py        # Main overlay window
├── config.json               # Your API credentials (gitignored)
├── config.example.json       # Template for config.json
├── requirements.txt          # Python dependencies
└── README.md
```

## Requirements

- Python 3.10+
- Windows 10/11 (for system audio capture via WASAPI)
- For offline mode: ~75MB for Whisper tiny model (auto-downloaded on first use)
- For translation: ~400MB for IndicTrans2 distilled models (auto-downloaded on first use)

## Dependencies

- `PyQt5` - Desktop UI framework
- `sounddevice` - Microphone capture
- `pyaudiowpatch` - WASAPI loopback for system audio
- `websockets` - Online API connection
- `faster-whisper` - Offline speech recognition
- `webrtcvad-wheels` - Voice activity detection
- `transformers`, `torch` - For IndicTrans2 translation
- `IndicTransToolkit` - Translation preprocessing

## Supported Translation Languages

The app supports translation to/from all 22 scheduled Indian languages:

- Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam
- Punjabi, Odia, Assamese, Urdu, Nepali, Sanskrit, Konkani, Maithili
- Dogri, Santali, Kashmiri, Manipuri, Sindhi, Bodo

## Configuration

Create a `config.json` file with your API credentials:

```json
{
  "api_key": "your_reverie_api_key",
  "app_id": "your_app_id",
  "default_language": "en",
  "default_domain": "generic",
  "translation_enabled": false,
  "translation_target_lang": "en",
  "show_original_text": true
}
```

> **Note**: Offline mode and translation work without API credentials.

## License

MIT License
