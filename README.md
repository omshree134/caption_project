# Caption Overlay

A desktop application for real-time speech-to-text captions with support for both online (Reverie API) and offline (Whisper) transcription.

## Features

- **Floating overlay** - Transparent, always-on-top caption window
- **Multiple audio sources** - Microphone, System Audio (Speaker), or Both
- **Online mode** - Real-time streaming via Reverie STT API
- **Offline mode** - Local transcription using Whisper (no internet required)
- **Multi-language support** - Hindi, English, Hindi-English mixed, and more

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

3. **Configure API keys** (for online mode)
   ```bash
   cp config.example.json config.json
   # Edit config.json with your Reverie API credentials
   ```

## Usage

```bash
python desktop_app.py
```

### Controls
- **Source**: Select audio input (Mic Only, Speaker Only, Mic + Speaker)
- **Language**: Choose transcription language
- **Offline**: Check to use local Whisper model (no internet needed)
- **Start/Stop**: Toggle captioning

## Requirements

- Python 3.10+
- Windows 10/11 (for system audio capture via WASAPI)
- For offline mode: ~75MB for Whisper tiny model (auto-downloaded on first use)

## Dependencies

- `PyQt5` - Desktop UI framework
- `sounddevice` - Microphone capture
- `pyaudiowpatch` - WASAPI loopback for system audio
- `websockets` - Online API connection
- `faster-whisper` - Offline speech recognition
- `webrtcvad-wheels` - Voice activity detection

## Configuration

Create a `config.json` file with your API credentials:

```json
{
    "api_key": "your_reverie_api_key",
    "app_id": "your_app_id",
    "default_language": "hi",
    "default_domain": "generic"
}
```

> **Note**: Offline mode works without API credentials.

## License

MIT License
