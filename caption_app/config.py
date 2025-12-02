"""
Configuration loader for the Caption App
"""

import json
from pathlib import Path


def load_config():
    """Load configuration from config.json"""
    config_path = Path(__file__).parent.parent / 'config.json'
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except:
        return {
            "api_key": "",
            "app_id": "",
            "default_language": "hi",
            "default_domain": "generic"
        }


# Global config instance
config = load_config()
