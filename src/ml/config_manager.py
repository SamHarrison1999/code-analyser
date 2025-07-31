# src/ml/config_manager.py

import json
from pathlib import Path

CONFIG_PATH = Path.home() / ".analyser_config.json"

DEFAULTS = {
    "huggingface_token": "",
    "huggingface_repo": "SamH1999/fintech-ai-annotations",
    "smtp_user": "",
    "smtp_pass": "",
    "smtp_host": "smtp.office365.com",
    "smtp_port": 587,
}


def load_config():
    if CONFIG_PATH.exists():
        try:
            return {**DEFAULTS, **json.loads(CONFIG_PATH.read_text(encoding="utf-8"))}
        except Exception:
            return DEFAULTS.copy()
    return DEFAULTS.copy()


def save_config(config: dict):
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
