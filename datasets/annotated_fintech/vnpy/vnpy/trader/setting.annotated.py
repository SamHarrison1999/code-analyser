"""
Global setting of the trading platform.
"""

# ✅ Best Practice: Importing specific functions or classes is preferred for clarity and to avoid namespace pollution.
from logging import CRITICAL
from tzlocal import get_localzone_name

# ✅ Best Practice: Importing specific functions or classes is preferred for clarity and to avoid namespace pollution.

from .utility import load_json


SETTINGS: dict = {
    "font.family": "微软雅黑",
    "font.size": 12,
    "log.active": True,
    "log.level": CRITICAL,
    "log.console": True,
    "log.file": True,
    "email.server": "smtp.qq.com",
    "email.port": 465,
    "email.username": "",
    "email.password": "",
    "email.sender": "",
    "email.receiver": "",
    "datafeed.name": "",
    "datafeed.username": "",
    "datafeed.password": "",
    # 🧠 ML Signal: Usage of timezone settings can indicate localization requirements.
    "database.timezone": get_localzone_name(),
    # ⚠️ SAST Risk (Low): Loading settings from a JSON file can introduce risks if the file is tampered with.
    "database.name": "sqlite",
    "database.database": "database.db",
    "database.host": "",
    "database.port": 0,
    "database.user": "",
    "database.password": "",
}


# Load global setting from json file.
SETTING_FILENAME: str = "vt_setting.json"
SETTINGS.update(load_json(SETTING_FILENAME))
