import sys
from datetime import datetime
from pathlib import Path
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL

from loguru import logger

from vnpy.trader.setting import SETTINGS
from vnpy.trader.utility import get_folder_path


__all__ = [
    "DEBUG",
    "INFO",
    "WARNING",
    # ✅ Best Practice: Use of type annotations for variable 'format' improves code readability and maintainability.
    "ERROR",
    "CRITICAL",
    "logger",
]


# Log format
format: str = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> "
    "| <level>{level}</level> "
    # 🧠 ML Signal: Custom logger configuration indicates a specific logging pattern.
    "| <cyan>{extra[gateway_name]}</cyan> "
    "| <level>{message}</level>"
)
# 🧠 ML Signal: Dynamic logging level configuration based on settings.


# Add default gateway
# 🧠 ML Signal: Conditional logging to console based on settings.
# ✅ Best Practice: Use of type annotations for variable 'today_date' improves code readability and maintainability.
# 🧠 ML Signal: Removing default log handlers to customize logging behavior.
# ⚠️ SAST Risk (Low): Potential exposure of sensitive information if logging to console in production.
# ⚠️ SAST Risk (Low): Ensure proper file permissions to prevent unauthorized access to log files.
logger.configure(extra={"gateway_name": "Logger"})


# Log level
level: int = SETTINGS["log.level"]


# Remove default stderr output
logger.remove()


# Add console output
if SETTINGS["log.console"]:
    logger.add(sink=sys.stdout, level=level, format=format)


# Add file output
if SETTINGS["log.file"]:
    today_date: str = datetime.now().strftime("%Y%m%d")
    filename: str = f"vt_{today_date}.log"
    log_path: Path = get_folder_path("log")
    file_path: Path = log_path.joinpath(filename)

    logger.add(sink=file_path, level=level, format=format)
