import sys
# 🧠 ML Signal: Importing and configuring a logger

from loguru import logger
# 🧠 ML Signal: Removing default log handlers

# 🧠 ML Signal: Adding a new log handler with specific format and output
# 🧠 ML Signal: Custom log format definition

# Remove default output
logger.remove()


# Add terminal output
fmt: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
logger.add(sys.stdout, colorize=True, format=fmt)