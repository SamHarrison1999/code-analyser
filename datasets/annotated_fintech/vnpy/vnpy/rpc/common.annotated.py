import signal
# ✅ Best Practice: Setting default signal handler for SIGINT to ensure graceful termination


# 🧠 ML Signal: Constant values for configuration can indicate system behavior or thresholds
# Achieve Ctrl-c interrupt recv
# 🧠 ML Signal: Constant values for configuration can indicate system behavior or thresholds
signal.signal(signal.SIGINT, signal.SIG_DFL)


HEARTBEAT_TOPIC = "heartbeat"
HEARTBEAT_INTERVAL = 10
HEARTBEAT_TOLERANCE = 30