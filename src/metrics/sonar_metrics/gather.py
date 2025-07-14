import logging
import os

# ✅ Safe import for dotenv (optional in production)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logging.warning("[Sonar] python-dotenv not installed; skipping .env loading")

from metrics.sonar_metrics.scanner import run_sonar
from metrics.sonar_metrics.plugins import load_plugins

# ✅ Dynamically load all sonar plugins once
plugins = load_plugins()

def gather_sonar_metrics(file_path: str) -> dict[str, float]:
    """
    Gathers SonarQube metrics for the specified file path using configured credentials and plugins.

    Args:
        file_path (str): Path to the file or directory to analyse.

    Returns:
        dict[str, float]: Dictionary of extracted SonarQube metric values.
    """
    try:
        logging.debug(f"[SonarQube] Gathering metrics for: {file_path}")

        # ✅ Read environment-based configuration
        sonar_user = os.getenv("SONAR_USER", "")
        sonar_password = os.getenv("SONAR_PASSWORD", "")
        sonar_token = os.getenv("SONAR_TOKEN", "")
        sonar_temp_dir = os.getenv("SONAR_TEMP_DIR", ".sonar_temp")
        coverage_report_path = os.getenv("SONAR_COVERAGE_PATH", "coverage.xml")
        sonar_url = os.getenv("SONAR_URL", "http://localhost:9000")

        # ❌ Ensure token is present
        if not sonar_token:
            logging.error("[SonarQube] Missing required SONAR_TOKEN environment variable")
            return {}

        # ✅ Call scanner
        raw_metrics = run_sonar(
            file_path=file_path,
            sonar_user=sonar_user,
            sonar_password=sonar_password,
            sonar_token=sonar_token,
            sonar_temp_dir=sonar_temp_dir,
            coverage_report_path=coverage_report_path,
            sonar_url=sonar_url
        )

        if not raw_metrics:
            logging.warning("[SonarQube] No metrics returned from scanner.")
            return {}

        # ✅ Apply all plugins
        result = {}
        for plugin in plugins:
            try:
                value = plugin.extract(raw_metrics, file_path)
                result[plugin.name()] = value  # ⬅️ Use raw metric name (no 'sonar.' prefix)
                logging.debug(f"[SonarQube] {plugin.name()} = {value}")
            except Exception as e:
                logging.warning(f"[SonarQube] Plugin '{plugin.name()}' failed: {e}")

        return result

    except Exception as e:
        logging.error(f"[SonarQube] Failed to gather metrics: {e}", exc_info=True)
        return {}
