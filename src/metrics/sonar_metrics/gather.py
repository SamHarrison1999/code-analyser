# File: code_analyser/src/metrics/sonar_metrics/gather.py

import logging
import os
from typing import List

# ✅ Safe import for dotenv (optional in production)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    logging.warning("[SonarQube] python-dotenv not installed; skipping .env loading")

from metrics.sonar_metrics.scanner import run_sonar
from metrics.sonar_metrics.plugins import load_plugins

# ✅ Load all sonar plugins at module scope
plugins = load_plugins()


def _load_sonar_env() -> dict[str, str]:
    return {
        "user": os.getenv("SONAR_USER", ""),
        "password": os.getenv("SONAR_PASSWORD", ""),
        "token": os.getenv("SONAR_TOKEN", ""),
        "temp_dir": os.getenv("SONAR_TEMP_DIR", ".sonar_temp"),
        "coverage_path": os.getenv("SONAR_COVERAGE_PATH", "coverage.xml"),
        "url": os.getenv("SONAR_URL", "http://localhost:9000").rstrip("/"),
    }


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
        config = _load_sonar_env()

        if not config["token"] or not config["url"]:
            logging.error("[SonarQube] Missing SONAR_TOKEN or SONAR_URL in environment")
            return {}

        # ✅ Call scanner
        raw_metrics = run_sonar(
            file_path=file_path,
            sonar_user=config["user"],
            sonar_password=config["password"],
            sonar_token=config["token"],
            sonar_temp_dir=config["temp_dir"],
            coverage_report_path=config["coverage_path"],
            sonar_url=config["url"],
        )

        if not raw_metrics:
            logging.warning("[SonarQube] No metrics returned from Sonar scanner.")
            return {}

        extracted = {}
        for plugin in plugins:
            try:
                value = plugin.extract(raw_metrics, file_path)
                extracted[plugin.name()] = value
                logging.debug(f"[SonarQube] {plugin.name()} = {value}")
            except Exception as e:
                logging.warning(f"[SonarQube] Plugin '{plugin.name()}' failed: {e}")

        return extracted

    except Exception as e:
        logging.error(
            f"[SonarQube] Failed to gather metrics: {type(e).__name__}: {e}",
            exc_info=True,
        )
        return {}


def gather_sonar_metrics_bundle(file_path: str) -> list[dict[str, str | float]]:
    """
    Returns a structured list of Sonar metric bundles including confidence and severity.

    Each bundle contains:
    - metric: str
    - value: float
    - confidence: float
    - severity: str

    Args:
        file_path (str): Path to the file or directory to analyse.

    Returns:
        list[dict[str, str | float]]: List of plugin-based metric bundles.
    """
    try:
        logging.debug(f"[SonarQube] Gathering metric bundles for: {file_path}")
        config = _load_sonar_env()

        if not config["token"] or not config["url"]:
            logging.error("[SonarQube] Missing SONAR_TOKEN or SONAR_URL in environment")
            return []

        raw_metrics = run_sonar(
            file_path=file_path,
            sonar_user=config["user"],
            sonar_password=config["password"],
            sonar_token=config["token"],
            sonar_temp_dir=config["temp_dir"],
            coverage_report_path=config["coverage_path"],
            sonar_url=config["url"],
        )

        if not raw_metrics:
            logging.warning("[SonarQube] No metrics returned for bundle extraction.")
            return []

        bundle = []
        for plugin in plugins:
            try:
                value = plugin.extract(raw_metrics, file_path)
                confidence = round(plugin.confidence_score(raw_metrics), 2)
                severity = plugin.severity_level(raw_metrics)
                bundle.append(
                    {
                        "metric": plugin.name(),
                        "value": value,
                        "confidence": confidence,
                        "severity": severity,
                    }
                )
                logging.debug(
                    f"[SonarQube] {plugin.name()} bundle: value={value}, confidence={confidence}, severity={severity}"
                )
            except Exception as e:
                logging.warning(
                    f"[SonarQube] Plugin '{plugin.name()}' bundle failed: {e}"
                )
                bundle.append(
                    {
                        "metric": plugin.name(),
                        "value": 0.0,
                        "confidence": 0.0,
                        "severity": "low",
                    }
                )

        return bundle

    except Exception as e:
        logging.exception(
            f"[SonarQube] Failed to gather metric bundle for {file_path}: {e}"
        )
        return []


def get_sonar_metric_names() -> List[str]:
    """
    Returns the names of all registered SonarQube plugin metrics in extraction order.

    Returns:
        List[str]: Ordered list of metric names.
    """
    return [plugin.name() for plugin in load_plugins()]
