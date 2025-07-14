# ✅ Runs sonar-scanner, fetches raw metrics from SonarQube API
import os
import subprocess
import requests
import logging
from typing import Any

def run_sonar(
    file_path: str,
    sonar_user: str,
    sonar_password: str,
    sonar_token: str,
    sonar_temp_dir: str,
    coverage_report_path: str,
    sonar_url: str
) -> dict[str, Any]:
    """
    Runs SonarQube static analysis and retrieves key metrics via the REST API.

    Args:
        file_path (str): File or directory to analyse.
        sonar_user (str): SonarQube username for API auth.
        sonar_password (str): SonarQube password for API auth.
        sonar_token (str): SonarQube analysis token for CLI auth.
        sonar_temp_dir (str): Temporary directory for scanner output.
        coverage_report_path (str): Path to coverage report.
        sonar_url (str): Base URL of the SonarQube server.

    Returns:
        dict[str, Any]: Dictionary of extracted numeric SonarQube metrics.
    """
    try:
        if os.path.isdir(file_path):
            sonar_sources = file_path
            sonar_project_base_dir = file_path
            sonar_inclusions = "-Dsonar.inclusions=**/*.py"
        elif os.path.isfile(file_path):
            sonar_sources = os.path.dirname(file_path)
            sonar_project_base_dir = sonar_sources
            sonar_inclusions = f"-Dsonar.inclusions={os.path.basename(file_path)}"
        else:
            raise ValueError(f"The provided file_path is not valid: {file_path}")

        sonar_test_inclusions = "-Dsonar.test.inclusions=test,tests,doc,docs"

        scanner_cmd = [
            "sonar-scanner.bat",
            "-Dsonar.projectKey=code_analyser",
            f"-Dsonar.sources={sonar_sources}",
            f"-Dsonar.projectBaseDir={sonar_project_base_dir}",
            sonar_inclusions,
            sonar_test_inclusions,
            "-Dsonar.exclusions=**/.venv/**,**/demos/**",
            f"-Dsonar.host.url={sonar_url}",
            f"-Dsonar.token={sonar_token}",
            f"-Dsonar.scanner.tempDir={sonar_temp_dir}",
            "-Dsonar.python.version=3.12",
            f"-Dsonar.python.coverage.reportPaths={coverage_report_path}",
            "-Dsonar.scm.disabled=true",
            "-Dsonar.sourceEncoding=UTF-8"
        ]

        # Remove any empty arguments
        scanner_cmd = [arg for arg in scanner_cmd if arg]

        logging.debug(f"[SonarQube] Running scanner: {' '.join(scanner_cmd)}")
        subprocess.run(scanner_cmd, check=True)

        # Metrics to extract from the SonarQube API
        metric_keys = ",".join([
            "ncloc", "files", "classes", "duplicated_lines", "duplicated_blocks",
            "duplicated_lines_density", "complexity", "cognitive_complexity", "comment_lines_density",
            "coverage", "tests", "test_success_density", "sqale_index", "sqale_rating", "bugs",
            "reliability_rating", "vulnerabilities", "security_rating", "code_smells"
        ])

        response = requests.get(
            f"{sonar_url}/api/measures/component",
            auth=(sonar_user, sonar_password),
            params={"component": "code_analyser", "metricKeys": metric_keys}
        )
        response.raise_for_status()

        measures = response.json().get("component", {}).get("measures", [])
        metrics: dict[str, Any] = {}

        for m in measures:
            try:
                metrics[m["metric"]] = float(m["value"])
            except (KeyError, ValueError, TypeError):
                metrics[m.get("metric", "unknown")] = 0

        logging.info(f"[SonarQube] Metrics retrieved for {file_path}: {metrics}")
        return metrics

    except subprocess.CalledProcessError as e:
        logging.error(f"[SonarQube] Scanner failed for {file_path}: {e}")
    except requests.RequestException as e:
        logging.error(f"[SonarQube] API request failed for {file_path}: {e}")
    except Exception as e:
        logging.exception(f"[SonarQube] Unexpected error for {file_path}: {type(e).__name__}: {e}")

    return {}
