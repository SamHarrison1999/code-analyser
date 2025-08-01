# File: code_analyser/src/metrics/sonar_metrics/scanner.py

import os
import subprocess
import requests
import logging
import platform
from typing import Any, List

# ✅ Automatically choose the correct sonar-scanner binary
SCANNER_BINARY = (
    "sonar-scanner.bat" if platform.system() == "Windows" else "sonar-scanner"
)


def run_sonar(
    file_path: str,
    sonar_user: str,
    sonar_password: str,
    sonar_token: str,
    sonar_temp_dir: str,
    coverage_report_path: str,
    sonar_url: str,
) -> dict[str, Any]:
    """
    Runs SonarQube static analysis and retrieves key metrics via the REST API.

    Args:
        file_path (str): File or directory to analyse.
        sonar_user (str): SonarQube username for API authentication.
        sonar_password (str): SonarQube password for API authentication.
        sonar_token (str): SonarQube token for scanner authentication.
        sonar_temp_dir (str): Temporary directory used by sonar-scanner.
        coverage_report_path (str): Path to coverage report file.
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
            raise ValueError(f"[SonarQube] Invalid file_path: {file_path}")

        scanner_cmd = [
            SCANNER_BINARY,
            "-Dsonar.projectKey=code_analyser",
            f"-Dsonar.sources={sonar_sources}",
            f"-Dsonar.projectBaseDir={sonar_project_base_dir}",
            sonar_inclusions,
            "-Dsonar.test.inclusions=test,tests,doc,docs",
            "-Dsonar.exclusions=**/.venv/**,**/demos/**",
            f"-Dsonar.host.url={sonar_url}",
            f"-Dsonar.token={sonar_token}",
            f"-Dsonar.scanner.tempDir={sonar_temp_dir}",
            "-Dsonar.python.version=3.12",
            f"-Dsonar.python.coverage.reportPaths={coverage_report_path}",
            "-Dsonar.scm.disabled=true",
            "-Dsonar.sourceEncoding=UTF-8",
        ]
        scanner_cmd = [arg for arg in scanner_cmd if arg.strip()]
        logging.debug(f"[SonarQube] Executing: {' '.join(scanner_cmd)}")
        subprocess.run(scanner_cmd, check=True)

        # Metrics to extract from the SonarQube API
        metric_keys = ",".join(
            [
                "ncloc",
                "files",
                "classes",
                "duplicated_lines",
                "duplicated_blocks",
                "duplicated_lines_density",
                "complexity",
                "cognitive_complexity",
                "comment_lines_density",
                "coverage",
                "tests",
                "test_success_density",
                "sqale_index",
                "sqale_rating",
                "bugs",
                "reliability_rating",
                "vulnerabilities",
                "security_rating",
                "code_smells",
            ]
        )

        response = requests.get(
            f"{sonar_url}/api/measures/component",
            auth=(sonar_user, sonar_password),
            params={"component": "code_analyser", "metricKeys": metric_keys},
        )
        response.raise_for_status()

        measures = response.json().get("component", {}).get("measures", [])
        metrics: dict[str, Any] = {}
        for m in measures:
            try:
                metrics[m["metric"]] = float(m["value"])
            except (KeyError, ValueError, TypeError):
                logging.warning(
                    f"[SonarQube] Could not parse metric {m.get('metric', 'unknown')}"
                )
                metrics[m.get("metric", "unknown")] = 0.0

        logging.info(f"[SonarQube] Metrics retrieved for {file_path}: {metrics}")
        return metrics

    except subprocess.CalledProcessError as e:
        logging.error(f"[SonarQube] Scanner failed for {file_path}: {e}")
    except requests.RequestException as e:
        logging.error(f"[SonarQube] API request failed: {e}")
    except Exception as e:
        logging.exception(f"[SonarQube] Unexpected error: {type(e).__name__}: {e}")

    return {}


def batch_run_sonar(
    paths: List[str],
    sonar_user: str,
    sonar_password: str,
    sonar_token: str,
    sonar_temp_dir: str,
    coverage_report_path: str,
    sonar_url: str,
) -> dict[str, dict[str, Any]]:
    """
    Runs Sonar analysis for a batch of paths and returns a mapping of path -> metrics.

    Args:
        file_path (str): File or directory to analyse.
        sonar_user (str): SonarQube username for API authentication.
        sonar_password (str): SonarQube password for API authentication.
        sonar_token (str): SonarQube token for scanner authentication.
        sonar_temp_dir (str): Temporary directory used by sonar-scanner.
        coverage_report_path (str): Path to coverage report file.
        sonar_url (str): Base URL of the SonarQube server.

    Returns:
        dict[str, Any]: Dictionary of extracted numeric SonarQube metrics.
    """
    all_results = {}
    for path in paths:
        logging.info(f"[SonarQube] Processing {path}")
        metrics = run_sonar(
            file_path=path,
            sonar_user=sonar_user,
            sonar_password=sonar_password,
            sonar_token=sonar_token,
            sonar_temp_dir=sonar_temp_dir,
            coverage_report_path=coverage_report_path,
            sonar_url=sonar_url,
        )
        all_results[path] = metrics
    return all_results
