import sys
import os
import shutil
import json
import subprocess
import logging
from typing import Any, Dict, List

def _get_radon_executable() -> str:
    """Return absolute path to radon executable, even in frozen mode."""
    if getattr(sys, "frozen", False):
        base_path = getattr(sys, "_MEIPASS", "")
        candidate = os.path.join(base_path, "radon.exe")
        if os.path.isfile(candidate):
            logging.debug(f"[Radon] Using bundled radon at: {candidate}")
            return candidate
    exe_path = shutil.which("radon")
    if exe_path:
        logging.debug(f"[Radon] Using system radon at: {exe_path}")
        return exe_path
    logging.warning("[Radon] radon not found in PATH")
    return "radon"

def run_radon(file_path: str) -> Dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["ANSICON"] = "0"

    if os.path.basename(file_path).startswith("tmp") and "AppData" in file_path:
        logging.warning(f"⚠️ Skipping Radon for temporary file: {file_path}")
        return default_radon_metrics()

    if not os.path.isfile(file_path):
        logging.error(f"❌ Radon file not found: {file_path}")
        return default_radon_metrics()
    if not os.access(file_path, os.R_OK):
        logging.error(f"❌ Radon file not readable: {file_path}")
        return default_radon_metrics()

    radon_exe = _get_radon_executable()

    def run_and_parse(cmd: List[str], label: str) -> Dict[str, Any]:
        try:
            logging.debug(f"[Radon] Running command: {' '.join(cmd)}")
            output = subprocess.check_output(cmd, encoding="utf-8", env=env, stderr=subprocess.DEVNULL)
            return json.loads(output)
        except subprocess.CalledProcessError as e:
            logging.error(f"❌ Radon '{label}' failed for {file_path}: {e}")
        except json.JSONDecodeError:
            logging.error(f"❌ Radon '{label}' returned invalid JSON for {file_path}")
        except FileNotFoundError:
            logging.error(f"❌ Radon executable not found: {cmd[0]}")
        except Exception as e:
            logging.error(f"❌ Radon '{label}' unexpected error: {type(e).__name__}: {e}")
        return {}

    raw_data = run_and_parse([radon_exe, "raw", "-j", file_path], "raw")
    hal_data = run_and_parse([radon_exe, "hal", "-j", file_path], "hal")

    return parse_radon_metrics(file_path, raw_data, hal_data)

def parse_radon_metrics(file_path: str, raw_data: Dict[str, Any], hal_data: Dict[str, Any]) -> Dict[str, Any]:
    file_raw = raw_data.get(file_path, {})
    file_hal = hal_data.get(file_path)
    if file_hal is None and hal_data:
        file_hal = list(hal_data.values())[0]
    file_hal = file_hal or {}

    functions = file_hal.get("functions", [])
    if isinstance(functions, dict):
        func_metrics = list(functions.values())
    elif isinstance(functions, list):
        func_metrics = functions
    else:
        func_metrics = []

    total_functions = len(func_metrics)
    avg_volume = avg_difficulty = avg_effort = 0.0

    if total_functions > 0:
        avg_volume = sum(float(f.get("volume", 0)) for f in func_metrics) / total_functions
        avg_difficulty = sum(float(f.get("difficulty", 0)) for f in func_metrics) / total_functions
        avg_effort = sum(float(f.get("effort", 0)) for f in func_metrics) / total_functions

    logging.info(
        f"📄 File: {file_path}\n"
        f" - Logical Lines: {file_raw.get('lloc', 0)}\n"
        f" - Blank Lines: {file_raw.get('blank', 0)}\n"
        f" - Docstrings: {file_raw.get('multi', 0)}\n"
        f" - Halstead Volume (avg): {avg_volume:.2f}\n"
        f" - Halstead Difficulty (avg): {avg_difficulty:.2f}\n"
        f" - Halstead Effort (avg): {avg_effort:.2f}"
    )

    return {
        "number_of_logical_lines": file_raw.get("lloc", 0),
        "number_of_blank_lines": file_raw.get("blank", 0),
        "number_of_doc_strings": file_raw.get("multi", 0),
        "average_halstead_volume": avg_volume,
        "average_halstead_difficulty": avg_difficulty,
        "average_halstead_effort": avg_effort
    }

def default_radon_metrics() -> Dict[str, float]:
    return {
        "number_of_logical_lines": 0,
        "number_of_blank_lines": 0,
        "number_of_doc_strings": 0,
        "average_halstead_volume": 0.0,
        "average_halstead_difficulty": 0.0,
        "average_halstead_effort": 0.0
    }
