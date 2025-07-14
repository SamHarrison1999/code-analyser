# File: tests/test_fail_if_nonzero.py

import subprocess
import sys
import os
from pathlib import Path


def test_fail_if_nonzero_triggers_exit(tmp_path):
    """
    Runs the CLI with --fail-threshold 0 on a file with non-zero metrics.
    Expects exit code 1 if any metric is > 0.
    """
    # ✅ Create a temporary file with real code
    test_file = tmp_path / "test_input.py"
    test_file.write_text("def add(x, y):\n    return x + y\n")

    # ✅ Locate CLI script and configure environment
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "src" / "metrics" / "main.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--file", str(test_file),
            "--fail-threshold", "0"
        ],
        capture_output=True,
        text=True,
        cwd=str(project_root),
        env=env
    )

    assert result.returncode == 1, (
        f"Expected exit code 1 due to metrics > 0, but got {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
