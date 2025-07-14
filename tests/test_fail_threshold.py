# File: tests/test_fail_threshold.py

import subprocess
import sys
import os
from pathlib import Path


def test_fail_threshold_triggers_exit(tmp_path):
    """
    Run CLI with --fail-threshold set to 0, which should cause failure
    if any metric is non-zero in the temporary test file.
    """
    # ✅ Create temporary test file with some basic code
    test_file = tmp_path / "threshold_case.py"
    test_file.write_text("def something():\n    print('test')\n")

    # ✅ Resolve CLI script and configure environment
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

    # Exit code should be 1 if any metric exceeds the threshold of 0
    assert result.returncode == 1, (
        f"Expected exit code 1 due to fail-threshold, got {result.returncode}.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
