# File: tests/test_fail_if_nonzero.py

import subprocess
import sys
from pathlib import Path


def test_fail_if_nonzero_triggers_exit():
    """
    Runs the CLI with --fail-threshold 0 on a file with non-zero metrics.
    Expects exit code 1 if any metric is > 0.
    """
    test_file = Path("example.py")
    assert test_file.exists(), "example.py must exist"

    result = subprocess.run(
        [
            sys.executable,
            "-m", "metrics.main",
            "--file", str(test_file),
            "--fail-threshold", "0"
        ],
        capture_output=True,
        text=True
    )

    assert result.returncode == 1, (
        f"Expected failure exit due to non-zero metrics, got {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
