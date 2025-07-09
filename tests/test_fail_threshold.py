# File: tests/test_fail_threshold.py

import subprocess
import sys
from pathlib import Path


def test_fail_threshold_triggers_exit():
    """
    Run CLI with --fail-threshold set to 0, which should cause failure
    if any metric is non-zero in example.py.
    """
    test_file = Path("example.py")
    assert test_file.exists(), "example.py must exist in project root"

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

    # Exit code should be 1 if any metric is non-zero
    assert result.returncode == 1, (
        f"Expected failure exit code due to threshold, got {result.returncode}.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
