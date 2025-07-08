# File: tests/test_summary_output_txt.py

import subprocess
import sys
from pathlib import Path
import os


def test_cli_summary_output_txt(tmp_path):
    """
    Runs the CLI with --save-summary-txt and validates that the markdown-style
    summary is saved correctly.
    """
    test_file = Path("example.py")
    assert test_file.exists(), "example.py must exist"

    summary_path = tmp_path / "summary.txt"

    # ✅ Best Practice: Run CLI via fully resolved path with correct PYTHONPATH
    script_path = Path(__file__).resolve().parent.parent / "src" / "metrics" / "main.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--file", str(test_file),
            "--save-summary-txt", str(summary_path)
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent / "src")},
    )

    assert result.returncode == 0, (
        f"CLI failed unexpectedly.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    assert summary_path.exists(), "Summary TXT file was not created"
    content = summary_path.read_text(encoding="utf-8")
    assert "Metric | Value" in content, "Expected metric table in summary output"
    assert "##" in content or "-" in content, "Expected markdown-style structure"
