# File: tests/test_summary_output_txt.py

import subprocess
import sys
import os
from pathlib import Path


def test_cli_summary_output_txt(tmp_path):
    """
    Runs the CLI with --save-summary-txt and validates that the markdown-style
    summary is saved correctly.
    """
    # ✅ Create a temporary test file with basic content
    test_file = tmp_path / "summary_test.py"
    test_file.write_text("def test():\n    return 42\n")

    summary_path = tmp_path / "summary.txt"

    # ✅ Locate the CLI script and project root
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "src" / "metrics" / "main.py"

    # ✅ Ensure PYTHONPATH is set for plugin loading
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--file", str(test_file),
            "--save-summary-txt", str(summary_path)
        ],
        capture_output=True,
        text=True,
        cwd=str(project_root),
        env=env,
    )

    assert result.returncode == 0, (
        f"CLI failed unexpectedly.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    assert summary_path.exists(), "Summary TXT file was not created"
    content = summary_path.read_text(encoding="utf-8")
    assert "Metric | Value" in content, "Expected 'Metric | Value' markdown table header in summary"
    assert any(marker in content for marker in ("##", "---")), "Expected markdown-style headings or separators"
