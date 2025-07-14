# File: tests/test_cli_output.py

import subprocess
import sys
from pathlib import Path
import json
import csv
import os


def test_cli_generates_json_and_csv(tmp_path):
    """
    Runs the CLI using --format both,
    and checks that both JSON and CSV outputs are created and valid.
    """
    # ✅ Create a temporary test file with minimal content
    test_file = tmp_path / "cli_test.py"
    test_file.write_text("def test(): return 1\n")

    json_out = tmp_path / "output.json"
    csv_out = tmp_path / "output.csv"

    # ✅ Locate project root and CLI script
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "src" / "metrics" / "main.py"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--file", str(test_file),
            "--format", "both",
            "--json-out", str(json_out),
            "--csv-out", str(csv_out),
        ],
        cwd=str(project_root),
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"CLI execution failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    # ✅ Validate JSON output structure
    assert json_out.exists(), "Expected JSON output file not found"
    with json_out.open(encoding="utf-8") as jf:
        data = json.load(jf)
        assert isinstance(data, dict), "Top-level JSON object must be a dictionary"
        assert "metrics" in data, "JSON output missing 'metrics' key"
        numeric_metrics = {
            k: v for k, v in data["metrics"].items()
            if isinstance(v, (int, float))
        }
        assert numeric_metrics, f"No numeric metrics found in JSON output: {data}"
        assert any(v > 0 for v in numeric_metrics.values()), f"All numeric metrics are zero: {numeric_metrics}"

    # ✅ Validate CSV output structure
    assert csv_out.exists(), "Expected CSV output file not found"
    with csv_out.open(newline='', encoding="utf-8") as cf:
        reader = list(csv.reader(cf))
        assert len(reader) >= 2, "CSV must have header and at least one row"
        assert reader[0][0] == "File", "CSV header must start with 'File'"
        assert str(test_file.name) in reader[1][0], f"CSV output missing expected file entry: {test_file.name}"
