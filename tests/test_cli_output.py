# File: tests/test_cli_output.py

import subprocess
import sys
from pathlib import Path
import json
import csv
import os


def test_cli_generates_json_and_csv(tmp_path):
    """
    Runs the CLI on example.py using --format both,
    and checks that both JSON and CSV outputs are created.
    """
    example_file = Path("example.py")
    assert example_file.exists(), "example.py must be in the root directory"

    json_out = tmp_path / "output.json"
    csv_out = tmp_path / "output.csv"

    # ✅ Best Practice: Use resolved full path to entry script and set PYTHONPATH to src
    script_path = Path(__file__).resolve().parent.parent / "src" / "metrics" / "main.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--file", str(example_file),
            "--format", "both",
            "--json-out", str(json_out),
            "--csv-out", str(csv_out),
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent / "src")},
    )

    assert result.returncode == 0, (
        f"CLI failed unexpectedly.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    assert json_out.exists(), "JSON output file not created"
    with open(json_out, encoding="utf-8") as jf:
        data = json.load(jf)
        assert isinstance(data, dict), "JSON output should be a dictionary"
        assert any(val > 0 for val in data.values()), "Expected non-zero metrics"

    assert csv_out.exists(), "CSV output file not created"
    with open(csv_out, newline='', encoding="utf-8") as cf:
        reader = list(csv.reader(cf))
        assert len(reader) >= 2, "CSV should have headers and one data row"
        assert reader[0][0] == "File", "CSV header should start with 'File'"
        assert reader[1][0] == "example.py", "CSV should include example.py row"
