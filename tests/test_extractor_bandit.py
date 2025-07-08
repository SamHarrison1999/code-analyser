"""
Integration test for BanditExtractor using a real Python file.
"""

import tempfile
import os
from metrics.bandit_metrics.extractor import BanditExtractor


def test_bandit_extractor_returns_valid_metrics():
    code = """
import subprocess

def insecure():
    subprocess.call("rm -rf /", shell=True)
    """

    with tempfile.NamedTemporaryFile("w+", suffix=".py", delete=False) as tmp:
        tmp.write(code)
        tmp.flush()
        path = tmp.name

    try:
        extractor = BanditExtractor(path)
        metrics = extractor.extract()

        assert isinstance(metrics, dict)
        assert all(isinstance(v, int) for v in metrics.values())
        assert sum(metrics.values()) > 0  # At least one issue should be detected
    finally:
        os.remove(path)
