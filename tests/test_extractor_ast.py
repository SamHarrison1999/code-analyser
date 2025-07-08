# File: tests/test_extractor_ast.py

import tempfile
import pytest
from metrics.ast_metrics.extractor import ASTMetricExtractor as MetricExtractor


@pytest.fixture
def ast_sample_code():
    return '''
"""
Module-level docstring.
"""

# TODO: refactor this later
def outer():
    def inner():
        return 42
    return inner()

class Foo:
    def __init__(self): pass
    def __str__(self): return "hello"

lambda_fn = lambda x: x + 1

for i in range(5):
    if i % 2 == 0:
        print(i)

assert True
'''


def test_ast_metrics_extracted_correctly(ast_sample_code):
    with tempfile.NamedTemporaryFile("w+", suffix=".py", delete=False) as tmp:
        tmp.write(ast_sample_code)
        tmp.flush()
        extractor = MetricExtractor(tmp.name)
        features = extractor.extract()

    # ✅ Best Practice: Convert dict to list of values before slicing
    # 🧠 ML Signal: Positional feature extraction order may affect metric alignment in downstream models
    ast_metrics = list(features.values())  # previously: features[-24:]

    # ✅ Best Practice: Ensure that metrics match expected number and types
    assert isinstance(ast_metrics, list), "Extracted AST metrics should be a list"
    assert all(isinstance(v, int) for v in ast_metrics), "All AST metrics should be integers"
    assert len(ast_metrics) >= 7, "Expected at least 7 AST features"
