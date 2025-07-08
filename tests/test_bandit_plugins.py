"""
Unit tests for Bandit plugin classes.

Validates:
- Correct counting of severity levels
- CWE plugin fallback when no data is present
"""

import pytest
from metrics.bandit_metrics.plugins.default_plugins import (
    HighSeverityIssues,
    MediumSeverityIssues,
    LowSeverityIssues,
    UndefinedSeverityIssues,
)
from metrics.bandit_metrics.plugins.cwe_plugin import (
    CWEFrequencyPlugin,
    MostFrequentCWEPlugin,
)


@pytest.fixture
def bandit_sample_data():
    return {
        "results": [
            {"issue_severity": "HIGH"},
            {"issue_severity": "MEDIUM"},
            {"issue_severity": "LOW"},
            {"issue_severity": "UNDEFINED"},
            {"issue_severity": "LOW"},
            {"issue_severity": "high"},  # case insensitive
            {"issue_severity": "ignored"},  # treated as UNDEFINED
            {"issue_severity": "MEDIUM", "cwe": {"id": 79}},
            {"issue_severity": "LOW", "cwe": {"id": 79}},
            {"issue_severity": "MEDIUM", "cwe": {"id": 20}},
        ]
    }


def test_severity_plugins(bandit_sample_data):
    assert HighSeverityIssues().extract(bandit_sample_data) == 2
    assert MediumSeverityIssues().extract(bandit_sample_data) == 3
    assert LowSeverityIssues().extract(bandit_sample_data) == 3
    assert UndefinedSeverityIssues().extract(bandit_sample_data) == 2


def test_cwe_plugins(bandit_sample_data):
    assert CWEFrequencyPlugin().extract(bandit_sample_data) == 2
    assert MostFrequentCWEPlugin().extract(bandit_sample_data) == 79
