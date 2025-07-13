import pytest
from unittest.mock import patch
from gui import chart_utils

class TestPylintScope:

    @patch("gui.chart_utils.get_shared_state")
    def test_filter_metrics_by_scope_pylint(self, mock_get_state):
        """
        Test that metrics are correctly filtered when 'pylint' scope is active.
        """
        # Mock the scope selection and available scopes
        class MockMetricScope:
            def get(self): return "pylint"

        mock_get_state.return_value.metric_scope = MockMetricScope()
        mock_get_state.return_value.available_scopes = {
            "pylint": {"convention", "refactor", "warning", "error", "fatal"}
        }

        sample_metrics = {
            "pylint": {
                "convention": 5,
                "refactor": 2,
                "warning": 1,
                "error": 0,
                "fatal": 0
            },
            "flake8": {
                "number_of_unused_imports": 3
            },
            "other": {
                "non_numeric": "skip"
            }
        }

        result = chart_utils.filter_metrics_by_scope(sample_metrics)

        expected_keys = {
            "pylint.convention",
            "pylint.refactor",
            "pylint.warning",
            "pylint.error",
            "pylint.fatal"
        }

        assert set(result.keys()) == expected_keys
        assert all(isinstance(v, float) for v in result.values())

    @patch("gui.chart_utils.get_shared_state")
    def test_include_metric_pylint_scope(self, mock_get_state):
        """
        Ensure filtering excludes non-pylint metrics and non-numeric values.
        """
        class MockMetricScope:
            def get(self): return "pylint"

        mock_get_state.return_value.metric_scope = MockMetricScope()
        mock_get_state.return_value.available_scopes = {
            "pylint": {"convention", "fatal", "warning"}
        }

        sample_metrics = {
            "pylint": {
                "convention": 2,
                "fatal": 1,
                "warning": "warn"  # Invalid, should be skipped
            },
            "radon": {
                "average_halstead_volume": 3.1
            }
        }

        filtered = chart_utils.filter_metrics_by_scope(sample_metrics)

        assert "pylint.convention" in filtered
        assert "pylint.fatal" in filtered
        assert "pylint.warning" not in filtered
        assert "radon.average_halstead_volume" not in filtered
