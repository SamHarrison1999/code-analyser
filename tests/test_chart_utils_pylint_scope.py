import sys
import os
from unittest.mock import MagicMock, patch

# ✅ Mock mplcursors before importing chart_utils
sys.modules["mplcursors"] = MagicMock()

# Setup test path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import types
import unittest
import gui.chart_utils as chart_utils


class TestPylintScope(unittest.TestCase):
    def setUp(self):
        # 👇 Build a fake gui.shared_state module with a mocked metric_scope
        mock_shared_state = types.ModuleType("shared_state")
        mock_metric_scope = MagicMock()
        mock_metric_scope.get.return_value = "pylint"
        mock_shared_state.metric_scope = mock_metric_scope

        # 👇 Inject this into sys.modules before gui.chart_utils uses it
        sys.modules["gui.shared_state"] = mock_shared_state

    @patch("gui.chart_utils.include_metric")
    def test_include_metric_pylint_scope(self, mock_include_metric):
        # Simulate expected returns for specific metrics
        mock_include_metric.side_effect = lambda m: m in {"convention", "fatal"}

        self.assertTrue(chart_utils.include_metric("convention"))
        self.assertTrue(chart_utils.include_metric("fatal"))
        self.assertFalse(chart_utils.include_metric("lines_of_code"))
        self.assertFalse(chart_utils.include_metric("non_pylint_metric"))

    @patch("gui.chart_utils.include_metric")
    def test_filter_metrics_by_scope_pylint(self, mock_get_names):
        metrics = {
            "convention": 3,
            "refactor": 2,
            "warning": 1,
            "non_pylint": 99,
        }
        filtered = chart_utils.filter_metrics_by_scope(metrics)
        self.assertEqual(set(filtered.keys()), {"convention", "refactor", "warning"})


if __name__ == "__main__":
    unittest.main()
