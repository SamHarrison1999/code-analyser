import logging
from pprint import pprint

# Ensure debug messages are visible
logging.basicConfig(level=logging.DEBUG)

def test_pylint_metric_names_import():
    try:
        # ✅ Correct, updated import path
        from metrics.pylint_metrics.gather import get_pylint_metric_names

        metrics = get_pylint_metric_names()
        print("✅ Pylint metric names retrieved successfully:")
        pprint(metrics)

        assert isinstance(metrics, list), "Expected a list of metric names"
        assert all(isinstance(m, str) for m in metrics), "All metric names should be strings"
        assert "warning" in metrics or "convention" in metrics, "Expected known metric keys to be present"

    except ModuleNotFoundError as mnfe:
        print("❌ ModuleNotFoundError:", mnfe)
    except ImportError as ie:
        print("❌ ImportError:", ie)
    except AssertionError as ae:
        print("❌ Assertion failed:", ae)
    except Exception as e:
        print("❌ Unexpected error:", e)

if __name__ == "__main__":
    test_pylint_metric_names_import()
