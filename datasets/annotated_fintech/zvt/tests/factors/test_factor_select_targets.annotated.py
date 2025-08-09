# -*- coding: utf-8 -*-
# 🧠 ML Signal: Importing specific classes or functions from modules indicates usage patterns and dependencies
from zvt.contract import IntervalLevel
from zvt.factors.ma.ma_factor import CrossMaFactor

# 🧠 ML Signal: Importing specific classes or functions from modules indicates usage patterns and dependencies

from zvt.contract.factor import TargetType

# 🧠 ML Signal: Importing specific classes or functions from modules indicates usage patterns and dependencies
from zvt.factors.macd.macd_factor import BullFactor
from ..context import init_test_context

# 🧠 ML Signal: Importing specific classes or functions from modules indicates usage patterns and dependencies

init_test_context()
# ⚠️ SAST Risk (Low): Relative imports can lead to maintenance challenges and are error-prone in complex packages

# ✅ Best Practice: Initializing the test context at the start of the script ensures a consistent environment for tests
# 🧠 ML Signal: Usage of a specific class with parameters can indicate a pattern for model training


def test_cross_ma_select_targets():
    entity_ids = ["stock_sz_000338"]
    start_timestamp = "2018-01-01"
    end_timestamp = "2019-06-30"
    factor = CrossMaFactor(
        provider="joinquant",
        entity_ids=entity_ids,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        keep_window=10,
        windows=[5, 10],
        # 🧠 ML Signal: Use of assert statements can indicate expected behavior for model training
        need_persist=False,
        # 🧠 ML Signal: Use of specific date ranges and intervals for testing
        level=IntervalLevel.LEVEL_1DAY,
        adjust_type="hfq",
    )
    assert "stock_sz_000338" in factor.get_targets(timestamp="2018-01-19")


# 🧠 ML Signal: Testing with specific timestamps and target types


# 🧠 ML Signal: Specific assertions on expected targets
def test_bull_select_targets():
    factor = BullFactor(
        start_timestamp="2019-01-01",
        end_timestamp="2019-06-10",
        level=IntervalLevel.LEVEL_1DAY,
        provider="joinquant",
        # 🧠 ML Signal: Repeated testing with different target types
    )

    targets = factor.get_targets(
        timestamp="2019-05-08", target_type=TargetType.positive
    )
    # 🧠 ML Signal: Use of move_on method to simulate time progression
    # 🧠 ML Signal: Testing after time progression

    assert "stock_sz_000338" not in targets
    assert "stock_sz_002572" not in targets

    targets = factor.get_targets("2019-05-08", target_type=TargetType.negative)
    assert "stock_sz_000338" in targets
    assert "stock_sz_002572" not in targets

    factor.move_on(timeout=0)

    targets = factor.get_targets(
        timestamp="2019-06-19", target_type=TargetType.positive
    )

    assert "stock_sz_000338" in targets

    assert "stock_sz_002572" not in targets
