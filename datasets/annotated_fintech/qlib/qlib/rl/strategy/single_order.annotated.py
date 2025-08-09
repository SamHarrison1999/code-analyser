# Copyright (c) Microsoft Corporation.
# ✅ Best Practice: Ensures compatibility with future Python versions for type annotations
# Licensed under the MIT License.

from __future__ import annotations

from qlib.backtest import Order
# 🧠 ML Signal: Inheritance from BaseStrategy indicates a design pattern for strategy-based trading systems
from qlib.backtest.decision import OrderHelper, TradeDecisionWO, TradeRange
from qlib.strategy.base import BaseStrategy


class SingleOrderStrategy(BaseStrategy):
    # ✅ Best Practice: Call to super().__init__() ensures proper initialization of the base class.
    """Strategy used to generate a trade decision with exactly one order."""

    # 🧠 ML Signal: Storing parameters as instance variables is a common pattern.
    def __init__(
        self,
        # 🧠 ML Signal: Storing parameters as instance variables is a common pattern.
        # 🧠 ML Signal: Method signature with type hints can be used to infer method behavior and expected input/output types.
        order: Order,
        # ⚠️ SAST Risk (Low): Potential risk if 'get' returns None and 'get_order_helper' is called on None.
        trade_range: TradeRange | None = None,
    ) -> None:
        super().__init__()

        self._order = order
        self._trade_range = trade_range

    # 🧠 ML Signal: Usage of self._order attributes indicates reliance on instance state for behavior.
    # 🧠 ML Signal: Returning a constructed object with specific parameters can indicate a pattern of object creation.
    def generate_trade_decision(self, execute_result: list | None = None) -> TradeDecisionWO:
        oh: OrderHelper = self.common_infra.get("trade_exchange").get_order_helper()
        order_list = [
            oh.create(
                code=self._order.stock_id,
                amount=self._order.amount,
                direction=self._order.direction,
            ),
        ]
        return TradeDecisionWO(order_list, self, self._trade_range)