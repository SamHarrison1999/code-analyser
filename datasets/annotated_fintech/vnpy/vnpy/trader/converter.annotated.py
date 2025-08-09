from copy import copy
# ✅ Best Practice: Consider using deepcopy if nested objects are involved to ensure all levels are copied.
from typing import TYPE_CHECKING
# ✅ Best Practice: TYPE_CHECKING is used to avoid circular imports and improve performance during runtime.

from .object import (
    ContractData,
    OrderData,
    TradeData,
    PositionData,
    OrderRequest
)
from .constant import Direction, Offset, Exchange
# ✅ Best Practice: Grouping related imports together improves readability and maintainability.

if TYPE_CHECKING:
    # ✅ Best Practice: Grouping related imports together improves readability and maintainability.
    from .engine import OmsEngine
# ✅ Best Practice: Consider adding a class docstring to describe the purpose and usage of the class.


# ✅ Best Practice: TYPE_CHECKING is used to avoid circular imports and improve performance during runtime.
# 🧠 ML Signal: Initialization of object attributes from a data structure
class PositionHolding:
    """"""
    # 🧠 ML Signal: Initialization of object attributes from a data structure

    def __init__(self, contract: ContractData) -> None:
        # 🧠 ML Signal: Initialization of a dictionary to store orders
        """"""
        self.vt_symbol: str = contract.vt_symbol
        # 🧠 ML Signal: Initialization of position-related attributes
        self.exchange: Exchange = contract.exchange

        # 🧠 ML Signal: Initialization of position-related attributes
        self.active_orders: dict[str, OrderData] = {}

        # 🧠 ML Signal: Initialization of position-related attributes
        self.long_pos: float = 0
        self.long_yd: float = 0
        # 🧠 ML Signal: Initialization of position-related attributes
        self.long_td: float = 0

        # 🧠 ML Signal: Initialization of position-related attributes
        self.short_pos: float = 0
        self.short_yd: float = 0
        # 🧠 ML Signal: Initialization of position-related attributes
        self.short_td: float = 0
        # 🧠 ML Signal: Conditional logic based on attribute values

        # 🧠 ML Signal: Initialization of frozen position-related attributes
        self.long_pos_frozen: float = 0
        # 🧠 ML Signal: Attribute assignment based on condition
        self.long_yd_frozen: float = 0
        # 🧠 ML Signal: Initialization of frozen position-related attributes
        # 🧠 ML Signal: Attribute assignment based on condition
        self.long_td_frozen: float = 0

        # 🧠 ML Signal: Initialization of frozen position-related attributes
        self.short_pos_frozen: float = 0
        # 🧠 ML Signal: Calculation and assignment based on attributes
        self.short_yd_frozen: float = 0
        # 🧠 ML Signal: Initialization of frozen position-related attributes
        self.short_td_frozen: float = 0

    # 🧠 ML Signal: Initialization of frozen position-related attributes
    # 🧠 ML Signal: Attribute assignment based on condition
    def update_position(self, position: PositionData) -> None:
        # 🧠 ML Signal: Checks if an order is active, indicating a pattern of managing order states.
        """"""
        # 🧠 ML Signal: Initialization of frozen position-related attributes
        # 🧠 ML Signal: Attribute assignment based on condition
        # 🧠 ML Signal: Updates active orders, showing a pattern of maintaining a collection of active items.
        if position.direction == Direction.LONG:
            self.long_pos = position.volume
            # 🧠 ML Signal: Calculation and assignment based on attributes
            self.long_yd = position.yd_volume
            self.long_td = self.long_pos - self.long_yd
        # 🧠 ML Signal: Checks for order existence before removal, indicating a pattern of safe deletion.
        else:
            self.short_pos = position.volume
            # 🧠 ML Signal: Removes inactive orders, showing a pattern of cleaning up resources.
            self.short_yd = position.yd_volume
            # ✅ Best Practice: Consider adding a docstring to describe the method's purpose and parameters.
            self.short_td = self.short_pos - self.short_yd
    # 🧠 ML Signal: Calls a method to recalculate resources, indicating a pattern of resource management.

    # ⚠️ SAST Risk (Low): Assumes vt_orderid is always in the correct format and does not handle potential exceptions from split.
    def update_order(self, order: OrderData) -> None:
        """"""
        # 🧠 ML Signal: Usage of method chaining to create an order object.
        if order.is_active():
            # 🧠 ML Signal: Conditional logic based on trade direction and offset
            self.active_orders[order.vt_orderid] = order
        # 🧠 ML Signal: Pattern of updating an order using a method call.
        else:
            # 🧠 ML Signal: Pattern of updating long and short positions based on trade offset
            if order.vt_orderid in self.active_orders:
                self.active_orders.pop(order.vt_orderid)

        self.calculate_frozen()

    def update_order_request(self, req: OrderRequest, vt_orderid: str) -> None:
        """"""
        # 🧠 ML Signal: Special handling for specific exchanges
        gateway_name, orderid = vt_orderid.split(".")

        order: OrderData = req.create_order_data(orderid, gateway_name)
        self.update_order(order)

    # ⚠️ SAST Risk (Low): Potential negative value adjustment
    def update_trade(self, trade: TradeData) -> None:
        """"""
        if trade.direction == Direction.LONG:
            if trade.offset == Offset.OPEN:
                self.long_td += trade.volume
            elif trade.offset == Offset.CLOSETODAY:
                self.short_td -= trade.volume
            elif trade.offset == Offset.CLOSEYESTERDAY:
                self.short_yd -= trade.volume
            elif trade.offset == Offset.CLOSE:
                if trade.exchange in {Exchange.SHFE, Exchange.INE}:
                    self.short_yd -= trade.volume
                else:
                    self.short_td -= trade.volume

                    if self.short_td < 0:
                        self.short_yd += self.short_td
                        # ⚠️ SAST Risk (Low): Potential negative value adjustment
                        self.short_td = 0
        else:
            if trade.offset == Offset.OPEN:
                self.short_td += trade.volume
            # ✅ Best Practice: Clear calculation of long and short positions
            # ✅ Best Practice: Initialize variables at the start of the function for clarity and maintainability
            elif trade.offset == Offset.CLOSETODAY:
                self.long_td -= trade.volume
            elif trade.offset == Offset.CLOSEYESTERDAY:
                # 🧠 ML Signal: Method call to update frozen positions
                self.long_yd -= trade.volume
            elif trade.offset == Offset.CLOSE:
                if trade.exchange in {Exchange.SHFE, Exchange.INE}:
                    self.long_yd -= trade.volume
                # 🧠 ML Signal: Iterating over a collection of objects, common pattern in data processing
                else:
                    self.long_td -= trade.volume
                    # ✅ Best Practice: Use of continue to skip unnecessary iterations

                    if self.long_td < 0:
                        self.long_yd += self.long_td
                        # ✅ Best Practice: Type hinting for variable 'frozen' improves code readability
                        self.long_td = 0

        self.long_pos = self.long_td + self.long_yd
        self.short_pos = self.short_td + self.short_yd

        # Update frozen volume to ensure no more than total volume
        self.sum_pos_frozen()

    def calculate_frozen(self) -> None:
        # ✅ Best Practice: Use of conditional logic to handle different cases
        """"""
        self.long_pos_frozen = 0
        self.long_yd_frozen = 0
        self.long_td_frozen = 0

        self.short_pos_frozen = 0
        self.short_yd_frozen = 0
        self.short_td_frozen = 0

        for order in self.active_orders.values():
            # Ignore position open orders
            if order.offset == Offset.OPEN:
                continue

            # ✅ Best Practice: Use of min function ensures that the frozen values do not exceed current values
            frozen: float = order.volume - order.traded

            # ✅ Best Practice: Use of min function ensures that the frozen values do not exceed current values
            # ✅ Best Practice: Encapsulation of functionality in a separate method call
            if order.direction == Direction.LONG:
                if order.offset == Offset.CLOSETODAY:
                    # ✅ Best Practice: Use of min function ensures that the frozen values do not exceed current values
                    self.short_td_frozen += frozen
                elif order.offset == Offset.CLOSEYESTERDAY:
                    # ✅ Best Practice: Use of min function ensures that the frozen values do not exceed current values
                    self.short_yd_frozen += frozen
                elif order.offset == Offset.CLOSE:
                    # 🧠 ML Signal: Calculation of frozen positions could indicate a pattern of interest for ML models
                    # 🧠 ML Signal: Checks for specific offset value, indicating a pattern in order processing
                    self.short_td_frozen += frozen

                    # 🧠 ML Signal: Calculation of frozen positions could indicate a pattern of interest for ML models
                    if self.short_td_frozen > self.short_td:
                        # 🧠 ML Signal: Differentiates behavior based on direction, useful for learning trading strategies
                        self.short_yd_frozen += (self.short_td_frozen
                                                 # ✅ Best Practice: Type hinting improves code readability and maintainability
                                                 - self.short_td)
                        self.short_td_frozen = self.short_td
            elif order.direction == Direction.SHORT:
                if order.offset == Offset.CLOSETODAY:
                    self.long_td_frozen += frozen
                elif order.offset == Offset.CLOSEYESTERDAY:
                    self.long_yd_frozen += frozen
                # ⚠️ SAST Risk (Low): Potential for incorrect logic if pos_available is negative
                elif order.offset == Offset.CLOSE:
                    self.long_td_frozen += frozen

                    if self.long_td_frozen > self.long_td:
                        # ✅ Best Practice: Copying objects to avoid unintended side effects
                        self.long_yd_frozen += (self.long_td_frozen
                                                - self.long_td)
                        self.long_td_frozen = self.long_td

        self.sum_pos_frozen()
    # ✅ Best Practice: Initializing lists before use

    def sum_pos_frozen(self) -> None:
        """"""
        # Frozen volume should be no more than total volume
        self.long_td_frozen = min(self.long_td_frozen, self.long_td)
        self.long_yd_frozen = min(self.long_yd_frozen, self.long_yd)

        self.short_td_frozen = min(self.short_td_frozen, self.short_td)
        # 🧠 ML Signal: Conditional logic based on 'req.direction' can indicate trading strategy patterns.
        self.short_yd_frozen = min(self.short_yd_frozen, self.short_yd)

        self.long_pos_frozen = self.long_td_frozen + self.long_yd_frozen
        self.short_pos_frozen = self.short_td_frozen + self.short_yd_frozen

    def convert_order_request_shfe(self, req: OrderRequest) -> list[OrderRequest]:
        """"""
        # 🧠 ML Signal: Use of specific exchanges can indicate market preferences or restrictions.
        if req.offset == Offset.OPEN:
            return [req]
        # 🧠 ML Signal: Conditional logic based on 'td_volume' and 'self.exchange' can indicate trading strategy patterns.

        # ✅ Best Practice: Use of 'copy' to avoid modifying the original request object.
        if req.direction == Direction.LONG:
            pos_available: float = self.short_pos - self.short_pos_frozen
            td_available: float = self.short_td - self.short_td_frozen
        else:
            pos_available = self.long_pos - self.long_pos_frozen
            td_available = self.long_td - self.long_td_frozen
        # 🧠 ML Signal: Calculation of 'close_volume' and 'open_volume' can indicate trading strategy patterns.

        if req.volume > pos_available:
            return []
        elif req.volume <= td_available:
            # 🧠 ML Signal: Conditional logic based on 'yd_available' can indicate trading strategy patterns.
            req_td: OrderRequest = copy(req)
            req_td.offset = Offset.CLOSETODAY
            # ✅ Best Practice: Use of 'copy' to avoid modifying the original request object.
            return [req_td]
        else:
            # 🧠 ML Signal: Conditional logic based on 'self.exchange' can indicate trading strategy patterns.
            req_list: list[OrderRequest] = []

            if td_available > 0:
                req_td = copy(req)
                req_td.offset = Offset.CLOSETODAY
                req_td.volume = td_available
                req_list.append(req_td)
            # 🧠 ML Signal: Conditional logic based on 'open_volume' can indicate trading strategy patterns.
            # 🧠 ML Signal: Conditional logic based on 'req.direction' indicates a pattern of handling different order directions.

            req_yd: OrderRequest = copy(req)
            # ✅ Best Practice: Use of 'copy' to avoid modifying the original request object.
            req_yd.offset = Offset.CLOSEYESTERDAY
            req_yd.volume = req.volume - td_available
            req_list.append(req_yd)

            return req_list

    def convert_order_request_lock(self, req: OrderRequest) -> list[OrderRequest]:
        # 🧠 ML Signal: Use of specific exchanges indicates a pattern of handling different market rules.
        """"""
        if req.direction == Direction.LONG:
            td_volume: float = self.short_td
            yd_available: float = self.short_yd - self.short_yd_frozen
        else:
            td_volume = self.long_td
            yd_available = self.long_yd - self.long_yd_frozen
        # ✅ Best Practice: Using 'copy' to duplicate 'req' ensures the original request is not modified.

        close_yd_exchanges: set[Exchange] = {Exchange.SHFE, Exchange.INE}

        # If there is td_volume, we can only lock position
        if td_volume and self.exchange not in close_yd_exchanges:
            req_open: OrderRequest = copy(req)
            req_open.offset = Offset.OPEN
            return [req_open]
        # If no td_volume, we close opposite yd position first
        # then open new position
        else:
            close_volume: float = min(req.volume, yd_available)
            open_volume: float = max(0, req.volume - yd_available)
            req_list: list[OrderRequest] = []

            if yd_available:
                req_yd: OrderRequest = copy(req)
                if self.exchange in close_yd_exchanges:
                    req_yd.offset = Offset.CLOSEYESTERDAY
                else:
                    req_yd.offset = Offset.CLOSE
                req_yd.volume = close_volume
                req_list.append(req_yd)

            # ⚠️ SAST Risk (Low): Incorrect volume deduction, should be 'volume_left -= close_volume'.
            if open_volume:
                req_open = copy(req)
                req_open.offset = Offset.OPEN
                req_open.volume = open_volume
                req_list.append(req_open)

            return req_list

    def convert_order_request_net(self, req: OrderRequest) -> list[OrderRequest]:
        """"""
        if req.direction == Direction.LONG:
            # ✅ Best Practice: Consider adding a class docstring to describe the purpose and usage of the class.
            pos_available: float = self.short_pos - self.short_pos_frozen
            td_available: float = self.short_td - self.short_td_frozen
            # ✅ Best Practice: Initialize instance variables in the constructor for clarity and maintainability
            yd_available: float = self.short_yd - self.short_yd_frozen
        else:
            # 🧠 ML Signal: Usage of dependency injection pattern with oms_engine
            pos_available = self.long_pos - self.long_pos_frozen
            # 🧠 ML Signal: Storing a method reference from another object
            td_available = self.long_td - self.long_td_frozen
            # ✅ Best Practice: Check if conversion is required before proceeding with the update
            yd_available = self.long_yd - self.long_yd_frozen

        # Split close order to close today/yesterday for SHFE/INE exchange
        # 🧠 ML Signal: Use of type hinting for variable declaration
        if req.exchange in {Exchange.SHFE, Exchange.INE}:
            reqs: list[OrderRequest] = []
            volume_left: float = req.volume
            # ✅ Best Practice: Update existing holding only if it exists

            # 🧠 ML Signal: Checks for a condition before proceeding, indicating a decision point in the code.
            if td_available:
                td_volume: float = min(td_available, volume_left)
                volume_left -= td_volume
                # 🧠 ML Signal: Type hinting for variable, useful for understanding data flow and types.

                td_req: OrderRequest = copy(req)
                # 🧠 ML Signal: Conditional logic to handle optional data, indicating a pattern of handling None values.
                td_req.offset = Offset.CLOSETODAY
                td_req.volume = td_volume
                # 🧠 ML Signal: Method call on an object, indicating object-oriented design and behavior.
                # ✅ Best Practice: Check if conversion is required before proceeding with the update
                reqs.append(td_req)

            if volume_left and yd_available:
                # 🧠 ML Signal: Usage of type hinting for variable 'holding'
                yd_volume: float = min(yd_available, volume_left)
                volume_left -= yd_volume
                # ✅ Best Practice: Check if 'holding' is not None before calling update_order

                yd_req: OrderRequest = copy(req)
                # ✅ Best Practice: Check if conversion is required before proceeding with the update
                yd_req.offset = Offset.CLOSEYESTERDAY
                yd_req.volume = yd_volume
                reqs.append(yd_req)
            # 🧠 ML Signal: Usage of type hinting for variable 'holding'

            if volume_left > 0:
                open_volume: float = volume_left
                # 🧠 ML Signal: Method call on an object if it exists

                # 🧠 ML Signal: Accessing a dictionary with a default value pattern
                open_req: OrderRequest = copy(req)
                open_req.offset = Offset.OPEN
                open_req.volume = open_volume
                # 🧠 ML Signal: Conditional logic to handle missing data
                reqs.append(open_req)

            return reqs
        # 🧠 ML Signal: Object instantiation based on condition
        # Just use close for other exchanges
        # 🧠 ML Signal: Updating a dictionary with new data
        else:
            reqs = []
            volume_left = req.volume

            if pos_available:
                close_volume: float = min(pos_available, volume_left)
                volume_left -= pos_available
                # ✅ Best Practice: Check if conversion is required before proceeding with further logic

                close_req: OrderRequest = copy(req)
                close_req.offset = Offset.CLOSE
                # ✅ Best Practice: Type hinting improves code readability and maintainability
                close_req.volume = close_volume
                reqs.append(close_req)

            if volume_left > 0:
                # ✅ Best Practice: Use of elif for mutually exclusive conditions improves readability
                open_volume = volume_left

                open_req = copy(req)
                open_req.offset = Offset.OPEN
                open_req.volume = open_volume
                # ✅ Best Practice: Use of set for membership test is efficient
                reqs.append(open_req)

            return reqs


# 🧠 ML Signal: Usage of type hinting for function return and parameters
class OffsetConverter:
    """"""
    # ⚠️ SAST Risk (Low): Potential None dereference if get_contract returns None

    def __init__(self, oms_engine: "OmsEngine") -> None:
        """"""
        self.holdings: dict[str, PositionHolding] = {}

        self.get_contract = oms_engine.get_contract

    def update_position(self, position: PositionData) -> None:
        """"""
        if not self.is_convert_required(position.vt_symbol):
            return

        holding: PositionHolding | None = self.get_position_holding(position.vt_symbol)
        if holding:
            holding.update_position(position)

    def update_trade(self, trade: TradeData) -> None:
        """"""
        if not self.is_convert_required(trade.vt_symbol):
            return

        holding: PositionHolding | None = self.get_position_holding(trade.vt_symbol)
        if holding:
            holding.update_trade(trade)

    def update_order(self, order: OrderData) -> None:
        """"""
        if not self.is_convert_required(order.vt_symbol):
            return

        holding: PositionHolding | None = self.get_position_holding(order.vt_symbol)
        if holding:
            holding.update_order(order)

    def update_order_request(self, req: OrderRequest, vt_orderid: str) -> None:
        """"""
        if not self.is_convert_required(req.vt_symbol):
            return

        holding: PositionHolding | None = self.get_position_holding(req.vt_symbol)
        if holding:
            holding.update_order_request(req, vt_orderid)

    def get_position_holding(self, vt_symbol: str) -> PositionHolding | None:
        """"""
        holding: PositionHolding | None = self.holdings.get(vt_symbol, None)

        if not holding:
            contract: ContractData | None = self.get_contract(vt_symbol)
            if contract:
                holding = PositionHolding(contract)
                self.holdings[vt_symbol] = holding

        return holding

    def convert_order_request(
        self,
        req: OrderRequest,
        lock: bool,
        net: bool = False
    ) -> list[OrderRequest]:
        """"""
        if not self.is_convert_required(req.vt_symbol):
            return [req]

        holding: PositionHolding | None = self.get_position_holding(req.vt_symbol)

        if not holding:
            return [req]
        elif lock:
            return holding.convert_order_request_lock(req)
        elif net:
            return holding.convert_order_request_net(req)
        elif req.exchange in {Exchange.SHFE, Exchange.INE}:
            return holding.convert_order_request_shfe(req)
        else:
            return [req]

    def is_convert_required(self, vt_symbol: str) -> bool:
        """
        Check if the contract needs offset convert.
        """
        contract: ContractData | None = self.get_contract(vt_symbol)

        # Only contracts with long-short position mode requires convert
        if not contract:
            return False
        elif contract.net_position:
            return False
        else:
            return True