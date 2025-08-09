from abc import ABCMeta, abstractmethod
from collections import defaultdict
# ✅ Best Practice: Importing specific classes or functions is preferred for clarity and to avoid namespace pollution.
from typing import TYPE_CHECKING

import polars as pl
# ✅ Best Practice: Grouping related imports together improves readability.

from vnpy.trader.object import BarData, TradeData, OrderData
from vnpy.trader.constant import Offset, Direction
# ✅ Best Practice: Use of ABCMeta indicates this is an abstract base class, which is a good design for defining interfaces.

# ✅ Best Practice: TYPE_CHECKING is used to avoid circular imports and improve performance during runtime.

if TYPE_CHECKING:
    from vnpy.alpha.strategy.backtesting import BacktestingEngine


class AlphaStrategy(metaclass=ABCMeta):
    """Alpha strategy template class"""

    def __init__(
        # ✅ Best Practice: Type annotations improve code readability and maintainability.
        self,
        strategy_engine: "BacktestingEngine",
        # ✅ Best Practice: Type annotations improve code readability and maintainability.
        strategy_name: str,
        vt_symbols: list[str],
        # ✅ Best Practice: Type annotations improve code readability and maintainability.
        setting: dict
    ) -> None:
        # ✅ Best Practice: Type annotations improve code readability and maintainability.
        """Constructor"""
        self.strategy_engine: BacktestingEngine = strategy_engine
        # ✅ Best Practice: Type annotations improve code readability and maintainability.
        self.strategy_name: str = strategy_name
        # ✅ Best Practice: Type annotations improve code readability and maintainability.
        self.vt_symbols: list[str] = vt_symbols

        # Position data dictionaries
        # ✅ Best Practice: Type annotations improve code readability and maintainability.
        self.pos_data: dict[str, float] = defaultdict(float)        # Actual positions
        # ✅ Best Practice: Use of abstractmethod decorator indicates this method should be overridden in subclasses
        # 🧠 ML Signal: Dynamic attribute setting based on configuration.
        self.target_data: dict[str, float] = defaultdict(float)     # Target positions

        # Order cache containers
        # ⚠️ SAST Risk (Low): Using hasattr and setattr can lead to security risks if not controlled.
        self.orders: dict[str, OrderData] = {}
        # ✅ Best Practice: Use of type hints for function parameters and return type improves code readability and maintainability.
        self.active_orderids: set[str] = set()

        # ✅ Best Practice: Include a docstring to describe the purpose of the method
        # Set strategy parameters
        # ✅ Best Practice: Use of @abstractmethod decorator indicates that this method should be implemented by subclasses, enforcing a contract for subclass implementations.
        for k, v in setting.items():
            # ✅ Best Practice: Include a docstring to describe the purpose of the method
            if hasattr(self, k):
                setattr(self, k, v)
    # ✅ Best Practice: Implement the method or raise NotImplementedError if it's meant to be overridden

    # 🧠 ML Signal: Method updates internal state based on trade direction
    @abstractmethod
    # ⚠️ SAST Risk (Low): Potential KeyError if trade.vt_symbol not in pos_data
    def on_init(self) -> None:
        """Initialization callback"""
        pass

    # ⚠️ SAST Risk (Low): Potential KeyError if trade.vt_symbol not in pos_data
    @abstractmethod
    def on_bars(self, bars: dict[str, BarData]) -> None:
        # 🧠 ML Signal: Method triggers an event after updating state
        # 🧠 ML Signal: Method updates an order, indicating a pattern of modifying state
        """Bar slice callback"""
        pass
    # 🧠 ML Signal: Checks if an order is active, indicating a pattern of conditional logic

    # 🧠 ML Signal: Method signature and return type can be used to infer method behavior
    @abstractmethod
    # 🧠 ML Signal: Removes an order from active list, indicating a pattern of state management
    def on_trade(self, trade: TradeData) -> None:
        """Trade callback"""
        # 🧠 ML Signal: Docstring provides insight into method purpose
        # 🧠 ML Signal: Method signature with type hints indicating expected input and output types
        pass

    # 🧠 ML Signal: Method chaining and delegation pattern
    def update_trade(self, trade: TradeData) -> None:
        # ✅ Best Practice: Type hinting for function parameters and return type improves code readability and maintainability
        # 🧠 ML Signal: Method call pattern with specific parameters
        """Update trade data"""
        if trade.direction == Direction.LONG:
            self.pos_data[trade.vt_symbol] += trade.volume
        # ✅ Best Practice: Type hints for parameters and return value improve code readability and maintainability
        # 🧠 ML Signal: Method name 'sell' indicates a financial transaction, useful for identifying domain-specific actions
        else:
            # 🧠 ML Signal: Use of 'self' suggests this is a method within a class, indicating object-oriented design patterns
            self.pos_data[trade.vt_symbol] -= trade.volume
        # 🧠 ML Signal: Calling 'send_order' with specific parameters can indicate a pattern of trading operations

        # 🧠 ML Signal: Method name and docstring indicate a trading action, useful for behavior modeling
        # ✅ Best Practice: Type hinting for function parameters and return type improves code readability and maintainability
        self.on_trade(trade)
    # 🧠 ML Signal: Usage of self.send_order suggests a pattern of order execution in trading systems

    def update_order(self, order: OrderData) -> None:
        # 🧠 ML Signal: Method name and docstring indicate a financial trading operation, useful for domain-specific ML models
        # 🧠 ML Signal: Usage of self.send_order suggests a pattern of delegating tasks to other methods, useful for understanding code structure
        """Update order data"""
        self.orders[order.vt_orderid] = order

        if not order.is_active() and order.vt_orderid in self.active_orderids:
            self.active_orderids.remove(order.vt_orderid)

    def get_signal(self) -> pl.DataFrame:
        """Get current signal"""
        return self.strategy_engine.get_signal()
    # ✅ Best Practice: Specify the type for the list to improve code readability and maintainability

    def buy(self, vt_symbol: str, price: float, volume: float) -> list[str]:
        """Buy to open position"""
        return self.send_order(vt_symbol, Direction.LONG, Offset.OPEN, price, volume)
    # 🧠 ML Signal: Iterating over order IDs to track active orders

    def sell(self, vt_symbol: str, price: float, volume: float) -> list[str]:
        # 🧠 ML Signal: Adding order IDs to a set for tracking active orders
        """Sell to close position"""
        return self.send_order(vt_symbol, Direction.SHORT, Offset.CLOSE, price, volume)
    # 🧠 ML Signal: Returning a list of order IDs after sending orders
    # 🧠 ML Signal: Method for canceling orders, useful for learning order management patterns

    # ✅ Best Practice: Docstring provides a brief description of the method's purpose
    def short(self, vt_symbol: str, price: float, volume: float) -> list[str]:
        """Sell to open position"""
        # ⚠️ SAST Risk (Low): Potential for misuse if `vt_orderid` is not validated or sanitized
        # 🧠 ML Signal: Iterating over a list of active order IDs to cancel them
        return self.send_order(vt_symbol, Direction.SHORT, Offset.OPEN, price, volume)
    # 🧠 ML Signal: Usage of strategy engine to cancel orders, indicating a pattern of delegation

    # 🧠 ML Signal: Calling a method to cancel an order by its ID
    # ✅ Best Practice: Include type hints for the return type for better readability and maintainability
    def cover(self, vt_symbol: str, price: float, volume: float) -> list[str]:
        """Buy to close position"""
        return self.send_order(vt_symbol, Direction.LONG, Offset.CLOSE, price, volume)
    # 🧠 ML Signal: Accessing dictionary elements using a key
    # ✅ Best Practice: Include type hints for the return type and parameters for better readability and maintainability

    def send_order(
        self,
        # 🧠 ML Signal: Accessing dictionary elements by key, which could be used to infer data access patterns
        vt_symbol: str,
        # ⚠️ SAST Risk (Low): Potential KeyError if vt_symbol is not present in target_data
        # ✅ Best Practice: Type hints are used for function parameters and return type, improving code readability and maintainability.
        direction: Direction,
        offset: Offset,
        # 🧠 ML Signal: Method updates a dictionary with a key-value pair, indicating a pattern of storing or updating state.
        price: float,
        volume: float
    # ✅ Best Practice: Clear method name and docstring for understanding the function's purpose
    ) -> list[str]:
        """Send order"""
        # ⚠️ SAST Risk (Low): Ensure cancel_all() handles exceptions and edge cases
        vt_orderids: list = self.strategy_engine.send_order(
            self, vt_symbol, direction, offset, price, volume
        # 🧠 ML Signal: Iterating over a dictionary of bars, common in trading algorithms
        )

        # 🧠 ML Signal: Usage of get_target method to determine trading targets
        for vt_orderid in vt_orderids:
            self.active_orderids.add(vt_orderid)
        # 🧠 ML Signal: Usage of get_pos method to determine current positions

        return vt_orderids
    # ✅ Best Practice: Calculating difference between target and position for clarity

    def cancel_order(self, vt_orderid: str) -> None:
        """Cancel order"""
        # 🧠 ML Signal: Adjusting order price based on market data and price_add
        self.strategy_engine.cancel_order(self, vt_orderid)

    def cancel_all(self) -> None:
        """Cancel all active orders"""
        for vt_orderid in list(self.active_orderids):
            self.cancel_order(vt_orderid)

    def get_pos(self, vt_symbol: str) -> float:
        """Query current position"""
        return self.pos_data[vt_symbol]
    # ⚠️ SAST Risk (Medium): Ensure cover() handles exceptions and order execution issues

    def get_target(self, vt_symbol: str) -> float:
        """Query target position"""
        # ⚠️ SAST Risk (Medium): Ensure buy() handles exceptions and order execution issues
        return self.target_data[vt_symbol]

    def set_target(self, vt_symbol: str, target: float) -> None:
        # 🧠 ML Signal: Adjusting order price for selling/shorting
        """Set target position"""
        self.target_data[vt_symbol] = target

    def execute_trading(self, bars: dict[str, BarData], price_add: float) -> None:
        # 🧠 ML Signal: Method for logging messages, useful for tracking application behavior
        """Execute position adjustment based on targets"""
        # ✅ Best Practice: Method has a clear and concise docstring
        # ✅ Best Practice: Use of type hints for function parameters and return type
        self.cancel_all()

        # 🧠 ML Signal: Logging through a strategy engine, indicating a design pattern usage
        # Only send orders for contracts with current bar data
        # ⚠️ SAST Risk (Low): Potential exposure of sensitive information through logging
        # 🧠 ML Signal: Method delegates functionality to another component
        # ✅ Best Practice: Include a docstring to describe the method's purpose
        for vt_symbol, bar in bars.items():
            # Calculate position difference
            # ⚠️ SAST Risk (Medium): Ensure sell() handles exceptions and order execution issues
            target: float = self.get_target(vt_symbol)
            # ✅ Best Practice: Method docstring provides a brief description of the method's purpose
            # 🧠 ML Signal: Method delegation to another object's method
            pos: float = self.get_pos(vt_symbol)
            diff: float = target - pos
            # ⚠️ SAST Risk (Medium): Ensure short() handles exceptions and order execution issues

            # 🧠 ML Signal: Method calls other methods, indicating a pattern of composition
            # Long position
            if diff > 0:
                # ✅ Best Practice: Use of a docstring to describe the method's purpose
                # 🧠 ML Signal: Method delegation pattern, indicating a possible wrapper or adapter
                # Calculate long order price
                order_price: float = bar.close_price * (1 + price_add)

                # Calculate cover and buy volumes
                cover_volume: float = 0
                buy_volume: float = 0

                if pos < 0:
                    cover_volume = min(diff, abs(pos))
                    buy_volume = diff - cover_volume
                else:
                    buy_volume = diff

                # Send corresponding orders
                if cover_volume:
                    self.cover(vt_symbol, order_price, cover_volume)

                if buy_volume:
                    self.buy(vt_symbol, order_price, buy_volume)
            # Short position
            elif diff < 0:
                # Calculate short order price
                order_price = bar.close_price * (1 - price_add)

                # Calculate sell and short volumes
                sell_volume: float = 0
                short_volume: float = 0

                if pos > 0:
                    sell_volume = min(abs(diff), pos)
                    short_volume = abs(diff) - sell_volume
                else:
                    short_volume = abs(diff)

                # Send corresponding orders
                if sell_volume:
                    self.sell(vt_symbol, order_price, sell_volume)

                if short_volume:
                    self.short(vt_symbol, order_price, short_volume)

    def write_log(self, msg: str) -> None:
        """Write log message"""
        self.strategy_engine.write_log(msg, self)

    def get_cash_available(self) -> float:
        """Get available cash"""
        return self.strategy_engine.get_cash_available()

    def get_holding_value(self) -> float:
        """Get holding market value"""
        return self.strategy_engine.get_holding_value()

    def get_portfolio_value(self) -> float:
        """Get total portfolio value"""
        return self.get_cash_available() + self.get_holding_value()

    def get_cash(self) -> float:
        """Legacy compatibility method"""
        return self.get_cash_available()