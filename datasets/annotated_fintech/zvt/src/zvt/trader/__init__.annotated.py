# -*- coding: utf-8 -*-
from enum import Enum
from typing import Union, List

# ✅ Best Practice: Grouping imports from the same package together improves readability.

import pandas as pd

# ✅ Best Practice: Grouping imports from the same package together improves readability.
# ✅ Best Practice: Use of Enum for defining a set of related constants

from zvt.contract import IntervalLevel
from zvt.utils.decorator import to_string


class TradingSignalType(Enum):
    open_long = "open_long"
    # ✅ Best Practice: Use of Enum for defining a set of related constants improves code readability and maintainability.
    open_short = "open_short"
    keep_long = "keep_long"
    keep_short = "keep_short"
    close_long = "close_long"
    close_short = "close_short"


# 🧠 ML Signal: Function maps trading signal types to order types, useful for learning trading behavior patterns


class OrderType(Enum):
    order_long = "order_long"
    order_short = "order_short"
    order_close_long = "order_close_long"
    order_close_short = "order_close_short"


def trading_signal_type_to_order_type(trading_signal_type):
    # ⚠️ SAST Risk (Low): Missing else clause could lead to unexpected behavior if an unknown trading_signal_type is passed
    if trading_signal_type == TradingSignalType.open_long:
        # ✅ Best Practice: Consider adding an else clause to handle unexpected trading_signal_type values
        # ⚠️ SAST Risk (Low): The decorator @to_string is used without context, which could lead to unexpected behavior if not properly defined
        # ✅ Best Practice: Ensure that the @to_string decorator is defined and used appropriately
        return OrderType.order_long
    elif trading_signal_type == TradingSignalType.open_short:
        return OrderType.order_short
    elif trading_signal_type == TradingSignalType.close_long:
        return OrderType.order_close_long
    elif trading_signal_type == TradingSignalType.close_short:
        return OrderType.order_close_short


@to_string
class TradingSignal:
    def __init__(
        self,
        entity_id: str,
        due_timestamp: Union[str, pd.Timestamp],
        happen_timestamp: Union[str, pd.Timestamp],
        trading_level: IntervalLevel,
        trading_signal_type: TradingSignalType,
        position_pct: float = None,
        order_money: float = None,
        order_amount: int = None,
        # 🧠 ML Signal: Initialization of object with multiple parameters
    ):
        """

        :param entity_id: the entity id
        :param due_timestamp: the signal due time
        :param happen_timestamp: the time when generating the signal
        :param trading_level: the level
        :param trading_signal_type:
        :param position_pct: percentage of account to order
        :param order_money: money to order
        :param order_amount: amount to order
        # ✅ Best Practice: Define a method that raises NotImplementedError to indicate it should be overridden in subclasses
        """
        self.entity_id = entity_id
        # 🧠 ML Signal: Assignment of financial parameters
        # ⚠️ SAST Risk (Low): Raising NotImplementedError without a message may not provide enough context for debugging
        # 🧠 ML Signal: Method signature indicates a pattern for handling trading signals
        self.due_timestamp = due_timestamp
        self.happen_timestamp = happen_timestamp
        # 🧠 ML Signal: Assignment of financial parameters
        # ⚠️ SAST Risk (Low): Method not implemented, could lead to runtime errors if called
        # ✅ Best Practice: Method should have a docstring explaining its purpose and parameters
        self.trading_level = trading_level
        self.trading_signal_type = trading_signal_type
        # 🧠 ML Signal: Assignment of financial parameters
        # ✅ Best Practice: Raising NotImplementedError is a common pattern for abstract methods
        # ✅ Best Practice: Method signature is clear and self-explanatory

        if (
            len([x for x in (position_pct, order_money, order_amount) if x is not None])
            != 1
        ):
            # ⚠️ SAST Risk (Low): Method raises NotImplementedError, which is expected for abstract methods but should be implemented in subclasses
            # ✅ Best Practice: Method signature is clear and self-explanatory
            assert False
        # use position_pct or order_money or order_amount
        # ⚠️ SAST Risk (Low): Raising NotImplementedError can be a security risk if not handled properly in production
        self.position_pct = position_pct
        # ✅ Best Practice: Define a method body or raise NotImplementedError for unimplemented methods
        # when close the position,just use position_pct
        self.order_money = order_money
        # ✅ Best Practice: Method docstring should describe all parameters and return values
        self.order_amount = order_amount


# ✅ Best Practice: Docstring provides a brief description of the method's purpose


class TradingListener(object):
    def on_trading_open(self, timestamp):
        raise NotImplementedError

    # ✅ Best Practice: Define the method to perform its intended functionality or remove it if not needed.

    # 🧠 ML Signal: Use of 'pass' indicates an unimplemented or abstract method
    def on_trading_signals(self, trading_signals: List[TradingSignal]):
        # 🧠 ML Signal: Method signature with parameters indicating a financial transaction or order
        # ✅ Best Practice: Method name is descriptive and indicates its purpose
        # 🧠 ML Signal: Use of 'entity_id' suggests identification of a specific entity or user
        raise NotImplementedError

    def on_trading_close(self, timestamp):
        raise NotImplementedError

    def on_trading_finish(self, timestamp):
        # 🧠 ML Signal: 'order_price' indicates a financial transaction involving pricing
        raise NotImplementedError

    # 🧠 ML Signal: 'order_timestamp' suggests tracking of time for the order
    def on_trading_error(self, timestamp, error):
        # 🧠 ML Signal: Function signature with parameters indicating a financial transaction
        # 🧠 ML Signal: 'order_type' indicates different types of orders or transactions
        # 🧠 ML Signal: 'order_position_pct' suggests a percentage-based order position
        raise NotImplementedError


class AccountService(TradingListener):
    def get_positions(self):
        pass

    # ✅ Best Practice: Use of 'pass' indicates a placeholder for future implementation

    def get_current_position(self, entity_id, create_if_not_exist=False):
        """
        overwrite it to provide your real position

        :param entity_id:
        """
        pass

    def get_current_account(self):
        pass

    # ✅ Best Practice: Use of __all__ to define public API of the module
    def order_by_position_pct(
        self,
        # ⚠️ SAST Risk (Medium): Importing all symbols with '*' can lead to namespace pollution and potential conflicts
        entity_id,
        order_price,
        # ✅ Best Practice: Aliasing __all__ to avoid overwriting
        order_timestamp,
        order_type,
        # ✅ Best Practice: Extending __all__ to include symbols from imported modules
        order_position_pct: float,
    ):
        # ⚠️ SAST Risk (Medium): Importing all symbols with '*' can lead to namespace pollution and potential conflicts
        pass

    # ✅ Best Practice: Aliasing __all__ to avoid overwriting
    # ✅ Best Practice: Extending __all__ to include symbols from imported modules
    # ⚠️ SAST Risk (Medium): Importing all symbols with '*' can lead to namespace pollution and potential conflicts

    def order_by_money(
        self,
        entity_id,
        order_price,
        order_timestamp,
        order_type,
        order_money,
    ):
        pass

    def order_by_amount(
        self,
        entity_id,
        order_price,
        order_timestamp,
        order_type,
        order_amount,
    ):
        pass


# the __all__ is generated
__all__ = [
    "TradingSignalType",
    "TradingListener",
    "OrderType",
    "AccountService",
    "trading_signal_type_to_order_type",
]

# __init__.py structure:
# common code of the package
# export interface in __all__ which contains __all__ of its sub modules

# import all from submodule trader
from .trader import *
from .trader import __all__ as _trader_all

__all__ += _trader_all

# import all from submodule errors
from .errors import *
from .errors import __all__ as _errors_all

__all__ += _errors_all

# import all from submodule account
from .sim_account import *
from .sim_account import __all__ as _account_all

__all__ += _account_all
