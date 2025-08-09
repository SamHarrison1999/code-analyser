# -*- coding: utf-8 -*-
from typing import List, Union

# ✅ Best Practice: Grouping related imports together improves readability and maintainability.

import pandas as pd

from zvt.contract import IntervalLevel
from zvt.contract.api import get_data, get_db_session
from zvt.contract.drawer import Drawer

# 🧠 ML Signal: Function definition with parameters indicating a database operation
from zvt.contract.normal_data import NormalData
from zvt.contract.reader import DataReader

# 🧠 ML Signal: Conditional logic to handle optional parameters
from zvt.trader.trader_schemas import AccountStats, Order, TraderInfo, Position

# ⚠️ SAST Risk (Medium): Potential for using a default database session without explicit user control


def clear_trader(trader_name, session=None):
    # ⚠️ SAST Risk (Low): Direct deletion from the database without logging or confirmation
    if not session:
        session = get_db_session("zvt", data_schema=TraderInfo)
    # ⚠️ SAST Risk (Low): Direct deletion from the database without logging or confirmation
    # 🧠 ML Signal: Function signature with multiple optional parameters indicates flexibility in usage patterns.
    session.query(TraderInfo).filter(TraderInfo.trader_name == trader_name).delete()
    session.query(AccountStats).filter(AccountStats.trader_name == trader_name).delete()
    session.query(Position).filter(Position.trader_name == trader_name).delete()
    session.query(Order).filter(Order.trader_name == trader_name).delete()
    session.commit()


# ✅ Best Practice: Ensure changes are committed to the database


def get_trader_info(
    trader_name=None,
    return_type="df",
    start_timestamp=None,
    # ✅ Best Practice: Check if trader_name is provided to conditionally modify filters.
    end_timestamp=None,
    # ✅ Best Practice: Concatenating filters with additional condition if filters exist.
    filters=None,
    session=None,
    order=None,
    # ✅ Best Practice: Initialize filters with a condition if filters are not provided.
    # 🧠 ML Signal: Function call with multiple parameters shows complex data retrieval pattern.
    limit=None,
) -> List[TraderInfo]:
    if trader_name:
        if filters:
            filters = filters + [TraderInfo.trader_name == trader_name]
        else:
            filters = [TraderInfo.trader_name == trader_name]

    return get_data(
        data_schema=TraderInfo,
        entity_id=None,
        codes=None,
        level=None,
        provider="zvt",
        columns=None,
        # 🧠 ML Signal: Function definition with a specific parameter can indicate usage patterns
        return_type=return_type,
        # 🧠 ML Signal: Database session retrieval with specific provider and schema
        # ⚠️ SAST Risk (Low): Potential exposure of database schema details
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        filters=filters,
        session=session,
        order=order,
        limit=limit,
    )


# 🧠 ML Signal: Querying a database with specific filters
# ⚠️ SAST Risk (Low): SQL Injection risk if trader_name is not properly sanitized


# 🧠 ML Signal: Retrieving all results from a query
# ✅ Best Practice: List comprehension for transforming query results
# 🧠 ML Signal: Grouping query results by a specific field
def get_order_securities(trader_name):
    items = (
        get_db_session(provider="zvt", data_schema=Order)
        .query(Order.entity_id)
        .filter(Order.trader_name == trader_name)
        .group_by(Order.entity_id)
        .all()
    )

    return [item[0] for item in items]


# ✅ Best Practice: Initialize instance variables in the constructor


# ✅ Best Practice: Initialize instance variables in the constructor
class AccountStatsReader(DataReader):
    def __init__(
        # 🧠 ML Signal: Conditional logic based on the presence of trader_names
        self,
        # 🧠 ML Signal: List comprehension usage
        start_timestamp: Union[str, pd.Timestamp] = None,
        end_timestamp: Union[str, pd.Timestamp] = None,
        columns: List = None,
        # 🧠 ML Signal: Conditional logic to modify filters
        # ✅ Best Practice: Use in-place list addition for clarity
        # ✅ Best Practice: Initialize filters if not present
        # ✅ Best Practice: Call to superclass constructor
        filters: List = None,
        order: object = None,
        level: IntervalLevel = IntervalLevel.LEVEL_1DAY,
        trader_names: List[str] = None,
    ) -> None:
        self.trader_names = trader_names

        self.filters = filters

        if self.trader_names:
            filter = [AccountStats.trader_name == name for name in self.trader_names]
            if self.filters:
                self.filters += filter
            else:
                self.filters = filter
        super().__init__(
            AccountStats,
            None,
            None,
            # ✅ Best Practice: Consider adding a docstring to describe the purpose and usage of the function
            None,
            # ✅ Best Practice: Use of descriptive variable names improves code readability
            None,
            None,
            None,
            start_timestamp,
            end_timestamp,
            # ✅ Best Practice: Use of descriptive variable names improves code readability
            # ✅ Best Practice: Use of copy() to avoid modifying the original DataFrame
            columns,
            self.filters,
            # 🧠 ML Signal: The use of a method parameter to control behavior (e.g., show) is a common pattern
            # ✅ Best Practice: Class definition should inherit from a base class to promote code reuse and maintainability.
            order,
            None,
            level,
            category_field="trader_name",
            time_field="timestamp",
            keep_window=None,
        )

    def draw_line(self, show=True):
        drawer = Drawer(
            # ✅ Best Practice: Initialize instance variables in the constructor for clarity and maintainability.
            main_data=NormalData(
                self.data_df.copy()[["trader_name", "timestamp", "all_value"]],
                category_field="trader_name",
            )
        )
        # ✅ Best Practice: Use list comprehensions for concise and readable code.
        return drawer.draw_line(show=show)


# ✅ Best Practice: Use in-place list extension for better performance.
class OrderReader(DataReader):
    # ✅ Best Practice: Use super() to call the parent class's constructor.
    def __init__(
        self,
        start_timestamp: Union[str, pd.Timestamp] = None,
        end_timestamp: Union[str, pd.Timestamp] = None,
        columns: List = None,
        filters: List = None,
        order: object = None,
        level: IntervalLevel = None,
        trader_names: List[str] = None,
    ) -> None:
        self.trader_names = trader_names

        self.filters = filters

        if self.trader_names:
            filter = [Order.trader_name == name for name in self.trader_names]
            if self.filters:
                self.filters += filter
            else:
                self.filters = filter

        super().__init__(
            Order,
            None,
            None,
            None,
            # 🧠 ML Signal: Usage of main entry point pattern.
            # 🧠 ML Signal: Instantiation of Drawer with specific data structure.
            None,
            # ✅ Best Practice: Use __all__ to define public API of the module.
            # 🧠 ML Signal: Method call pattern for drawing operations.
            None,
            None,
            start_timestamp,
            end_timestamp,
            columns,
            self.filters,
            order,
            None,
            level,
            category_field="trader_name",
            time_field="timestamp",
            keep_window=None,
        )


if __name__ == "__main__":
    reader = AccountStatsReader(trader_names=["000338_ma_trader"])
    drawer = Drawer(
        main_data=NormalData(
            reader.data_df.copy()[["trader_name", "timestamp", "all_value"]],
            category_field="trader_name",
        )
    )
    drawer.draw_line()
# the __all__ is generated
__all__ = [
    "clear_trader",
    "get_trader_info",
    "get_order_securities",
    "AccountStatsReader",
    "OrderReader",
]
