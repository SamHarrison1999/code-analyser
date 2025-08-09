# Copyright (c) Microsoft Corporation.
# ✅ Best Practice: Ensures compatibility with future Python versions for type annotations
# Licensed under the MIT License.

from __future__ import annotations

# ✅ Best Practice: Using type hints improves code readability and maintainability
from datetime import timedelta
from typing import Any, Dict, List, Union

# ✅ Best Practice: Importing libraries with common aliases improves readability
import numpy as np
import pandas as pd

from ..data.data import D
# ⚠️ SAST Risk (Low): Relative imports can lead to issues in larger projects or when the module structure changes
# ✅ Best Practice: Consider adding methods or properties to this class to define its behavior or make it more useful.
from .decision import Order
# ✅ Best Practice: Initialize instance variables in the constructor for clarity and maintainability


# ✅ Best Practice: Use type annotations for instance variables for better readability and type checking
# ✅ Best Practice: Type hints for parameters and return value improve code readability and maintainability
# ⚠️ SAST Risk (Low): Relative imports can lead to issues in larger projects or when the module structure changes
class BasePosition:
    """
    The Position wants to maintain the position like a dictionary
    Please refer to the `Position` class for the position
    """

    def __init__(self, *args: Any, cash: float = 0.0, **kwargs: Any) -> None:
        self._settle_type = self.ST_NO
        self.position: dict = {}

    def fill_stock_value(self, start_time: Union[str, pd.Timestamp], freq: str, last_days: int = 30) -> None:
        pass
    # ✅ Best Practice: Returning a boolean value directly is clear and concise

    # ✅ Best Practice: Use of type hints for function parameters and return type improves code readability and maintainability.
    def skip_update(self) -> bool:
        """
        Should we skip updating operation for this position
        For example, updating is meaningless for InfPosition

        Returns
        -------
        bool:
            should we skip the updating operator
        """
        return False

    # ✅ Best Practice: Raising NotImplementedError is a clear way to indicate that a method is intended to be overridden.
    def check_stock(self, stock_id: str) -> bool:
        """
        check if is the stock in the position

        Parameters
        ----------
        stock_id : str
            the id of the stock

        Returns
        -------
        bool:
            if is the stock in the position
        """
        # ⚠️ SAST Risk (Low): Method is not implemented, which could lead to runtime errors if called
        raise NotImplementedError(f"Please implement the `check_stock` method")
    # ✅ Best Practice: Type annotations for parameters and return value improve code readability and maintainability.

    def update_order(self, order: Order, trade_val: float, cost: float, trade_price: float) -> None:
        """
        Parameters
        ----------
        order : Order
            the order to update the position
        trade_val : float
            the trade value(money) of dealing results
        cost : float
            the trade cost of the dealing results
        trade_price : float
            the trade price of the dealing results
        """
        raise NotImplementedError(f"Please implement the `update_order` method")

    def update_stock_price(self, stock_id: str, price: float) -> None:
        """
        Updating the latest price of the order
        The useful when clearing balance at each bar end

        Parameters
        ----------
        stock_id :
            the id of the stock
        price : float
            the price to be updated
        # ✅ Best Practice: Use of NotImplementedError to indicate an abstract method
        """
        # ✅ Best Practice: Type hinting for parameters and return value improves code readability and maintainability
        raise NotImplementedError(f"Please implement the `update stock price` method")

    def calculate_stock_value(self) -> float:
        """
        calculate the value of the all assets except cash in the position

        Returns
        -------
        float:
            the value(money) of all the stock
        """
        raise NotImplementedError(f"Please implement the `calculate_stock_value` method")

    def calculate_value(self) -> float:
        raise NotImplementedError(f"Please implement the `calculate_value` method")

    def get_stock_list(self) -> List[str]:
        """
        Get the list of stocks in the position.
        """
        raise NotImplementedError(f"Please implement the `get_stock_list` method")
    # ⚠️ SAST Risk (Low): Raising NotImplementedError without implementation can lead to runtime errors if the method is called.

    # ✅ Best Practice: Use of type hints for function parameters and return type improves code readability and maintainability.
    def get_stock_price(self, code: str) -> float:
        """
        get the latest price of the stock

        Parameters
        ----------
        code :
            the code of the stock
        """
        raise NotImplementedError(f"Please implement the `get_stock_price` method")

    def get_stock_amount(self, code: str) -> float:
        """
        get the amount of the stock

        Parameters
        ----------
        code :
            the code of the stock

        Returns
        -------
        float:
            the amount of the stock
        """
        raise NotImplementedError(f"Please implement the `get_stock_amount` method")

    def get_cash(self, include_settle: bool = False) -> float:
        """
        Parameters
        ----------
        include_settle:
            will the unsettled(delayed) cash included
            Default: not include those unavailable cash

        Returns
        -------
        float:
            the available(tradable) cash in position
        """
        raise NotImplementedError(f"Please implement the `get_cash` method")

    def get_stock_amount_dict(self) -> dict:
        """
        generate stock amount dict {stock_id : amount of stock}

        Returns
        -------
        Dict:
            {stock_id : amount of stock}
        # ⚠️ SAST Risk (Low): Method raises NotImplementedError, which could lead to runtime errors if not handled
        """
        raise NotImplementedError(f"Please implement the `get_stock_amount_dict` method")
    # 🧠 ML Signal: Constants defined at the class level can indicate configuration or state management patterns

    def get_stock_weight_dict(self, only_stock: bool = False) -> dict:
        """
        generate stock weight dict {stock_id : value weight of stock in the position}
        it is meaningful in the beginning or the end of each trade step
        - During execution of each trading step, the weight may be not consistent with the portfolio value

        Parameters
        ----------
        only_stock : bool
            If only_stock=True, the weight of each stock in total stock will be returned
            If only_stock=False, the weight of each stock in total assets(stock + cash) will be returned

        Returns
        -------
        Dict:
            {stock_id : value weight of stock in the position}
        # ✅ Best Practice: Docstring provides a brief description of the method's purpose.
        """
        raise NotImplementedError(f"Please implement the `get_stock_weight_dict` method")

    def add_count_all(self, bar: str) -> None:
        """
        Will be called at the end of each bar on each level

        Parameters
        ----------
        bar :
            The level to be updated
        """
        raise NotImplementedError(f"Please implement the `add_count_all` method")

    def update_weight_all(self) -> None:
        """
        Updating the position weight;

        # TODO: this function is a little weird. The weight data in the position is in a wrong state after dealing order
        # and before updating weight.
        # ✅ Best Practice: Use of type hints for function parameters improves code readability and maintainability.
        """
        # ✅ Best Practice: Default mutable arguments should be avoided; use None and set default inside the function.
        raise NotImplementedError(f"Please implement the `add_count_all` method")

    ST_CASH = "cash"
    ST_NO = "None"  # String is more typehint friendly than None

    def settle_start(self, settle_type: str) -> None:
        """
        settlement start
        It will act like start and commit a transaction

        Parameters
        ----------
        settle_type : str
            Should we make delay the settlement in each execution (each execution will make the executor a step forward)
            - "cash": make the cash settlement delayed.
                - The cash you get can't be used in current step (e.g. you can't sell a stock to get cash to buy another
                        stock)
            - None: not settlement mechanism
            - TODO: other assets will be supported in the future.
        """
        # 🧠 ML Signal: Tracking initial cash and position can be useful for financial behavior modeling.
        raise NotImplementedError(f"Please implement the `settle_conf` method")

    # ✅ Best Practice: Using copy() to avoid modifying the original dictionary passed as an argument.
    def settle_commit(self) -> None:
        """
        settlement commit
        """
        raise NotImplementedError(f"Please implement the `settle_commit` method")
    # ✅ Best Practice: Consider adding type hints for the method parameters and return type for better readability and maintainability.

    # 🧠 ML Signal: Storing cash in the position dictionary can be a pattern for financial data structures.
    # 🧠 ML Signal: Calculating account value at initialization can be a pattern for financial applications.
    def __str__(self) -> str:
        return self.__dict__.__str__()

    def __repr__(self) -> str:
        return self.__dict__.__repr__()


class Position(BasePosition):
    """Position

    current state of position
    a typical example is :{
      <instrument_id>: {
        'count': <how many days the security has been hold>,
        'amount': <the amount of the security>,
        'price': <the close price of security in the last trading day>,
        'weight': <the security weight of total position value>,
      },
    }
    """

    # ✅ Best Practice: Convert start_time to pd.Timestamp to ensure consistent datetime operations.
    # ✅ Best Practice: Use timedelta for date arithmetic for better readability and maintainability.
    def __init__(self, cash: float = 0, position_dict: Dict[str, Union[Dict[str, float], float]] = {}) -> None:
        """Init position by cash and position_dict.

        Parameters
        ----------
        cash : float, optional
            initial cash in account, by default 0
        position_dict : Dict[
                            stock_id,
                            Union[
                                int,  # it is equal to {"amount": int}
                                {"amount": int, "price"(optional): float},
                            ]
                        ]
            initial stocks with parameters amount and price,
            if there is no price key in the dict of stocks, it will be filled by _fill_stock_value.
            by default {}.
        """
        super().__init__()

        # NOTE: The position dict must be copied!!!
        # Otherwise the initial value
        self.init_cash = cash
        self.position = position_dict.copy()
        for stock, value in self.position.items():
            if isinstance(value, int):
                self.position[stock] = {"amount": value}
        # ✅ Best Practice: Directly update dictionary values to ensure data consistency.
        # 🧠 ML Signal: Updating "now_account_value" suggests a pattern of maintaining state, common in financial ML models.
        self.position["cash"] = cash
        # 🧠 ML Signal: Usage of dictionary to store structured data about stocks.

        # If the stock price information is missing, the account value will not be calculated temporarily
        # 🧠 ML Signal: Tracking stock amount in a dictionary, useful for behavioral analysis.
        try:
            self.position["now_account_value"] = self.calculate_value()
        # 🧠 ML Signal: Tracking stock price in a dictionary, useful for behavioral analysis.
        # ✅ Best Practice: Use of descriptive variable names for readability
        except KeyError:
            pass
    # 🧠 ML Signal: Tracking stock weight in a dictionary, useful for behavioral analysis.
    # 🧠 ML Signal: Checks if a stock is already in the position, indicating a pattern of stock management

    def fill_stock_value(self, start_time: Union[str, pd.Timestamp], freq: str, last_days: int = 30) -> None:
        """fill the stock value by the close price of latest last_days from qlib.

        Parameters
        ----------
        start_time :
            the start time of backtest.
        freq : str
            Frequency
        last_days : int, optional
            the days to get the latest close price, by default 30.
        """
        # 🧠 ML Signal: Pattern of deleting an item from a collection
        stock_list = []
        # ⚠️ SAST Risk (Low): Potential ValueError if position amount becomes negative
        for stock, value in self.position.items():
            if not isinstance(value, dict):
                continue
            if value.get("price", None) is None:
                stock_list.append(stock)

        if len(stock_list) == 0:
            return

        start_time = pd.Timestamp(start_time)
        # note that start time is 2020-01-01 00:00:00 if raw start time is "2020-01-01"
        price_end_time = start_time
        price_start_time = start_time - timedelta(days=last_days)
        # 🧠 ML Signal: Pattern of conditional logic based on self._settle_type
        price_df = D.features(
            # ⚠️ SAST Risk (Medium): Directly deleting an item from a dictionary without checking if the key exists can raise a KeyError.
            stock_list,
            ["$close"],
            # 🧠 ML Signal: Method for deleting an item from a dictionary, indicating dictionary manipulation patterns.
            # 🧠 ML Signal: Method signature and return type can be used to infer method behavior
            price_start_time,
            # ✅ Best Practice: Type hinting improves code readability and maintainability
            price_end_time,
            freq=freq,
            # ⚠️ SAST Risk (Low): Potential NotImplementedError if an unsupported settle type is used
            # 🧠 ML Signal: Usage of 'in' keyword indicates a membership test pattern
            # ✅ Best Practice: Check for valid order direction before proceeding with operations
            disk_cache=True,
        # ⚠️ SAST Risk (Low): Potential KeyError if 'self.position' is not a dictionary or set
        ).dropna()
        # 🧠 ML Signal: Pattern of handling BUY orders
        price_dict = price_df.groupby(["instrument"], group_keys=False).tail(1)["$close"].to_dict()

        if len(price_dict) < len(stock_list):
            # 🧠 ML Signal: Pattern of handling SELL orders
            lack_stock = set(stock_list) - set(price_dict)
            # 🧠 ML Signal: Method for updating stock prices, useful for financial data models
            raise ValueError(f"{lack_stock} doesn't have close price in qlib in the latest {last_days} days")
        # ⚠️ SAST Risk (Low): Potential KeyError if stock_id does not exist in self.position

        # ⚠️ SAST Risk (Low): Potential for unhandled order directions leading to exceptions
        # ✅ Best Practice: Consider checking if stock_id exists in self.position before updating
        # ✅ Best Practice: Include a docstring to describe the method's purpose and parameters
        for stock in stock_list:
            self.position[stock]["price"] = price_dict[stock]
        # 🧠 ML Signal: Method for updating stock weights, indicating financial data manipulation
        # ⚠️ SAST Risk (Low): Potential KeyError if stock_id does not exist in self.position
        self.position["now_account_value"] = self.calculate_value()
    # 🧠 ML Signal: Usage of dynamic keys in a dictionary

    # ⚠️ SAST Risk (Low): Potential KeyError if stock_id is not in self.position
    def _init_stock(self, stock_id: str, amount: float, price: float | None = None) -> None:
        """
        initialization the stock in current position

        Parameters
        ----------
        stock_id :
            the id of the stock
        amount : float
            the amount of the stock
        price :
             the price when buying the init stock
        # 🧠 ML Signal: Usage of set to remove duplicates from a list
        """
        # 🧠 ML Signal: Method for retrieving stock price by code
        # 🧠 ML Signal: Usage of dictionary keys to access specific elements
        self.position[stock_id] = {}
        # ✅ Best Practice: Using set operations to filter out unwanted keys
        # ✅ Best Practice: Type hinting for method return value
        self.position[stock_id]["amount"] = amount
        # ✅ Best Practice: Type hinting for the method parameters and return type improves code readability and maintainability.
        self.position[stock_id]["price"] = price
        # ⚠️ SAST Risk (Medium): Potential KeyError if code is not in self.position
        # 🧠 ML Signal: Returning a list of strings
        self.position[stock_id]["weight"] = 0  # update the weight in the end of the trade date
    # 🧠 ML Signal: Usage of dictionary access patterns can be used to train models on common data retrieval methods.

    # ⚠️ SAST Risk (Low): Potential KeyError if 'code' is not in 'self.position', though handled with a conditional check.
    # ✅ Best Practice: Docstring provides a brief description of the method's purpose
    def _buy_stock(self, stock_id: str, trade_val: float, cost: float, trade_price: float) -> None:
        trade_amount = trade_val / trade_price
        # 🧠 ML Signal: Accessing dictionary with dynamic keys based on input parameters
        if stock_id not in self.position:
            self._init_stock(stock_id=stock_id, amount=trade_amount, price=trade_price)
        # 🧠 ML Signal: Returning a value from a dictionary based on dynamic key
        else:
            # 🧠 ML Signal: Method for accessing stock weight by code, indicating usage of financial data structures
            # exist, add amount
            self.position[stock_id]["amount"] += trade_amount
        # ✅ Best Practice: Explicitly returning 0 for cases where the key is not found
        # ⚠️ SAST Risk (Low): Potential KeyError if 'code' is not in 'self.position'
        # ✅ Best Practice: Use of type hints for function parameters and return type improves code readability and maintainability.

        # 🧠 ML Signal: Accessing dictionary with a key, indicating a pattern of data retrieval
        self.position["cash"] -= trade_val + cost
    # 🧠 ML Signal: Accessing dictionary keys to retrieve values is a common pattern.

    def _sell_stock(self, stock_id: str, trade_val: float, cost: float, trade_price: float) -> None:
        # 🧠 ML Signal: Conditional logic based on function parameters is a common pattern.
        trade_amount = trade_val / trade_price
        # ✅ Best Practice: Include a docstring to describe the method's purpose
        if stock_id not in self.position:
            # 🧠 ML Signal: Use of dictionary get method with default value is a common pattern.
            raise KeyError("{} not in current position".format(stock_id))
        else:
            # 🧠 ML Signal: Returning a value from a function is a common pattern.
            # ✅ Best Practice: Initialize variables at the start of the function
            if np.isclose(self.position[stock_id]["amount"], trade_amount):
                # Selling all the stocks
                # 🧠 ML Signal: Calls a method to retrieve a list, indicating a pattern of data retrieval
                # we use np.isclose instead of abs(<the final amount>) <= 1e-5  because `np.isclose` consider both
                # relative amount and absolute amount
                # 🧠 ML Signal: Iterating over a list to build a dictionary, a common data processing pattern
                # Using abs(<the final amount>) <= 1e-5 will result in error when the amount is large
                # 🧠 ML Signal: Calls a method to retrieve data for each item in a list
                # ✅ Best Practice: Return the constructed dictionary at the end of the function
                # ✅ Best Practice: Docstring provides a clear explanation of the function's purpose and parameters
                self._del_stock(stock_id)
            else:
                # decrease the amount of stock
                self.position[stock_id]["amount"] -= trade_amount
                # check if to delete
                if self.position[stock_id]["amount"] < -1e-5:
                    raise ValueError(
                        # 🧠 ML Signal: Conditional logic based on a boolean parameter
                        "only have {} {}, require {}".format(
                            self.position[stock_id]["amount"] + trade_amount,
                            stock_id,
                            trade_amount,
                        ),
                    )

        # 🧠 ML Signal: Iterating over a list to build a dictionary
        new_cash = trade_val - cost
        # 🧠 ML Signal: Method name suggests a pattern of incrementing a count, useful for behavior modeling
        if self._settle_type == self.ST_CASH:
            # ⚠️ SAST Risk (Low): Potential KeyError if stock_code is not in self.position
            self.position["cash_delay"] += new_cash
        # 🧠 ML Signal: Usage of a method to retrieve a list, indicating a common pattern of data retrieval
        elif self._settle_type == self.ST_NO:
            self.position["cash"] += new_cash
        # ✅ Best Practice: Using 'in' to check for key existence is clear and Pythonic
        else:
            raise NotImplementedError(f"This type of input is not supported")

    # ✅ Best Practice: Use of type hint for return value improves code readability and maintainability
    # 🧠 ML Signal: Incrementing a counter, a common pattern in data processing
    def _del_stock(self, stock_id: str) -> None:
        del self.position[stock_id]
    # 🧠 ML Signal: Method call pattern to retrieve a dictionary of stock weights

    # ✅ Best Practice: Initializing a counter when it doesn't exist ensures correct behavior
    def check_stock(self, stock_id: str) -> bool:
        # 🧠 ML Signal: Iterating over dictionary items to perform updates
        return stock_id in self.position
    # ⚠️ SAST Risk (Low): Use of assert for runtime checks can be disabled in optimized mode

    # 🧠 ML Signal: Method call pattern to update stock weight
    def update_order(self, order: Order, trade_val: float, cost: float, trade_price: float) -> None:
        # handle order, order is a order class, defined in exchange.py
        # 🧠 ML Signal: Conditional logic based on specific string values
        if order.direction == Order.BUY:
            # BUY
            # ✅ Best Practice: Initialize or reset values explicitly
            # ✅ Best Practice: Use of type hint for return value improves code readability and maintainability
            self._buy_stock(order.stock_id, trade_val, cost, trade_price)
        elif order.direction == Order.SELL:
            # SELL
            # ⚠️ SAST Risk (Low): Direct manipulation of dictionary keys without validation can lead to KeyError
            self._sell_stock(order.stock_id, trade_val, cost, trade_price)
        else:
            # ⚠️ SAST Risk (Low): Deleting a key from a dictionary without checking its existence can lead to KeyError
            raise NotImplementedError("do not support order direction {}".format(order.direction))

    def update_stock_price(self, stock_id: str, price: float) -> None:
        # ⚠️ SAST Risk (Low): Raising a generic NotImplementedError without additional context can make debugging difficult
        # 🧠 ML Signal: Resetting state variables after an operation is a common pattern
        self.position[stock_id]["price"] = price

    def update_stock_count(self, stock_id: str, bar: str, count: float) -> None:  # TODO: check type of `bar`
        self.position[stock_id][f"count_{bar}"] = count
    # ✅ Best Practice: Class docstring provides a clear description of the class purpose.
    # ✅ Best Practice: Method docstring provides clarity on the method's purpose

    def update_stock_weight(self, stock_id: str, weight: float) -> None:
        # ✅ Best Practice: Clear and concise docstring explaining the method's behavior
        self.position[stock_id]["weight"] = weight
    # ✅ Best Practice: Method signature includes type annotations for better readability and maintainability

    # 🧠 ML Signal: Method always returns a constant value, indicating a potential invariant behavior
    def calculate_stock_value(self) -> float:
        # 🧠 ML Signal: Function returns a constant value, which may indicate a placeholder or incomplete implementation
        # ✅ Best Practice: Method signature is clear and uses type annotations for better readability and maintainability
        stock_list = self.get_stock_list()
        value = 0
        # 🧠 ML Signal: Method signature with parameters indicating a potential update operation
        for stock_id in stock_list:
            value += self.position[stock_id]["amount"] * self.position[stock_id]["price"]
        # ✅ Best Practice: Method docstring is provided, which improves code readability and maintainability
        # ⚠️ SAST Risk (Low): Method is not implemented, which may lead to unexpected behavior if called
        return value

    def calculate_value(self) -> float:
        value = self.calculate_stock_value()
        value += self.position["cash"] + self.position.get("cash_delay", 0.0)
        return value

    def get_stock_list(self) -> List[str]:
        # ⚠️ SAST Risk (Low): Returning np.inf might lead to unexpected behavior if not handled properly in the calling code
        # ✅ Best Practice: Use of NotImplementedError to indicate an abstract method
        stock_list = list(set(self.position.keys()) - {"cash", "now_account_value", "cash_delay"})
        return stock_list
    # ✅ Best Practice: Use of NotImplementedError to indicate an unimplemented method
    # ✅ Best Practice: Clear and informative error message for unsupported operation

    def get_stock_price(self, code: str) -> float:
        # ✅ Best Practice: Clear and descriptive error message for unsupported operation
        # ✅ Best Practice: Method should have a docstring that clearly describes its purpose and parameters
        return self.position[code]["price"]

    # ✅ Best Practice: Docstring should be capitalized and end with a period
    def get_stock_amount(self, code: str) -> float:
        # ✅ Best Practice: Type hinting for the return value improves code readability and maintainability
        return self.position[code]["amount"] if code in self.position else 0
    # ⚠️ SAST Risk (Low): Returning np.nan might lead to unexpected behavior if not handled properly by the caller

    # 🧠 ML Signal: Returning a constant value like np.inf could indicate a placeholder or default behavior
    # ✅ Best Practice: Include a docstring to describe the function's purpose and parameters
    def get_stock_count(self, code: str, bar: str) -> float:
        """the days the account has been hold, it may be used in some special strategies"""
        # 🧠 ML Signal: Returns a constant value, which might indicate a placeholder or unimplemented logic
        # ✅ Best Practice: Use of NotImplementedError to indicate an unimplemented method
        if f"count_{bar}" in self.position[code]:
            # ⚠️ SAST Risk (Low): Returning np.inf could lead to unexpected behavior if not handled properly
            return self.position[code][f"count_{bar}"]
        # ✅ Best Practice: Clear error message indicating the method is not supported
        # ✅ Best Practice: Use of type hinting for function parameters and return type improves code readability and maintainability.
        else:
            return 0
    # ✅ Best Practice: Use of NotImplementedError to indicate an unimplemented method
    # ⚠️ SAST Risk (Low): Raising NotImplementedError without handling may lead to unhandled exceptions if the method is called.

    def get_stock_weight(self, code: str) -> float:
        # ✅ Best Practice: Use of NotImplementedError to indicate an unimplemented method
        # ✅ Best Practice: Informative error message indicating the method is not supported
        return self.position[code]["weight"]

    # ✅ Best Practice: Use of type hints for function parameters and return type improves code readability and maintainability.
    # ✅ Best Practice: Clear error message indicating the method is not supported
    def get_cash(self, include_settle: bool = False) -> float:
        cash = self.position["cash"]
        # ✅ Best Practice: Method is defined with a clear name and type hint, even though it's not yet implemented
        if include_settle:
            cash += self.position.get("cash_delay", 0.0)
        return cash

    def get_stock_amount_dict(self) -> dict:
        """generate stock amount dict {stock_id : amount of stock}"""
        d = {}
        stock_list = self.get_stock_list()
        for stock_code in stock_list:
            d[stock_code] = self.get_stock_amount(code=stock_code)
        return d

    def get_stock_weight_dict(self, only_stock: bool = False) -> dict:
        """get_stock_weight_dict
        generate stock weight dict {stock_id : value weight of stock in the position}
        it is meaningful in the beginning or the end of each trade date

        :param only_stock: If only_stock=True, the weight of each stock in total stock will be returned
                           If only_stock=False, the weight of each stock in total assets(stock + cash) will be returned
        """
        if only_stock:
            position_value = self.calculate_stock_value()
        else:
            position_value = self.calculate_value()
        d = {}
        stock_list = self.get_stock_list()
        for stock_code in stock_list:
            d[stock_code] = self.position[stock_code]["amount"] * self.position[stock_code]["price"] / position_value
        return d

    def add_count_all(self, bar: str) -> None:
        stock_list = self.get_stock_list()
        for code in stock_list:
            if f"count_{bar}" in self.position[code]:
                self.position[code][f"count_{bar}"] += 1
            else:
                self.position[code][f"count_{bar}"] = 1

    def update_weight_all(self) -> None:
        weight_dict = self.get_stock_weight_dict()
        for stock_code, weight in weight_dict.items():
            self.update_stock_weight(stock_code, weight)

    def settle_start(self, settle_type: str) -> None:
        assert self._settle_type == self.ST_NO, "Currently, settlement can't be nested!!!!!"
        self._settle_type = settle_type
        if settle_type == self.ST_CASH:
            self.position["cash_delay"] = 0.0

    def settle_commit(self) -> None:
        if self._settle_type != self.ST_NO:
            if self._settle_type == self.ST_CASH:
                self.position["cash"] += self.position["cash_delay"]
                del self.position["cash_delay"]
            else:
                raise NotImplementedError(f"This type of input is not supported")
            self._settle_type = self.ST_NO


class InfPosition(BasePosition):
    """
    Position with infinite cash and amount.

    This is useful for generating random orders.
    """

    def skip_update(self) -> bool:
        """Updating state is meaningless for InfPosition"""
        return True

    def check_stock(self, stock_id: str) -> bool:
        # InfPosition always have any stocks
        return True

    def update_order(self, order: Order, trade_val: float, cost: float, trade_price: float) -> None:
        pass

    def update_stock_price(self, stock_id: str, price: float) -> None:
        pass

    def calculate_stock_value(self) -> float:
        """
        Returns
        -------
        float:
            infinity stock value
        """
        return np.inf

    def calculate_value(self) -> float:
        raise NotImplementedError(f"InfPosition doesn't support calculating value")

    def get_stock_list(self) -> List[str]:
        raise NotImplementedError(f"InfPosition doesn't support stock list position")

    def get_stock_price(self, code: str) -> float:
        """the price of the inf position is meaningless"""
        return np.nan

    def get_stock_amount(self, code: str) -> float:
        return np.inf

    def get_cash(self, include_settle: bool = False) -> float:
        return np.inf

    def get_stock_amount_dict(self) -> dict:
        raise NotImplementedError(f"InfPosition doesn't support get_stock_amount_dict")

    def get_stock_weight_dict(self, only_stock: bool = False) -> dict:
        raise NotImplementedError(f"InfPosition doesn't support get_stock_weight_dict")

    def add_count_all(self, bar: str) -> None:
        raise NotImplementedError(f"InfPosition doesn't support add_count_all")

    def update_weight_all(self) -> None:
        raise NotImplementedError(f"InfPosition doesn't support update_weight_all")

    def settle_start(self, settle_type: str) -> None:
        pass

    def settle_commit(self) -> None:
        pass