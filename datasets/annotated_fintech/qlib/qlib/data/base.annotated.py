# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


# ✅ Best Practice: Use of relative import for internal module
from __future__ import division
# ✅ Best Practice: Inheriting from abc.ABC indicates that this class is intended to be abstract.
from __future__ import print_function

import abc
import pandas as pd
from ..log import get_module_logger


class Expression(abc.ABC):
    """
    Expression base class

    Expression is designed to handle the calculation of data with the format below
    data with two dimension for each instrument,

    - feature
    - time:  it  could be observation time or period time.

        - period time is designed for Point-in-time database.  For example, the period time maybe 2014Q4, its value can observed for multiple times(different value may be observed at different time due to amendment).
    # ✅ Best Practice: Local import to avoid circular dependencies
    """
    # ✅ Best Practice: Use of dunder method __lt__ for implementing less-than comparison

    # 🧠 ML Signal: Custom operator overloading can indicate domain-specific logic
    def __str__(self):
        # ✅ Best Practice: Local import to avoid circular dependencies
        return type(self).__name__
    # ✅ Best Practice: Use of dunder method for operator overloading

    # 🧠 ML Signal: Use of custom operation class for comparison
    def __repr__(self):
        # ✅ Best Practice: Local import to avoid circular dependencies
        return str(self)
    # ✅ Best Practice: Use of dunder method for equality comparison

    # 🧠 ML Signal: Custom operator overloading can indicate domain-specific logic
    def __gt__(self, other):
        # ✅ Best Practice: Local import to avoid circular dependencies
        from .ops import Gt  # pylint: disable=C0415
        # ✅ Best Practice: Use of dunder method for operator overloading

        # 🧠 ML Signal: Custom equality logic using an imported operation
        return Gt(self, other)
    # ✅ Best Practice: Local import to avoid circular dependencies

    def __ge__(self, other):
        # 🧠 ML Signal: Custom implementation of inequality operator
        # ✅ Best Practice: Importing inside a function can reduce initial load time and limit scope.
        from .ops import Ge  # pylint: disable=C0415

        # ✅ Best Practice: Use of dunder method for operator overloading
        # 🧠 ML Signal: Overloading the addition operator to customize object behavior.
        return Ge(self, other)

    # ✅ Best Practice: Local import to avoid circular dependencies
    def __lt__(self, other):
        # ✅ Best Practice: Use of dunder method for operator overloading
        from .ops import Lt  # pylint: disable=C0415
        # 🧠 ML Signal: Use of custom addition operation

        # ✅ Best Practice: Local import to avoid circular dependencies
        return Lt(self, other)
    # ✅ Best Practice: Use of dunder method for operator overloading

    # 🧠 ML Signal: Custom operator overloading
    def __le__(self, other):
        # ✅ Best Practice: Local import to avoid circular dependencies
        from .ops import Le  # pylint: disable=C0415

        # 🧠 ML Signal: Use of custom operation class for subtraction
        # ✅ Best Practice: Importing inside a function can reduce initial load time and avoid circular imports.
        return Le(self, other)

    # 🧠 ML Signal: Overloading the multiplication operator to define custom behavior.
    # ✅ Best Practice: Use of dunder method for operator overloading
    def __eq__(self, other):
        from .ops import Eq  # pylint: disable=C0415
        # ✅ Best Practice: Local import to avoid circular dependencies

        # ✅ Best Practice: Use of double underscore indicates a special method, which is a good practice for operator overloading.
        return Eq(self, other)
    # 🧠 ML Signal: Use of custom multiplication operation

    # ✅ Best Practice: Local import can be beneficial for reducing initial load time and avoiding circular imports.
    def __ne__(self, other):
        # ✅ Best Practice: Use of dunder method for operator overloading
        from .ops import Ne  # pylint: disable=C0415
        # 🧠 ML Signal: Usage of custom operator overloading can indicate specific domain logic or patterns.

        # ✅ Best Practice: Local import to avoid circular dependencies
        return Ne(self, other)
    # ✅ Best Practice: Define special method for division operation

    # 🧠 ML Signal: Use of custom division operation
    def __add__(self, other):
        # ✅ Best Practice: Importing within function scope to limit import to where it's needed
        from .ops import Add  # pylint: disable=C0415
        # ✅ Best Practice: Use of dunder method for operator overloading

        # 🧠 ML Signal: Usage of custom division operation
        return Add(self, other)
    # ✅ Best Practice: Local import to avoid circular dependencies

    # ✅ Best Practice: Use of dunder method for operator overloading
    def __radd__(self, other):
        # 🧠 ML Signal: Use of custom division operation
        from .ops import Add  # pylint: disable=C0415
        # ✅ Best Practice: Local import to avoid circular dependencies

        return Add(other, self)
    # 🧠 ML Signal: Custom operator overloading for power operation
    # ✅ Best Practice: Importing inside a function can reduce initial load time and avoid circular imports.

    def __sub__(self, other):
        # ✅ Best Practice: Use of dunder method for operator overloading
        # 🧠 ML Signal: Usage of reverse operator overloading can indicate advanced Python usage patterns.
        from .ops import Sub  # pylint: disable=C0415

        # ✅ Best Practice: Local import to avoid circular dependencies
        return Sub(self, other)
    # ✅ Best Practice: Use of double underscore method indicates operator overloading

    # 🧠 ML Signal: Custom operator overloading can indicate complex object behavior
    def __rsub__(self, other):
        # ✅ Best Practice: Local import can reduce initial load time and avoid circular imports
        from .ops import Sub  # pylint: disable=C0415
        # ✅ Best Practice: Use of dunder method for operator overloading

        # 🧠 ML Signal: Usage of custom operator overloading
        return Sub(other, self)
    # ✅ Best Practice: Local import to avoid circular dependencies

    # ✅ Best Practice: Use of dunder method for operator overloading
    def __mul__(self, other):
        # 🧠 ML Signal: Custom operator overloading pattern
        from .ops import Mul  # pylint: disable=C0415
        # ✅ Best Practice: Local import to avoid circular dependencies

        return Mul(self, other)
    # 🧠 ML Signal: Use of custom operator overloading

    def __rmul__(self, other):
        from .ops import Mul  # pylint: disable=C0415

        return Mul(self, other)

    def __div__(self, other):
        from .ops import Div  # pylint: disable=C0415

        return Div(self, other)

    def __rdiv__(self, other):
        from .ops import Div  # pylint: disable=C0415

        return Div(other, self)

    def __truediv__(self, other):
        from .ops import Div  # pylint: disable=C0415

        return Div(self, other)

    def __rtruediv__(self, other):
        from .ops import Div  # pylint: disable=C0415

        return Div(other, self)

    def __pow__(self, other):
        from .ops import Power  # pylint: disable=C0415

        return Power(self, other)

    def __rpow__(self, other):
        from .ops import Power  # pylint: disable=C0415
        # ✅ Best Practice: Use of a tuple as a cache key ensures immutability and hashability.

        return Power(other, self)
    # 🧠 ML Signal: Caching mechanism usage pattern.

    def __and__(self, other):
        from .ops import And  # pylint: disable=C0415
        # ⚠️ SAST Risk (Low): Potential logic error if start_index and end_index are not validated properly.

        return And(self, other)

    # 🧠 ML Signal: Pattern of calling an internal method for data loading.
    def __rand__(self, other):
        from .ops import And  # pylint: disable=C0415

        return And(other, self)

    # ⚠️ SAST Risk (Low): Logging of exception details could expose sensitive information.
    def __or__(self, other):
        from .ops import Or  # pylint: disable=C0415

        return Or(self, other)

    def __ror__(self, other):
        # ✅ Best Practice: Naming the series for better identification and debugging.
        # ✅ Best Practice: Raising NotImplementedError in abstract methods is a good practice to enforce implementation in subclasses.
        from .ops import Or  # pylint: disable=C0415

        # 🧠 ML Signal: Caching the result for future use.
        return Or(other, self)
    # ✅ Best Practice: Using @abc.abstractmethod decorator indicates that this method is intended to be overridden in a subclass.

    def load(self, instrument, start_index, end_index, *args):
        """load  feature
        This function is responsible for loading feature/expression based on the expression engine.

        The concrete implementation will be separated into two parts:

        1) caching data, handle errors.

            - This part is shared by all the expressions and implemented in Expression
        2) processing and calculating data based on the specific expression.

            - This part is different in each expression and implemented in each expression

        Expression Engine is shared by different data.
        Different data will have different extra information for `args`.

        Parameters
        ----------
        instrument : str
            instrument code.
        start_index : str
            feature start index [in calendar].
        end_index : str
            feature end  index  [in calendar].

        *args may contain following information:
        1) if it is used in basic expression engine data, it contains following arguments
            freq: str
                feature frequency.

        2) if is used in PIT data, it contains following arguments
            cur_pit:
                it is designed for the point-in-time data.
            period: int
                This is used for query specific period.
                The period is represented with int in Qlib. (e.g. 202001 may represent the first quarter in 2020)

        Returns
        ----------
        pd.Series
            feature series: The index of the series is the calendar index
        # ✅ Best Practice: Import statements are typically placed at the top of the file
        """
        from .cache import H  # pylint: disable=C0415
        # ✅ Best Practice: Class docstring provides a clear description of the class purpose and usage.
        # 🧠 ML Signal: Usage of a method from an imported module
        # 🧠 ML Signal: Method call with multiple parameters

        # cache
        cache_key = str(self), instrument, start_index, end_index, *args
        if cache_key in H["f"]:
            return H["f"][cache_key]
        if start_index is not None and end_index is not None and start_index > end_index:
            raise ValueError("Invalid index range: {} {}".format(start_index, end_index))
        try:
            series = self._load_internal(instrument, start_index, end_index, *args)
        except Exception as e:
            get_module_logger("data").debug(
                f"Loading data error: instrument={instrument}, expression={str(self)}, "
                f"start_index={start_index}, end_index={end_index}, args={args}. "
                f"error info: {str(e)}"
            )
            raise
        series.name = str(self)
        H["f"][cache_key] = series
        return series

    @abc.abstractmethod
    def _load_internal(self, instrument, start_index, end_index, *args) -> pd.Series:
        raise NotImplementedError("This function must be implemented in your newly defined feature")

    @abc.abstractmethod
    def get_longest_back_rolling(self):
        """Get the longest length of historical data the feature has accessed

        This is designed for getting the needed range of the data to calculate
        the features in specific range at first.  However, situations like
        Ref(Ref($close, -1), 1) can not be handled rightly.

        So this will only used for detecting the length of historical data needed.
        """
        # TODO: forward operator like Ref($close, -1) is not supported yet.
        raise NotImplementedError("This function must be implemented in your newly defined feature")

    @abc.abstractmethod
    def get_extended_window_size(self):
        """get_extend_window_size

        For to calculate this Operator in range[start_index, end_index]
        We have to get the *leaf feature* in
        range[start_index - lft_etd, end_index + rght_etd].

        Returns
        ----------
        (int, int)
            lft_etd, rght_etd
        """
        raise NotImplementedError("This function must be implemented in your newly defined feature")


class Feature(Expression):
    """Static Expression

    This kind of feature will load data from provider
    """

    def __init__(self, name=None):
        if name:
            self._name = name
        else:
            self._name = type(self).__name__

    def __str__(self):
        return "$" + self._name

    def _load_internal(self, instrument, start_index, end_index, freq):
        # load
        from .data import FeatureD  # pylint: disable=C0415

        return FeatureD.feature(instrument, str(self), start_index, end_index, freq)

    def get_longest_back_rolling(self):
        return 0

    def get_extended_window_size(self):
        return 0, 0


class PFeature(Feature):
    def __str__(self):
        return "$$" + self._name

    def _load_internal(self, instrument, start_index, end_index, cur_time, period=None):
        from .data import PITD  # pylint: disable=C0415

        return PITD.period_feature(instrument, str(self), start_index, end_index, cur_time, period)


class ExpressionOps(Expression):
    """Operator Expression

    This kind of feature will use operator for feature
    construction on the fly.
    """