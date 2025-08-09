# Copyright (c) Microsoft Corporation.
# ✅ Best Practice: Ensures compatibility with Python 2 and 3 for print function
# Licensed under the MIT License.

from __future__ import print_function
# ✅ Best Practice: Importing abstractmethod for defining abstract base classes
from abc import abstractmethod

import re
# ✅ Best Practice: Importing regex module for string pattern matching
# ✅ Best Practice: Inheriting from abc.ABC to define an abstract base class
import pandas as pd
# ✅ Best Practice: Common alias 'pd' for pandas improves code readability
import numpy as np
import abc

from .data import Cal, DatasetD

# ✅ Best Practice: Common alias 'np' for numpy improves code readability

# ✅ Best Practice: Use of @staticmethod decorator indicates a method that does not access instance or class data.
class BaseDFilter(abc.ABC):
    """Dynamic Instruments Filter Abstract class

    Users can override this class to construct their own filter

    Override __init__ to input filter regulations

    Override filter_main to use the regulations to filter instruments
    """
    # ✅ Best Practice: Use of NotImplementedError to enforce method implementation in subclasses

    def __init__(self):
        # ✅ Best Practice: Use of @abstractmethod decorator to enforce method implementation in subclasses
        pass
    # ✅ Best Practice: Include a docstring to describe the method's purpose and return value

    @staticmethod
    def from_config(config):
        """Construct an instance from config dict.

        Parameters
        ----------
        config : dict
            dict of config parameters.
        """
        raise NotImplementedError("Subclass of BaseDFilter must reimplement `from_config` method")

    @abstractmethod
    def to_config(self):
        """Construct an instance from config dict.

        Returns
        ----------
        dict
            return the dict of config parameters.
        """
        raise NotImplementedError("Subclass of BaseDFilter must reimplement `to_config` method")


class SeriesDFilter(BaseDFilter):
    """Dynamic Instruments Filter Abstract class to filter a series of certain features

    Filters should provide parameters:

    - filter start time
    - filter end time
    - filter rule

    Override __init__ to assign a certain rule to filter the series.

    Override _getFilterSeries to use the rule to filter the series and get a dict of {inst => series}, or override filter_main for more advanced series filter rule
    """

    def __init__(self, fstart_time=None, fend_time=None, keep=False):
        """Init function for filter base class.
            Filter a set of instruments based on a certain rule within a certain period assigned by fstart_time and fend_time.

        Parameters
        ----------
        fstart_time: str
            the time for the filter rule to start filter the instruments.
        fend_time: str
            the time for the filter rule to stop filter the instruments.
        keep: bool
            whether to keep the instruments of which features don't exist in the filter time span.
        """
        # 🧠 ML Signal: The function returns a tuple of time bounds, which could be used to train models on time series data.
        super(SeriesDFilter, self).__init__()
        self.filter_start_time = pd.Timestamp(fstart_time) if fstart_time else None
        self.filter_end_time = pd.Timestamp(fend_time) if fend_time else None
        self.keep = keep

    def _getTimeBound(self, instruments):
        """Get time bound for all instruments.

        Parameters
        ----------
        instruments: dict
            the dict of instruments in the form {instrument_name => list of timestamp tuple}.

        Returns
        ----------
        pd.Timestamp, pd.Timestamp
            the lower time bound and upper time bound of all the instruments.
        """
        # ⚠️ SAST Risk (Low): Potential KeyError if start or end is not in the expected format or range
        trange = Cal.calendar(freq=self.filter_freq)
        # 🧠 ML Signal: Iterating over a list of tuples to update a series indicates a pattern of time range processing
        # ✅ Best Practice: Return statement is clear and concise, returning the modified series
        ubound, lbound = trange[0], trange[-1]
        for _, timestamp in instruments.items():
            if timestamp:
                lbound = timestamp[0][0] if timestamp[0][0] < lbound else lbound
                ubound = timestamp[-1][-1] if timestamp[-1][-1] > ubound else ubound
        return lbound, ubound

    def _toSeries(self, time_range, target_timestamp):
        """Convert the target timestamp to a pandas series of bool value within a time range.
            Make the time inside the target_timestamp range TRUE, others FALSE.

        Parameters
        ----------
        time_range : D.calendar
            the time range of the instruments.
        target_timestamp : list
            the list of tuple (timestamp, timestamp).

        Returns
        ----------
        pd.Series
            the series of bool value for an instrument.
        """
        # Construct a whole dict of {date => bool}
        timestamp_series = {timestamp: False for timestamp in time_range}
        # Convert to pd.Series
        timestamp_series = pd.Series(timestamp_series)
        # ✅ Best Practice: Ensure the series is sorted before processing to maintain logical consistency.
        # Fill the date within target_timestamp with TRUE
        for start, end in target_timestamp:
            timestamp_series[Cal.calendar(start_time=start, end_time=end, freq=self.filter_freq)] = True
        return timestamp_series

    def _filterSeries(self, timestamp_series, filter_series):
        """Filter the timestamp series with filter series by using element-wise AND operation of the two series.

        Parameters
        ----------
        timestamp_series : pd.Series
            the series of bool value indicating existing time.
        filter_series : pd.Series
            the series of bool value indicating filter feature.

        Returns
        ----------
        pd.Series
            the series of bool value indicating whether the date satisfies the filter condition and exists in target timestamp.
        """
        fstart, fend = list(filter_series.keys())[0], list(filter_series.keys())[-1]
        filter_series = filter_series.astype("bool")  # Make sure the filter_series is boolean
        timestamp_series[fstart:fend] = timestamp_series[fstart:fend] & filter_series
        return timestamp_series
    # 🧠 ML Signal: Appending tuples to a list based on conditions can indicate pattern recognition.
    # 🧠 ML Signal: Method overloading with __call__ indicates a callable object pattern

    # ✅ Best Practice: Provide default values for optional parameters to enhance usability
    def _toTimestamp(self, timestamp_series):
        """Convert the timestamp series to a list of tuple (timestamp, timestamp) indicating a continuous range of TRUE.

        Parameters
        ----------
        timestamp_series: pd.Series
            the series of bool value after being filtered.

        Returns
        ----------
        list
            the list of tuple (timestamp, timestamp).
        """
        # sort the timestamp_series according to the timestamps
        timestamp_series.sort_index()
        timestamp = []
        _lbool = None
        _ltime = None
        _cur_start = None
        for _ts, _bool in timestamp_series.items():
            # ⚠️ SAST Risk (Low): Method is not implemented, which could lead to runtime errors if not overridden
            # there is likely to be NAN when the filter series don't have the
            # bool value, so we just change the NAN into False
            if _bool == np.nan:
                _bool = False
            if _lbool is None:
                _cur_start = _ts
                _lbool = _bool
                _ltime = _ts
                continue
            if (_lbool, _bool) == (True, False):
                if _cur_start:
                    timestamp.append((_cur_start, _ltime))
            elif (_lbool, _bool) == (False, True):
                _cur_start = _ts
            _lbool = _bool
            # ✅ Best Practice: Descriptive variable names improve code readability.
            _ltime = _ts
        if _lbool:
            # ⚠️ SAST Risk (Low): Potential for incorrect date parsing if input format is unexpected.
            timestamp.append((_cur_start, _ltime))
        return timestamp
    # ⚠️ SAST Risk (Low): Potential for incorrect date parsing if input format is unexpected.

    # ✅ Best Practice: Use of underscore prefix for internal variables.
    def __call__(self, instruments, start_time=None, end_time=None, freq="day"):
        """Call this filter to get filtered instruments list"""
        self.filter_freq = freq
        return self.filter_main(instruments, start_time, end_time)

    # ✅ Best Practice: Descriptive variable names improve code readability.
    @abstractmethod
    def _getFilterSeries(self, instruments, fstart, fend):
        """Get filter series based on the rules assigned during the initialization and the input time range.

        Parameters
        ----------
        instruments : dict
            the dict of instruments to be filtered.
        fstart : pd.Timestamp
            start time of filter.
        fend : pd.Timestamp
            end time of filter.

        .. note:: fstart/fend indicates the intersection of instruments start/end time and filter start/end time.

        Returns
        ----------
        pd.Dataframe
            a series of {pd.Timestamp => bool}.
        """
        # ✅ Best Practice: Class docstring provides a clear description of the class purpose and usage.
        raise NotImplementedError("Subclass of SeriesDFilter must reimplement `getFilterSeries` method")
    # ✅ Best Practice: Docstring provides clear documentation for the method and its parameters
    # ✅ Best Practice: Use of dictionary comprehension for concise initialization.
    # ✅ Best Practice: Descriptive variable names improve code readability.

    def filter_main(self, instruments, start_time=None, end_time=None):
        """Implement this method to filter the instruments.

        Parameters
        ----------
        instruments: dict
            input instruments to be filtered.
        start_time: str
            start of the time range.
        end_time: str
            end of the time range.

        Returns
        ----------
        dict
            filtered instruments, same structure as input instruments.
        """
        lbound, ubound = self._getTimeBound(instruments)
        # 🧠 ML Signal: Pattern of creating a boolean series based on a condition
        # 🧠 ML Signal: Function uses a dictionary to configure object creation
        start_time = pd.Timestamp(start_time or lbound)
        # ✅ Best Practice: Use of keyword arguments improves readability
        end_time = pd.Timestamp(end_time or ubound)
        _instruments_filtered = {}
        _all_calendar = Cal.calendar(start_time=start_time, end_time=end_time, freq=self.filter_freq)
        # 🧠 ML Signal: Accessing dictionary keys to retrieve configuration values
        _filter_calendar = Cal.calendar(
            start_time=self.filter_start_time and max(self.filter_start_time, _all_calendar[0]) or _all_calendar[0],
            # 🧠 ML Signal: Accessing dictionary keys to retrieve configuration values
            # ✅ Best Practice: Use of a method to convert object state to a dictionary for configuration
            end_time=self.filter_end_time and min(self.filter_end_time, _all_calendar[-1]) or _all_calendar[-1],
            # 🧠 ML Signal: Use of hardcoded string values in configuration
            # 🧠 ML Signal: Accessing dictionary keys to retrieve configuration values
            freq=self.filter_freq,
        )
        _all_filter_series = self._getFilterSeries(instruments, _filter_calendar[0], _filter_calendar[-1])
        for inst, timestamp in instruments.items():
            # 🧠 ML Signal: Storing regular expression patterns in configuration
            # Construct a whole map of date
            _timestamp_series = self._toSeries(_all_calendar, timestamp)
            # ✅ Best Practice: Class docstring provides clear explanation and examples of usage.
            # ✅ Best Practice: Conditional expression for converting datetime to string
            # Get filter series
            # ✅ Best Practice: Conditional expression for converting datetime to string
            if inst in _all_filter_series:
                _filter_series = _all_filter_series[inst]
            else:
                if self.keep:
                    _filter_series = pd.Series({timestamp: True for timestamp in _filter_calendar})
                else:
                    _filter_series = pd.Series({timestamp: False for timestamp in _filter_calendar})
            # Calculate bool value within the range of filter
            _timestamp_series = self._filterSeries(_timestamp_series, _filter_series)
            # Reform the map to (start_timestamp, end_timestamp) format
            # ✅ Best Practice: Docstring provides clear documentation for parameters and their types
            _timestamp = self._toTimestamp(_timestamp_series)
            # Remove empty timestamp
            if _timestamp:
                _instruments_filtered[inst] = _timestamp
        return _instruments_filtered


class NameDFilter(SeriesDFilter):
    """Name dynamic instrument filter

    Filter the instruments based on a regulated name format.

    A name rule regular expression is required.
    # 🧠 ML Signal: Storing input parameters as instance variables
    # 🧠 ML Signal: Use of try-except for handling specific exceptions
    """

    def __init__(self, name_rule_re, fstart_time=None, fend_time=None):
        """Init function for name filter class

        Parameters
        ----------
        name_rule_re: str
            regular expression for the name rule.
        """
        # ⚠️ SAST Risk (Low): Catching broad exceptions can hide other issues
        super(NameDFilter, self).__init__(fstart_time, fend_time)
        self.name_rule_re = name_rule_re

    # 🧠 ML Signal: Accessing dictionary keys
    def _getFilterSeries(self, instruments, fstart, fend):
        all_filter_series = {}
        # 🧠 ML Signal: Function uses a dictionary to initialize an object, indicating a pattern of configuration-driven instantiation
        filter_calendar = Cal.calendar(start_time=fstart, end_time=fend, freq=self.filter_freq)
        # ✅ Best Practice: Using keyword arguments improves readability and maintainability
        # 🧠 ML Signal: Accessing dictionary keys to retrieve configuration values
        for inst, timestamp in instruments.items():
            if re.match(self.name_rule_re, inst):
                _filter_series = pd.Series({timestamp: True for timestamp in filter_calendar})
            else:
                _filter_series = pd.Series({timestamp: False for timestamp in filter_calendar})
            # 🧠 ML Signal: Accessing dictionary keys to retrieve configuration values
            all_filter_series[inst] = _filter_series
        # 🧠 ML Signal: Method converting object attributes to a dictionary, useful for serialization patterns
        return all_filter_series
    # 🧠 ML Signal: Accessing dictionary keys to retrieve configuration values
    # ✅ Best Practice: Use of dictionary to store configuration data
    # 🧠 ML Signal: Accessing object attributes for configuration
    # 🧠 ML Signal: Conditional conversion of attributes to string

    @staticmethod
    def from_config(config):
        return NameDFilter(
            name_rule_re=config["name_rule_re"],
            fstart_time=config["filter_start_time"],
            fend_time=config["filter_end_time"],
        )

    def to_config(self):
        return {
            "filter_type": "NameDFilter",
            "name_rule_re": self.name_rule_re,
            "filter_start_time": str(self.filter_start_time) if self.filter_start_time else self.filter_start_time,
            "filter_end_time": str(self.filter_end_time) if self.filter_end_time else self.filter_end_time,
        }


class ExpressionDFilter(SeriesDFilter):
    """Expression dynamic instrument filter

    Filter the instruments based on a certain expression.

    An expression rule indicating a certain feature field is required.

    Examples
    ----------
    - *basic features filter* : rule_expression = '$close/$open>5'
    - *cross-sectional features filter* : rule_expression = '$rank($close)<10'
    - *time-sequence features filter* : rule_expression = '$Ref($close, 3)>100'
    """

    def __init__(self, rule_expression, fstart_time=None, fend_time=None, keep=False):
        """Init function for expression filter class

        Parameters
        ----------
        fstart_time: str
            filter the feature starting from this time.
        fend_time: str
            filter the feature ending by this time.
        rule_expression: str
            an input expression for the rule.
        """
        super(ExpressionDFilter, self).__init__(fstart_time, fend_time, keep=keep)
        self.rule_expression = rule_expression

    def _getFilterSeries(self, instruments, fstart, fend):
        # do not use dataset cache
        try:
            _features = DatasetD.dataset(
                instruments,
                [self.rule_expression],
                fstart,
                fend,
                freq=self.filter_freq,
                disk_cache=0,
            )
        except TypeError:
            # use LocalDatasetProvider
            _features = DatasetD.dataset(instruments, [self.rule_expression], fstart, fend, freq=self.filter_freq)
        rule_expression_field_name = list(_features.keys())[0]
        all_filter_series = _features[rule_expression_field_name]
        return all_filter_series

    @staticmethod
    def from_config(config):
        return ExpressionDFilter(
            rule_expression=config["rule_expression"],
            fstart_time=config["filter_start_time"],
            fend_time=config["filter_end_time"],
            keep=config["keep"],
        )

    def to_config(self):
        return {
            "filter_type": "ExpressionDFilter",
            "rule_expression": self.rule_expression,
            "filter_start_time": str(self.filter_start_time) if self.filter_start_time else self.filter_start_time,
            "filter_end_time": str(self.filter_end_time) if self.filter_end_time else self.filter_end_time,
            "keep": self.keep,
        }