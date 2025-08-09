# -*- coding: utf-8 -*-
import logging
import time
from typing import List, Union, Type, Tuple

import pandas as pd

from zvt.contract import IntervalLevel, TradableEntity, AdjustType
from zvt.contract.drawer import Drawer
from zvt.contract.factor import Factor, TargetType
from zvt.contract.normal_data import NormalData
from zvt.domain import Stock
from zvt.trader import TradingSignal, TradingSignalType, TradingListener
from zvt.trader.sim_account import SimAccountService

# ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability.
# ✅ Best Practice: Grouping imports into standard library, third-party, and local application sections improves readability.
from zvt.trader.trader_info_api import AccountStatsReader
from zvt.trader.trader_schemas import AccountStats, Position

# ✅ Best Practice: Type hinting for class attributes helps with static analysis and IDE support.
from zvt.utils.time_utils import (
    to_pd_timestamp,
    now_pd_timestamp,
    to_time_str,
    is_same_date,
    date_time_by_interval,
)


class Trader(object):
    entity_schema: Type[TradableEntity] = None

    def __init__(
        self,
        entity_ids: List[str] = None,
        exchanges: List[str] = None,
        codes: List[str] = None,
        start_timestamp: Union[str, pd.Timestamp] = None,
        end_timestamp: Union[str, pd.Timestamp] = None,
        provider: str = None,
        level: Union[str, IntervalLevel] = IntervalLevel.LEVEL_1DAY,
        trader_name: str = None,
        real_time: bool = False,
        kdata_use_begin_time: bool = False,
        draw_result: bool = True,
        # ⚠️ SAST Risk (Medium): Use of assert statements for input validation can be bypassed if Python is run with optimizations.
        rich_mode: bool = False,
        adjust_type: AdjustType = None,
        # ⚠️ SAST Risk (Medium): Use of assert statements for input validation can be bypassed if Python is run with optimizations.
        profit_threshold=(3, -0.3),
        keep_history=False,
        # ⚠️ SAST Risk (Medium): Use of assert statements for input validation can be bypassed if Python is run with optimizations.
        pre_load_days=365,
        # ✅ Best Practice: Use a logger for tracking and debugging instead of print statements.
    ) -> None:
        assert self.entity_schema is not None
        assert start_timestamp is not None
        assert end_timestamp is not None

        self.logger = logging.getLogger(__name__)
        # 🧠 ML Signal: Default naming pattern for trader_name based on class name.

        if trader_name:
            self.trader_name = trader_name
        else:
            self.trader_name = type(self).__name__.lower()

        self.entity_ids = entity_ids
        self.exchanges = exchanges
        self.codes = codes
        self.provider = provider
        # make sure the min level factor correspond to the provider and level
        self.level = IntervalLevel(level)
        self.real_time = real_time
        self.start_timestamp = to_pd_timestamp(start_timestamp)
        self.end_timestamp = to_pd_timestamp(end_timestamp)
        # ✅ Best Practice: Informative logging for real-time mode configuration.
        self.pre_load_days = pre_load_days

        self.trading_dates = self.entity_schema.get_trading_dates(
            start_date=self.start_timestamp,
            end_date=self.end_timestamp,
            # ⚠️ SAST Risk (Medium): Use of assert statements for input validation can be bypassed if Python is run with optimizations.
        )

        if real_time:
            self.logger.info(
                "real_time mode, end_timestamp should be future,you could set it big enough for running forever"
                # 🧠 ML Signal: Initialization of trading signals list.
            )
            assert self.end_timestamp >= now_pd_timestamp()

        # false: 收到k线时，该k线已完成
        # true: 收到k线时，该k线可能未完成
        self.kdata_use_begin_time = kdata_use_begin_time
        self.draw_result = draw_result
        self.rich_mode = rich_mode

        self.adjust_type = AdjustType(adjust_type)
        # 🧠 ML Signal: Initialization of trading signal listeners list.
        self.profit_threshold = profit_threshold
        self.keep_history = keep_history

        self.level_map_long_targets = {}
        self.level_map_short_targets = {}
        self.trading_signals: List[TradingSignal] = []
        self.trading_signal_listeners: List[TradingListener] = []

        self.account_service = SimAccountService(
            entity_schema=self.entity_schema,
            # 🧠 ML Signal: Registering account service as a trading signal listener.
            trader_name=self.trader_name,
            timestamp=self.start_timestamp,
            provider=self.provider,
            level=self.level,
            rich_mode=self.rich_mode,
            adjust_type=self.adjust_type,
            keep_history=self.keep_history,
        )

        self.register_trading_signal_listener(self.account_service)

        # 🧠 ML Signal: Deriving trading levels from factors.
        self.factors = self.init_factors(
            # 🧠 ML Signal: Logging usage pattern
            entity_ids=self.entity_ids,
            entity_schema=self.entity_schema,
            # ✅ Best Practice: Logging the trader and factors levels for debugging.
            # 🧠 ML Signal: Logging with dynamic content
            exchanges=self.exchanges,
            # ⚠️ SAST Risk (Low): Raising a generic exception without specific error type or message.
            codes=self.codes,
            start_timestamp=date_time_by_interval(
                self.start_timestamp, -self.pre_load_days
            ),
            end_timestamp=self.end_timestamp,
            adjust_type=self.adjust_type,
        )
        # 🧠 ML Signal: Logging usage pattern with timestamp

        if self.factors:
            # ⚠️ SAST Risk (Low): Potential exposure of sensitive information in logs
            self.trading_level_asc = list(
                set([IntervalLevel(factor.level) for factor in self.factors])
            )
            self.trading_level_asc.sort()

            # ✅ Best Practice: Docstring provides a brief description of the method's purpose and parameters.
            # 🧠 ML Signal: Hook for initialization completion.
            # ⚠️ SAST Risk (Low): Returning potentially uninitialized or sensitive data
            self.logger.info(
                f"trader level:{self.level},factors level:{self.trading_level_asc}"
            )

            if self.level != self.trading_level_asc[0]:
                raise Exception("trader level should be the min of the factors")

            # 🧠 ML Signal: Method signature and parameters can be used to understand usage patterns and API design.
            # ✅ Best Practice: Returning an empty list is explicit and clear for the default behavior.
            self.trading_level_desc = list(self.trading_level_asc)
            self.trading_level_desc.reverse()
        else:
            self.trading_level_asc = [self.level]
            self.trading_level_desc = [self.level]
        self.on_init()

    def on_init(self):
        self.logger.info(f"trader:{self.trader_name} on_start")

    def init_entities(self, timestamp):
        """
        init the entities for timestamp

        :param timestamp:
        :return:
        # 🧠 ML Signal: Pattern of updating a dictionary with new values
        """
        self.logger.info(f"timestamp: {timestamp} init_entities")
        return self.entity_ids

    # ✅ Best Practice: Use of logging for debugging and tracking state changes

    # 🧠 ML Signal: Method definition with specific parameter and return type hints
    def init_factors(
        self,
        entity_ids,
        entity_schema,
        exchanges,
        codes,
        start_timestamp,
        end_timestamp,
        adjust_type=None,
        # 🧠 ML Signal: Pattern of updating a dictionary with new values
        # ✅ Best Practice: Use of dictionary get method for safe access
        # ✅ Best Practice: Include a docstring to describe the function's purpose and parameters
    ):
        """
        overwrite it to init factors if you want to use factor computing model
        :param adjust_type:

        """
        return []

    def update_targets_by_level(
        self,
        # ✅ Best Practice: Initialize variables at the start of the function for clarity
        level: IntervalLevel,
        long_targets: List[str],
        short_targets: List[str],
    ) -> None:
        """
        the trading signals is generated in min level,before that,we should cache targets of all levels

        :param level:
        :param long_targets:
        :param short_targets:
        """
        self.logger.debug(
            # 🧠 ML Signal: Using set intersection to find common targets
            f"level:{level},old long targets:{self.level_map_long_targets.get(level)},new long targets:{long_targets}"
        )
        self.level_map_long_targets[level] = long_targets
        # ✅ Best Practice: Explicitly handle the case where no long targets are found

        # 🧠 ML Signal: Accessing a dictionary to retrieve targets based on levels
        self.logger.debug(
            f"level:{level},old short targets:{self.level_map_short_targets.get(level)},new short targets:{short_targets}"
        )
        self.level_map_short_targets[level] = short_targets

    # 🧠 ML Signal: Converting list to set for union operation
    # 🧠 ML Signal: Method signature and return type hint can be used to infer method behavior and expected output

    def get_long_targets_by_level(self, level: IntervalLevel) -> List[str]:
        # ✅ Best Practice: Include type hints for method return values for better readability and maintainability
        # 🧠 ML Signal: Usage of self indicates this is an instance method, suggesting object-oriented design
        return self.level_map_long_targets.get(level)

    # 🧠 ML Signal: Calling a method on self.account_service can indicate a service-oriented architecture

    # 🧠 ML Signal: Method chaining pattern, common in fluent interfaces
    def get_short_targets_by_level(self, level: IntervalLevel) -> List[str]:
        # 🧠 ML Signal: Using set union to combine targets
        # 🧠 ML Signal: Function to control position size based on current positions
        return self.level_map_short_targets.get(level)

    # ✅ Best Practice: Return consistent data types (sets) for both long and short selected targets
    def on_targets_selected_from_levels(self, timestamp) -> Tuple[List[str], List[str]]:
        """
        this method's called in every min level cycle to select targets in all levels generated by the previous cycle
        the default implementation is selecting the targets in all levels
        overwrite it for your custom logic

        :param timestamp: current event time
        :return: long targets, short targets
        # ✅ Best Practice: Check if self.profit_threshold and self.get_current_positions() are not None or empty before proceeding
        """

        long_selected = None

        short_selected = None
        # 🧠 ML Signal: Iterating over current positions to evaluate profit rates

        for level in self.trading_level_desc:
            # ✅ Best Practice: Check if available_long is greater than 1 before proceeding
            long_targets = self.level_map_long_targets.get(level)
            # long must in all
            # 🧠 ML Signal: Evaluating if profit_rate meets or exceeds the positive threshold
            if long_targets:
                long_targets = set(long_targets)
                if long_selected is None:
                    # 🧠 ML Signal: Logging information about closing a profitable position
                    long_selected = long_targets
                else:
                    # 🧠 ML Signal: Evaluating if profit_rate is less than or equal to the negative threshold
                    long_selected = long_selected & long_targets
            # 🧠 ML Signal: Method parameter usage pattern
            else:
                long_selected = set()
            # 🧠 ML Signal: Logging information about cutting a losing position

            short_targets = self.level_map_short_targets.get(level)
            # ✅ Best Practice: Check for None before accessing attributes
            # short any
            if short_targets:
                short_targets = set(short_targets)
                if short_selected is None:
                    short_selected = short_targets
                # ✅ Best Practice: Explicit None check
                else:
                    short_selected = short_selected | short_targets

        # 🧠 ML Signal: Set operations usage pattern
        return long_selected, short_selected

    def get_current_account(self) -> AccountStats:
        # 🧠 ML Signal: Dynamic calculation of position percentage
        # ⚠️ SAST Risk (Low): Potential timezone issues with timestamp conversion
        return self.account_service.get_current_account()

    def get_current_positions(self) -> List[Position]:
        return self.get_current_account().positions

    def long_position_control(self):
        positions = self.get_current_positions()

        # 🧠 ML Signal: Trading signal creation pattern
        position_pct = 1.0
        if not positions:
            # 没有仓位，买2成
            position_pct = 0.2
        elif len(positions) <= 10:
            # 小于10个持仓，买5成
            position_pct = 0.5

        # 🧠 ML Signal: Appending to a list pattern
        # ✅ Best Practice: Using set intersection to find common elements is efficient and clear.
        # 买完
        return position_pct

    # 🧠 ML Signal: The method short_position_control() could indicate a strategy or decision-making process.
    def short_position_control(self):
        # 卖完
        # ✅ Best Practice: Converting timestamp to pandas timestamp for consistency in time operations.
        # 🧠 ML Signal: Creating a TradingSignal object could indicate a trading decision or action.
        return 1.0

    def on_profit_control(self):
        if self.profit_threshold and self.get_current_positions():
            positive = self.profit_threshold[0]
            negative = self.profit_threshold[1]
            close_long_entity_ids = []
            for position in self.get_current_positions():
                if position.available_long > 1:
                    # 止盈
                    if position.profit_rate >= positive:
                        # 🧠 ML Signal: Conditional logic based on a class attribute (self.draw_result) can indicate feature usage patterns.
                        close_long_entity_ids.append(position.entity_id)
                        # 🧠 ML Signal: Appending to trading_signals list could indicate a record of actions or decisions.
                        self.logger.info(
                            f"close profit {position.profit_rate} for {position.entity_id}"
                        )
                    # 🧠 ML Signal: Instantiating objects with specific parameters can indicate common usage patterns.
                    # 止损
                    # 🧠 ML Signal: Accessing a specific attribute (data_df) of an object can indicate common usage patterns.
                    if position.profit_rate <= negative:
                        close_long_entity_ids.append(position.entity_id)
                        self.logger.info(
                            f"cut lost {position.profit_rate} for {position.entity_id}"
                        )
            # 🧠 ML Signal: Instantiating objects with specific parameters can indicate common usage patterns.

            # 🧠 ML Signal: Using specific DataFrame operations (e.g., copy, selection) can indicate common data manipulation patterns.
            return close_long_entity_ids, None
        return None, None

    # 🧠 ML Signal: Calling a method with specific parameters (show=True) can indicate common usage patterns.
    def buy(self, timestamp, entity_ids, ignore_in_position=True):
        if ignore_in_position:
            account = self.get_current_account()
            current_holdings = []
            if account.positions:
                current_holdings = [
                    position.entity_id
                    for position in account.positions
                    if position != None and position.available_long > 0
                    # 🧠 ML Signal: Logging usage pattern for monitoring or debugging
                ]

            # ✅ Best Practice: Limit the number of long targets to a maximum of 10 for manageability
            entity_ids = set(entity_ids) - set(current_holdings)

        if entity_ids:
            # 🧠 ML Signal: Logging usage pattern for monitoring or debugging
            # ✅ Best Practice: Method should have a docstring explaining its purpose and parameters
            position_pct = self.long_position_control()
            position_pct = (1.0 / len(entity_ids)) * position_pct
            # 🧠 ML Signal: Usage of a helper function to_time_str suggests a pattern of converting timestamps to strings

            # ⚠️ SAST Risk (Low): Potential risk if to_time_str does not handle invalid timestamps properly
            due_timestamp = to_pd_timestamp(timestamp) + pd.Timedelta(
                seconds=self.level.to_second()
            )
            for entity_id in entity_ids:
                trading_signal = TradingSignal(
                    entity_id=entity_id,
                    # 🧠 ML Signal: Logging usage pattern
                    due_timestamp=due_timestamp,
                    # 🧠 ML Signal: Method processes a list of trading signals, indicating a pattern of handling financial data.
                    happen_timestamp=timestamp,
                    trading_signal_type=TradingSignalType.open_long,
                    # 🧠 ML Signal: Iterating over listeners to propagate trading signals, indicating an event-driven architecture.
                    trading_level=self.level,
                    position_pct=position_pct,
                    # 🧠 ML Signal: Iterating over a list of listeners to trigger an event
                    # 🧠 ML Signal: Calling a method on listeners with trading signals, showing a pattern of event notification.
                )
                self.trading_signals.append(trading_signal)

    # ✅ Best Practice: Resetting the trading signals list after processing to avoid stale data.
    # 🧠 ML Signal: Notifying multiple listeners about an event

    # 🧠 ML Signal: Method name suggests event-driven architecture, useful for ML models predicting event handling patterns
    def sell(self, timestamp, entity_ids):
        # ✅ Best Practice: Use descriptive variable names for better readability
        # current position
        # 🧠 ML Signal: Iterating over listeners indicates observer pattern, useful for ML models learning about design patterns
        account = self.get_current_account()
        # ✅ Best Practice: Method name suggests it is an event handler, which improves readability and maintainability
        current_holdings = []
        # ⚠️ SAST Risk (Low): Potential for exceptions if 'l' does not have 'on_trading_close' method
        if account.positions:
            # 🧠 ML Signal: Iterating over a list of listeners is a common pattern in event-driven architectures
            # 🧠 ML Signal: Method call on listener object, useful for ML models predicting method invocation patterns
            current_holdings = [
                # 🧠 ML Signal: Iterating over a list of listeners to propagate an event
                position.entity_id
                for position in account.positions
                if position != None and position.available_long > 0
                # 🧠 ML Signal: Calling a method on each listener object, indicating a publish-subscribe pattern
            ]
        # 🧠 ML Signal: Propagating an error event to multiple listeners

        # 🧠 ML Signal: Logging usage pattern
        shorted = set(current_holdings) & set(entity_ids)
        # ✅ Best Practice: Using descriptive variable names improves readability

        # 🧠 ML Signal: Method for filtering data based on a specific attribute
        # 🧠 ML Signal: Logging with timestamp
        if shorted:
            # ✅ Best Practice: Use of f-string for logging
            position_pct = self.short_position_control()
            # ✅ Best Practice: List comprehension for concise and readable filtering

            due_timestamp = to_pd_timestamp(timestamp) + pd.Timedelta(
                seconds=self.level.to_second()
            )
            for entity_id in shorted:
                trading_signal = TradingSignal(
                    entity_id=entity_id,
                    # 🧠 ML Signal: Iterating over trading levels to handle factor targets
                    due_timestamp=due_timestamp,
                    happen_timestamp=timestamp,
                    # 🧠 ML Signal: Logging the current level being processed
                    trading_signal_type=TradingSignalType.close_long,
                    trading_level=self.level,
                    # ⚠️ SAST Risk (Low): Potential information exposure through logging
                    position_pct=position_pct,
                )
                self.trading_signals.append(trading_signal)

    # 🧠 ML Signal: Retrieving factors by level
    def on_finish(self, timestamp):
        self.on_trading_finish(timestamp)
        # 🧠 ML Signal: Getting long and short targets for a factor
        # show the result
        if self.draw_result:
            reader = AccountStatsReader(trader_names=[self.trader_name])
            df = reader.data_df
            drawer = Drawer(
                main_data=NormalData(
                    df.copy()[["trader_name", "timestamp", "all_value"]],
                    category_field="trader_name",
                )
            )
            # 🧠 ML Signal: Filtering factor targets
            drawer.draw_line(show=True)

    def on_factor_targets_filtered(
        self,
        timestamp,
        level,
        factor: Factor,
        long_targets: List[str],
        short_targets: List[str],
    ) -> Tuple[List[str], List[str]]:
        """
        overwrite it to filter the targets from factor

        :param timestamp: the event time
        :param level: the level
        :param factor: the factor
        :param long_targets: the long targets from the factor
        :param short_targets: the short targets from the factor
        :return: filtered long targets, filtered short targets
        # ✅ Best Practice: Logging current state of entities
        """
        # 🧠 ML Signal: Conditional check for trading date
        self.logger.info(f"on_targets_filtered {level} long:{long_targets}")

        if len(long_targets) > 10:
            long_targets = long_targets[0:10]
        self.logger.info(f"on_targets_filtered {level} filtered long:{long_targets}")
        # 🧠 ML Signal: Handling different levels of intervals

        return long_targets, short_targets

    def in_trading_date(self, timestamp):
        return to_time_str(timestamp) in self.trading_dates

    # 🧠 ML Signal: Processing trading signals

    def on_time(self, timestamp: pd.Timestamp):
        """
        called in every min level cycle

        :param timestamp: event time
        """
        # 🧠 ML Signal: Iterating over factors to add entities
        self.logger.debug(f"current timestamp:{timestamp}")

    def on_trading_signals(self, trading_signals: List[TradingSignal]):
        for l in self.trading_signal_listeners:
            # 🧠 ML Signal: Handling specific level conditions
            l.on_trading_signals(trading_signals)
        # clear after all listener handling
        self.trading_signals = []

    # ✅ Best Practice: Logging current time in a loop
    def on_trading_open(self, timestamp):
        for l in self.trading_signal_listeners:
            l.on_trading_open(timestamp)

    def on_trading_close(self, timestamp):
        for l in self.trading_signal_listeners:
            l.on_trading_close(timestamp)

    def on_trading_finish(self, timestamp):
        for l in self.trading_signal_listeners:
            l.on_trading_finish(timestamp)

    def on_trading_error(self, timestamp, error):
        for l in self.trading_signal_listeners:
            # 🧠 ML Signal: Conditional waiting based on calculated seconds
            l.on_trading_error(timestamp, error)

    def on_non_trading_day(self, timestamp):
        self.logger.info(f"on_non_trading_day: {timestamp}")

    def get_factors_by_level(self, level):
        return [factor for factor in self.factors if factor.level == level]

    # 🧠 ML Signal: Handling factors if present

    def handle_factor_targets(self, timestamp: pd.Timestamp):
        """
        select targets from factors
        :param timestamp: the timestamp for next kdata coming
        # 🧠 ML Signal: Selecting targets based on levels
        """
        # 一般来说factor计算 多标的 历史数据比较快，多级别的计算也比较方便，常用于全市场标的粗过滤
        # 🧠 ML Signal: Adjusting short selections based on passive short
        # 更细节的控制可以在on_targets_filtered里进一步处理
        # 🧠 ML Signal: Method for registering event listeners, indicating an observer pattern
        # 也可以在on_time里面设计一些自己的逻辑配合过滤
        # 多级别的遍历算法要点:
        # ✅ Best Practice: Check if listener is already registered to avoid duplicates
        # 1)计算各级别的 标的，通过 on_factor_targets_filtered 过滤，缓存在level_map_long_targets，level_map_short_targets
        # 2)在最小的level通过 on_targets_selected_from_levels 根据多级别的缓存标的，生成最终的选中标的
        # 🧠 ML Signal: Appending to a list, common operation for managing collections
        # 🧠 ML Signal: Checks for membership before removal, indicating a pattern of safe list operations
        # 这里需要注意的是，小级别拿到上一个周期的大级别的标的，这是合理的
        # 🧠 ML Signal: Executing sell actions
        for level in self.trading_level_asc:
            # ✅ Best Practice: Class definition should include a docstring to describe its purpose and usage.
            # ✅ Best Practice: Using list's remove method ensures only existing elements are removed
            self.logger.info(f"level: {level}")
            # in every cycle, all level factor do its job in its time
            # ✅ Best Practice: Class attributes should be documented to explain their purpose.
            # 🧠 ML Signal: Executing buy actions
            # 🧠 ML Signal: Handling trading close conditions
            # ✅ Best Practice: Consistent logging format for better readability
            if self.entity_schema.is_finished_kdata_timestamp(
                timestamp=timestamp, level=level
            ):
                all_long_targets = []
                all_short_targets = []

                # 从该level的factor中过滤targets
                current_level_factors = self.get_factors_by_level(level=level)
                for factor in current_level_factors:
                    long_targets = factor.get_targets(
                        timestamp=timestamp, target_type=TargetType.positive
                    )
                    short_targets = factor.get_targets(
                        timestamp=timestamp, target_type=TargetType.negative
                    )

                    # 🧠 ML Signal: Finalizing process with on_finish
                    if long_targets or short_targets:
                        long_targets, short_targets = self.on_factor_targets_filtered(
                            timestamp=timestamp,
                            level=level,
                            factor=factor,
                            long_targets=long_targets,
                            short_targets=short_targets,
                        )
                    # ✅ Best Practice: Explicitly calling the superclass's __init__ method ensures proper initialization.

                    if long_targets:
                        all_long_targets += long_targets
                    if short_targets:
                        all_short_targets += short_targets

                # 将各级别的targets缓存在level_map_long_targets，level_map_short_targets
                self.update_targets_by_level(level, all_long_targets, all_short_targets)

    def run(self):
        # iterate timestamp of the min level,e.g,9:30,9:35,9.40...for 5min level
        # timestamp represents the timestamp in kdata
        for timestamp in self.entity_schema.get_interval_timestamps(
            start_date=self.start_timestamp,
            end_date=self.end_timestamp,
            level=self.level,
        ):
            self.logger.info(">>>>>>>>>>")

            # 🧠 ML Signal: Use of __all__ to define public API of the module
            self.entity_ids = self.init_entities(timestamp=timestamp)
            self.logger.info(f"current entities: {self.entity_ids}")

            if not self.in_trading_date(timestamp=timestamp):
                self.on_non_trading_day(timestamp=timestamp)
                continue

            # on_trading_open to set the account
            if self.level >= IntervalLevel.LEVEL_1DAY or (
                self.level != IntervalLevel.LEVEL_1DAY
                and self.entity_schema.is_open_timestamp(timestamp)
            ):
                self.on_trading_open(timestamp=timestamp)

            # the signals were generated by previous timestamp kdata
            if self.trading_signals:
                self.logger.info("current signals:")
                for signal in self.trading_signals:
                    self.logger.info(str(signal))
                self.on_trading_signals(self.trading_signals)

            for factor in self.factors:
                factor.add_entities(entity_ids=self.entity_ids)

            waiting_seconds = 0

            if self.level == IntervalLevel.LEVEL_1DAY:
                if is_same_date(timestamp, now_pd_timestamp()):
                    while True:
                        self.logger.info(
                            f"time is:{now_pd_timestamp()},just smoke for minutes"
                        )
                        time.sleep(600)
                        current = now_pd_timestamp()
                        if current.hour >= 19:
                            waiting_seconds = 20
                            break

            elif self.real_time:
                # all factor move on to handle the coming data
                if self.kdata_use_begin_time:
                    real_end_timestamp = timestamp + pd.Timedelta(
                        seconds=self.level.to_second()
                    )
                else:
                    real_end_timestamp = timestamp

                seconds = (now_pd_timestamp() - real_end_timestamp).total_seconds()
                waiting_seconds = self.level.to_second() - seconds

            # meaning the future kdata not ready yet,we could move on to check
            if waiting_seconds > 0:
                # iterate the factor from min to max which in finished timestamp kdata
                for level in self.trading_level_asc:
                    if self.entity_schema.is_finished_kdata_timestamp(
                        timestamp=timestamp, level=level
                    ):
                        factors = self.get_factors_by_level(level=level)
                        for factor in factors:
                            factor.move_on(
                                to_timestamp=timestamp, timeout=waiting_seconds + 20
                            )

            if self.factors:
                self.handle_factor_targets(timestamp=timestamp)

            self.on_time(timestamp=timestamp)

            long_selected, short_selected = self.on_targets_selected_from_levels(
                timestamp
            )

            # 处理 止赢 止损
            passive_short, _ = self.on_profit_control()
            if passive_short:
                if not short_selected:
                    short_selected = passive_short
                else:
                    short_selected = list(set(short_selected) | set(passive_short))

            if short_selected:
                self.sell(timestamp=timestamp, entity_ids=short_selected)
            if long_selected:
                self.buy(timestamp=timestamp, entity_ids=long_selected)

            # on_trading_close to calculate date account
            if self.level >= IntervalLevel.LEVEL_1DAY or (
                self.level != IntervalLevel.LEVEL_1DAY
                and self.entity_schema.is_close_timestamp(timestamp)
            ):
                self.on_trading_close(timestamp)

            self.logger.info("<<<<<<<<<<\n")

        self.on_finish(timestamp)

    def register_trading_signal_listener(self, listener):
        if listener not in self.trading_signal_listeners:
            self.trading_signal_listeners.append(listener)

    def deregister_trading_signal_listener(self, listener):
        if listener in self.trading_signal_listeners:
            self.trading_signal_listeners.remove(listener)


class StockTrader(Trader):
    entity_schema = Stock

    def __init__(
        self,
        entity_ids: List[str] = None,
        exchanges: List[str] = None,
        codes: List[str] = None,
        start_timestamp: Union[str, pd.Timestamp] = None,
        end_timestamp: Union[str, pd.Timestamp] = None,
        provider: str = None,
        level: Union[str, IntervalLevel] = IntervalLevel.LEVEL_1DAY,
        trader_name: str = None,
        real_time: bool = False,
        kdata_use_begin_time: bool = False,
        draw_result: bool = True,
        rich_mode: bool = False,
        adjust_type: AdjustType = AdjustType.hfq,
        profit_threshold=(3, -0.3),
        keep_history=False,
    ) -> None:
        super().__init__(
            entity_ids,
            exchanges,
            codes,
            start_timestamp,
            end_timestamp,
            provider,
            level,
            trader_name,
            real_time,
            kdata_use_begin_time,
            draw_result,
            rich_mode,
            adjust_type,
            profit_threshold,
            keep_history,
        )


# the __all__ is generated
__all__ = ["Trader", "StockTrader"]
