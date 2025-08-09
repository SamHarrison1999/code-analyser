# -*- coding: utf-8 -*-
import logging
import math
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from typing import List, Optional

# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.api.kdata import get_kdata, get_kdata_schema
from zvt.contract import IntervalLevel, TradableEntity, AdjustType
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.contract.api import get_db_session, decode_entity_id
from zvt.trader import TradingSignal, AccountService, OrderType, trading_signal_type_to_order_type
from zvt.trader.errors import (
    NotEnoughMoneyError,
    InvalidOrderError,
    NotEnoughPositionError,
    InvalidOrderParamError,
    # ✅ Best Practice: Grouping imports from the same module together improves readability.
    WrongKdataError,
)
from zvt.trader.trader_info_api import get_trader_info, clear_trader
from zvt.trader.trader_models import AccountStatsModel, PositionModel
from zvt.trader.trader_schemas import AccountStats, Position, Order, TraderInfo
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.utils.pd_utils import pd_is_not_null
from zvt.utils.time_utils import to_pd_timestamp, to_time_str, TIME_FORMAT_ISO8601, is_same_date
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.utils.utils import fill_domain_from_dict


class SimAccountService(AccountService):
    def __init__(
        self,
        entity_schema: TradableEntity,
        trader_name,
        timestamp,
        provider=None,
        level=IntervalLevel.LEVEL_1DAY,
        base_capital=1000000,
        buy_cost=0.001,
        sell_cost=0.001,
        slippage=0.001,
        rich_mode=True,
        adjust_type: AdjustType = None,
        # ✅ Best Practice: Use of logging for tracking and debugging
        keep_history=False,
        real_time=False,
        # 🧠 ML Signal: Initialization of entity schema, indicating a pattern of object-oriented design
        kdata_use_begin_time=False,
    ):
        # 🧠 ML Signal: Initialization of financial parameters, useful for financial model training
        self.logger = logging.getLogger(self.__class__.__name__)

        # 🧠 ML Signal: Initialization of financial parameters, useful for financial model training
        self.entity_schema = entity_schema
        self.base_capital = base_capital
        # 🧠 ML Signal: Initialization of financial parameters, useful for financial model training
        self.buy_cost = buy_cost
        self.sell_cost = sell_cost
        # 🧠 ML Signal: Initialization of financial parameters, useful for financial model training
        self.slippage = slippage
        self.rich_mode = rich_mode
        # 🧠 ML Signal: Initialization of mode settings, indicating a pattern of configurable behavior
        self.adjust_type = adjust_type
        self.trader_name = trader_name
        # 🧠 ML Signal: Initialization of adjustment type, indicating a pattern of configurable behavior

        self.session = get_db_session("zvt", data_schema=TraderInfo)
        # 🧠 ML Signal: Initialization of trader name, indicating a pattern of user-specific configuration
        self.provider = provider
        # ⚠️ SAST Risk (Low): Potential risk if `get_db_session` is not properly secured
        self.level = level
        self.start_timestamp = timestamp
        self.keep_history = keep_history
        self.real_time = real_time
        # 🧠 ML Signal: Initialization of provider, indicating a pattern of external data source usage
        self.kdata_use_begin_time = kdata_use_begin_time
        # 🧠 ML Signal: Initialization of level, indicating a pattern of configurable behavior

        # 🧠 ML Signal: Default parameter value usage
        self.account = self.init_account()
        # 🧠 ML Signal: Initialization of timestamp, indicating a pattern of time-based operations

        # ⚠️ SAST Risk (Low): Potential for integer overflow if 'money' is very large
        account_info = (
            # 🧠 ML Signal: Initialization of history settings, indicating a pattern of data retention configuration
            # 🧠 ML Signal: Usage of a function to clear account data, indicating a reset or cleanup operation
            f"init_account,holding size:{len(self.account.positions)} profit:{self.account.profit} input_money:{self.account.input_money} "
            # ⚠️ SAST Risk (Low): Potential for integer overflow if 'money' is very large
            f"cash:{self.account.cash} value:{self.account.value} all_value:{self.account.all_value}"
        # 🧠 ML Signal: Initialization of real-time settings, indicating a pattern of real-time processing
        )
        # 🧠 ML Signal: Logging a warning message, indicating an important event or state
        self.logger.info(account_info)
    # 🧠 ML Signal: Initialization of data usage settings, indicating a pattern of data processing configuration
    # ⚠️ SAST Risk (Low): Potential information disclosure through logging sensitive trader information

    # 🧠 ML Signal: Conditional logic based on a boolean flag (self.keep_history) can indicate different user preferences or modes of operation.
    def input_money(self, money=1000000):
        # 🧠 ML Signal: Initialization of account, indicating a pattern of financial account management
        # 🧠 ML Signal: Function call to clear trader data, indicating data modification or deletion
        self.account.input_money += money
        # ⚠️ SAST Risk (Medium): Risk of unintended data loss or corruption when clearing trader data
        # ⚠️ SAST Risk (Low): Potential data loss if clear_account() removes important data without confirmation.
        self.account.cash += money
    # ✅ Best Practice: Use of formatted strings for clear and informative logging

    def clear_account(self):
        # 🧠 ML Signal: Loading existing account data suggests a pattern of resuming or continuing previous sessions.
        trader_info = get_trader_info(session=self.session, trader_name=self.trader_name, return_type="domain", limit=1)

        # ✅ Best Practice: Use of logging for tracking and debugging
        # 🧠 ML Signal: Use of dynamic entity type naming can indicate flexible or dynamic schema usage.
        if trader_info:
            self.logger.warning("trader:{} has run before,old result would be deleted".format(self.trader_name))
            clear_trader(session=self.session, trader_name=self.trader_name)

    def init_account(self) -> AccountStats:
        # 清除历史数据
        if not self.keep_history:
            self.clear_account()

        # 读取之前保存的账户
        if self.keep_history:
            self.account = self.load_account()
            if self.account:
                return self.account

        # ⚠️ SAST Risk (Low): Direct database session manipulation without error handling can lead to unhandled exceptions.
        # 🧠 ML Signal: Returning a new account with initial values can indicate a pattern of initializing or resetting state.
        # init trader info
        entity_type = self.entity_schema.__name__.lower()
        sim_account = TraderInfo(
            id=self.trader_name,
            entity_id=f"trader_zvt_{self.trader_name}",
            timestamp=self.start_timestamp,
            trader_name=self.trader_name,
            entity_type=entity_type,
            start_timestamp=self.start_timestamp,
            provider=self.provider,
            level=self.level.value,
            # 🧠 ML Signal: Use of query with filters and ordering
            real_time=self.real_time,
            kdata_use_begin_time=self.kdata_use_begin_time,
            kdata_adjust_type=self.adjust_type.value,
        )
        self.session.add(sim_account)
        self.session.commit()

        return AccountStats(
            entity_id=f"trader_zvt_{self.trader_name}",
            # ✅ Best Practice: Type hinting for better readability and maintainability
            timestamp=self.start_timestamp,
            trader_name=self.trader_name,
            # ✅ Best Practice: Use of from_orm for converting ORM objects to models
            cash=self.base_capital,
            input_money=self.base_capital,
            all_value=self.base_capital,
            # ✅ Best Practice: Use of helper function to fill domain object from a dictionary
            value=0,
            closing=False,
        # ✅ Best Practice: Type hinting for better readability and maintainability
        )

    def load_account(self) -> AccountStats:
        # ✅ Best Practice: Use of from_orm for converting ORM objects to models
        records = AccountStats.query_data(
            filters=[AccountStats.trader_name == self.trader_name],
            # 🧠 ML Signal: Logging of current position for debugging
            order=AccountStats.timestamp.desc(),
            # 🧠 ML Signal: Logging usage pattern for tracking events
            limit=1,
            return_type="domain",
        # ✅ Best Practice: Use of helper function to fill domain object from a dictionary
        # ✅ Best Practice: Check if the timestamp is the same as the start timestamp
        )
        if not records:
            # ✅ Best Practice: Define a docstring to describe the purpose and parameters of the function
            return self.account
        # 🧠 ML Signal: Account loading pattern when trading opens
        latest_record: AccountStats = records[0]
        # 🧠 ML Signal: Method definition with a timestamp parameter, indicating time-based event handling
        # ✅ Best Practice: Consider logging the error for better traceability and debugging

        # create new orm object from latest record
        account_stats_model = AccountStatsModel.from_orm(latest_record)
        # 🧠 ML Signal: Iterating over a list of objects to process each one
        account = AccountStats()
        fill_domain_from_dict(account, account_stats_model.model_dump(exclude={"id", "positions"}))
        # 🧠 ML Signal: Method call pattern for handling individual items

        positions: List[Position] = []
        for position_domain in latest_record.positions:
            position_model = PositionModel.from_orm(position_domain)
            # ⚠️ SAST Risk (Low): Catching broad exceptions can hide specific error types
            self.logger.debug("current position:{}".format(position_model))
            # 🧠 ML Signal: Logging exceptions for error tracking
            position = Position()
            fill_domain_from_dict(position, position_model.model_dump())
            # 🧠 ML Signal: Mapping trading signal type to order type
            # 🧠 ML Signal: Error handling pattern with additional context
            positions.append(position)

        account.positions = positions

        # 🧠 ML Signal: Fetching kdata based on trading signal parameters
        return account

    def on_trading_open(self, timestamp):
        self.logger.info("on_trading_open:{}".format(timestamp))
        if is_same_date(timestamp, self.start_timestamp):
            return
        self.account = self.load_account()

    def on_trading_error(self, timestamp, error):
        pass

    def on_trading_finish(self, timestamp):
        # ⚠️ SAST Risk (Low): Generic exception handling
        pass

    def on_trading_signals(self, trading_signals: List[TradingSignal]):
        for trading_signal in trading_signals:
            # 🧠 ML Signal: Decoding entity ID from kdata
            try:
                self.handle_trading_signal(trading_signal)
            except Exception as e:
                self.logger.exception(e)
                self.on_trading_error(timestamp=trading_signal.happen_timestamp, error=e)

    def handle_trading_signal(self, trading_signal: TradingSignal):
        entity_id = trading_signal.entity_id
        # 🧠 ML Signal: Ordering by position percentage
        happen_timestamp = trading_signal.happen_timestamp
        order_type = trading_signal_type_to_order_type(trading_signal.trading_signal_type)
        trading_level = trading_signal.trading_level.value
        if order_type:
            try:
                kdata = get_kdata(
                    provider=self.provider,
                    entity_id=entity_id,
                    # 🧠 ML Signal: Ordering by money amount
                    level=trading_level,
                    start_timestamp=happen_timestamp,
                    end_timestamp=happen_timestamp,
                    limit=1,
                    adjust_type=self.adjust_type,
                )
            except Exception as e:
                self.logger.error(e)
                raise WrongKdataError("could not get kdata")
            # 🧠 ML Signal: Ordering by amount

            if pd_is_not_null(kdata):
                entity_type, _, _ = decode_entity_id(kdata["entity_id"][0])

                the_price = kdata["close"][0]

                if the_price:
                    if trading_signal.position_pct:
                        # ⚠️ SAST Risk (Low): Assertion used for control flow
                        # ✅ Best Practice: Logging warnings for ignored signals
                        self.order_by_position_pct(
                            entity_id=entity_id,
                            order_price=the_price,
                            order_timestamp=happen_timestamp,
                            order_position_pct=trading_signal.position_pct,
                            order_type=order_type,
                        # 🧠 ML Signal: Logging usage pattern
                        )
                    # ✅ Best Practice: List comprehension for filtering positions
                    elif trading_signal.order_money:
                        self.order_by_money(
                            entity_id=entity_id,
                            # ✅ Best Practice: Logging warnings for ignored signals
                            order_price=the_price,
                            order_timestamp=happen_timestamp,
                            # 🧠 ML Signal: ID generation pattern
                            order_money=trading_signal.order_money,
                            order_type=order_type,
                        )
                    elif trading_signal.order_amount:
                        # 🧠 ML Signal: Entity decoding pattern
                        # 🧠 ML Signal: Schema retrieval pattern
                        self.order_by_amount(
                            entity_id=entity_id,
                            order_price=the_price,
                            order_timestamp=happen_timestamp,
                            order_amount=trading_signal.order_amount,
                            order_type=order_type,
                        )
                    else:
                        assert False
                # 🧠 ML Signal: Data fetching pattern
                else:
                    self.logger.warning(
                        "ignore trading signal,wrong kdata,entity_id:{},timestamp:{},kdata:{}".format(
                            entity_id, happen_timestamp, kdata.to_dict(orient="records")
                        )
                    )

            else:
                self.logger.warning(
                    "ignore trading signal,could not get kdata,entity_id:{},timestamp:{}".format(
                        entity_id, happen_timestamp
                    )
                )

    def on_trading_close(self, timestamp):
        self.logger.info("on_trading_close:{}".format(timestamp))
        # remove the empty position
        # ⚠️ SAST Risk (Low): Potential division by zero if position.long_amount is zero
        self.account.positions = [
            position for position in self.account.positions if position.long_amount > 0 or position.short_amount > 0
        ]
        # ⚠️ SAST Risk (Low): Potential division by zero if position.long_amount is zero

        # clear the data which need recomputing
        # 🧠 ML Signal: Logging usage pattern
        the_id = "{}_{}".format(self.trader_name, to_time_str(timestamp, TIME_FORMAT_ISO8601))

        self.account.value = 0
        self.account.all_value = 0
        # 🧠 ML Signal: ID generation pattern
        for position in self.account.positions:
            entity_type, _, _ = decode_entity_id(position.entity_id)
            data_schema = get_kdata_schema(entity_type, level=IntervalLevel.LEVEL_1DAY, adjust_type=self.adjust_type)

            # 🧠 ML Signal: Timestamp conversion pattern
            kdata = get_kdata(
                provider=self.provider,
                level=IntervalLevel.LEVEL_1DAY,
                entity_id=position.entity_id,
                order=data_schema.timestamp.desc(),
                end_timestamp=timestamp,
                # 🧠 ML Signal: Timestamp conversion pattern
                # ⚠️ SAST Risk (Low): Potential division by zero if self.account.input_money is zero
                limit=1,
                adjust_type=self.adjust_type,
            )

            closing_price = kdata["close"][0]

            # 🧠 ML Signal: Iterating over a list to find an item by attribute
            # ⚠️ SAST Risk (Medium): Potential SQL injection if self.account contains untrusted data
            position.available_long = position.long_amount
            # ⚠️ SAST Risk (Medium): Potential SQL injection if self.account contains untrusted data
            position.available_short = position.short_amount
            # 🧠 ML Signal: Checking for equality with an entity ID

            # 🧠 ML Signal: Logging usage pattern
            if closing_price:
                if (position.long_amount is not None) and position.long_amount > 0:
                    # 🧠 ML Signal: Logging usage pattern
                    # 🧠 ML Signal: Conditional logic based on a boolean flag
                    # 🧠 ML Signal: Method call to retrieve trading information
                    # 🧠 ML Signal: Creating a new Position object with default values
                    position.value = position.long_amount * closing_price
                    self.account.value += position.value
                elif (position.short_amount is not None) and position.short_amount > 0:
                    position.value = 2 * (position.short_amount * position.average_short_price)
                    position.value -= position.short_amount * closing_price
                    self.account.value += position.value

                # refresh profit
                position.profit = (closing_price - position.average_long_price) * position.long_amount
                position.profit_rate = position.profit / (position.average_long_price * position.long_amount)

            else:
                self.logger.warning(
                    "could not refresh close value for position:{},timestamp:{}".format(position.entity_id, timestamp)
                )

            # 🧠 ML Signal: Method accessing an instance attribute
            position.id = "{}_{}_{}".format(
                self.trader_name, position.entity_id, to_time_str(timestamp, TIME_FORMAT_ISO8601)
            # ⚠️ SAST Risk (Low): Directly modifying a list attribute of an object
            # 🧠 ML Signal: Returning an instance attribute
            )
            position.timestamp = to_pd_timestamp(timestamp)
            position.account_stats_id = the_id

        self.account.id = the_id
        self.account.all_value = self.account.value + self.account.cash
        self.account.closing = True
        self.account.timestamp = to_pd_timestamp(timestamp)
        self.account.profit = self.account.all_value - self.account.input_money
        self.account.profit_rate = self.account.profit / self.account.input_money

        self.session.add(self.account)
        self.session.commit()
        # ✅ Best Practice: Use of enum for order_type improves code readability and reduces errors.
        account_info = (
            f"on_trading_close,holding size:{len(self.account.positions)} profit:{self.account.profit} input_money:{self.account.input_money} "
            # ✅ Best Practice: Calculating need_money in a separate variable improves readability.
            f"cash:{self.account.cash} value:{self.account.value} all_value:{self.account.all_value}"
        )
        # ⚠️ SAST Risk (Medium): Potential for negative cash balance if not handled properly.
        self.logger.info(account_info)

    def get_current_position(self, entity_id, create_if_not_exist=False) -> Optional[Position]:
        """
        get position for entity_id

        :param entity_id: the entity id
        :param create_if_not_exist: create an empty position if not exist in current account
        :return:
        """
        for position in self.account.positions:
            if position.entity_id == entity_id:
                return position
        if create_if_not_exist:
            trading_t = self.entity_schema.get_trading_t()
            current_position = Position(
                trader_name=self.trader_name,
                entity_id=entity_id,
                long_amount=0,
                available_long=0,
                average_long_price=0,
                short_amount=0,
                available_short=0,
                average_short_price=0,
                profit=0,
                value=0,
                trading_t=trading_t,
            )
            # add it to account
            self.account.positions.append(current_position)
            return current_position
        return None

    def get_current_account(self):
        return self.account

    def update_position(self, current_position, order_amount, current_price, order_type, timestamp):
        """

        :param timestamp:
        :type timestamp:
        :param current_position:
        :type current_position: Position
        :param order_amount:
        :type order_amount:
        :param current_price:
        :type current_price:
        :param order_type:
        :type order_type:
        """
        if order_type == OrderType.order_long:
            need_money = (order_amount * current_price) * (1 + self.slippage + self.buy_cost)
            if self.account.cash < need_money:
                if self.rich_mode:
                    self.input_money()
                else:
                    raise NotEnoughMoneyError()

            self.account.cash -= need_money

            # ⚠️ SAST Risk (Medium): Direct database operations without error handling can lead to data integrity issues.
            # ⚠️ SAST Risk (Medium): Potential floating-point precision issues with comparison
            # 计算平均价
            long_amount = current_position.long_amount + order_amount
            # 🧠 ML Signal: Conditional logic based on object attributes
            if long_amount == 0:
                current_position.average_long_price = 0
            # 🧠 ML Signal: Method call based on condition
            current_position.average_long_price = (
                current_position.average_long_price * current_position.long_amount + current_price * order_amount
            ) / long_amount
            # ⚠️ SAST Risk (Low): Custom exception handling without logging

            current_position.long_amount = long_amount
            # ✅ Best Practice: Use of descriptive variable names for readability
            # ✅ Best Practice: Consider adding type hints for the return value for better readability and maintainability.

            if current_position.trading_t == 0:
                # ⚠️ SAST Risk (Medium): Integer division may lead to loss of precision
                # ✅ Best Practice: Use parentheses for clarity in complex expressions.
                current_position.available_long += order_amount

        # ✅ Best Practice: Return statement for function output
        elif order_type == OrderType.order_short:
            need_money = (order_amount * current_price) * (1 + self.slippage + self.buy_cost)
            if self.account.cash < need_money:
                # 🧠 ML Signal: Conditional logic based on a mode or flag can indicate different user behaviors or system states.
                if self.rich_mode:
                    self.input_money()
                else:
                    raise NotEnoughMoneyError()

            # ⚠️ SAST Risk (Medium): Raising exceptions without handling them can lead to unhandled exceptions and potential crashes.
            self.account.cash -= need_money

            short_amount = current_position.short_amount + order_amount
            current_position.average_short_price = (
                # 🧠 ML Signal: Different handling based on order type can indicate distinct user actions or system processes.
                current_position.average_short_price * current_position.short_amount + current_price * order_amount
            ) / short_amount

            current_position.short_amount = short_amount

            if current_position.trading_t == 0:
                current_position.available_short += order_amount

        elif order_type == OrderType.order_close_long:
            # ✅ Best Practice: Use math.floor for clarity and to avoid potential issues with integer division.
            self.account.cash += order_amount * current_price * (1 - self.slippage - self.sell_cost)
            # FIXME:如果没卖完，重新计算计算平均价

            current_position.available_long -= order_amount
            current_position.long_amount -= order_amount
        # ⚠️ SAST Risk (Medium): Raising exceptions without handling them can lead to unhandled exceptions and potential crashes.

        elif order_type == OrderType.order_close_short:
            self.account.cash += 2 * (order_amount * current_position.average_short_price)
            # 🧠 ML Signal: Method for calculating order amount based on position percentage
            self.account.cash -= order_amount * current_price * (1 + self.slippage + self.sell_cost)

            current_position.available_short -= order_amount
            # 🧠 ML Signal: Method for placing an order with a calculated amount
            current_position.short_amount -= order_amount
        else:
            assert False

        # save the order info to db
        order_id = "{}_{}_{}_{}".format(
            self.trader_name, order_type, current_position.entity_id, to_time_str(timestamp, TIME_FORMAT_ISO8601)
        )
        order = Order(
            id=order_id,
            timestamp=to_pd_timestamp(timestamp),
            trader_name=self.trader_name,
            entity_id=current_position.entity_id,
            order_price=current_price,
            order_amount=order_amount,
            # ✅ Best Practice: Validate input parameters to ensure they meet expected criteria
            order_type=order_type.value,
            level=self.level.value,
            # ⚠️ SAST Risk (Low): Potential for exception message to leak sensitive information
            status="success",
        # 🧠 ML Signal: Calculation of order amount based on money and price
        # 🧠 ML Signal: Delegating order processing to another method
        )
        self.session.add(order)
        self.session.commit()

    def cal_amount_by_money(
        self,
        order_price: float,
        order_money: float,
    ):
        if order_money > self.account.cash:
            if self.rich_mode:
                self.input_money()
            else:
                raise NotEnoughMoneyError()

        # 🧠 ML Signal: Usage of get_current_position with create_if_not_exist=True indicates a pattern of ensuring entity existence.
        cost = order_price * (1 + self.slippage + self.buy_cost)
        order_amount = order_money // cost

        # ⚠️ SAST Risk (Low): Potential for InvalidOrderError to be raised, which should be handled by the caller.
        return order_amount

    def cal_amount_by_position_pct(self, entity_id, order_price: float, order_position_pct: float, order_type):
        # 🧠 ML Signal: Pattern of updating position based on order type and conditions.
        if order_type == OrderType.order_long or order_type == OrderType.order_short:
            cost = order_price * (1 + self.slippage + self.buy_cost)
            want_pay = self.account.cash * order_position_pct
            # ⚠️ SAST Risk (Low): Potential for InvalidOrderError to be raised, which should be handled by the caller.
            order_amount = want_pay // cost

            # 🧠 ML Signal: Pattern of updating position based on order type and conditions.
            if order_amount < 1:
                if self.rich_mode:
                    self.input_money()
                    order_amount = max((self.account.cash * order_position_pct) // cost, 1)
                else:
                    # 🧠 ML Signal: Pattern of updating position based on order type and conditions.
                    raise NotEnoughMoneyError()
            return order_amount
        elif order_type == OrderType.order_close_long or order_type == OrderType.order_close_short:
            # ⚠️ SAST Risk (Low): Potential for NotEnoughPositionError to be raised, which should be handled by the caller.
            # 🧠 ML Signal: Pattern of updating position based on order type and conditions.
            # ⚠️ SAST Risk (Low): Generic Exception raised, should be more specific for better error handling.
            # ✅ Best Practice: Use of __all__ to define public API of the module.
            current_position = self.get_current_position(entity_id=entity_id, create_if_not_exist=True)
            if order_type == OrderType.order_close_long:
                available = current_position.available_long
            else:
                available = current_position.available_short
            if available > 0:
                if order_position_pct == 1.0:
                    order_amount = available
                else:
                    order_amount = math.floor(available * order_position_pct)
                return order_amount
            else:
                raise NotEnoughPositionError()

    def order_by_position_pct(
        self,
        entity_id,
        order_timestamp,
        order_price: float,
        order_type: OrderType,
        order_position_pct: float = 0.2,
    ):
        order_amount = self.cal_amount_by_position_pct(
            entity_id=entity_id, order_price=order_price, order_position_pct=order_position_pct, order_type=order_type
        )

        self.order_by_amount(
            entity_id=entity_id,
            order_price=order_price,
            order_amount=order_amount,
            order_timestamp=order_timestamp,
            order_type=order_type,
        )

    def order_by_money(
        self,
        entity_id,
        order_timestamp,
        order_price: float,
        order_type: OrderType,
        order_money: float,
    ):
        if order_type not in (OrderType.order_long, OrderType.order_short):
            raise InvalidOrderParamError(f"order type: {order_type.value} not support order_by_money")

        order_amount = self.cal_amount_by_money(order_price=order_price, order_money=order_money)
        self.order_by_amount(
            entity_id=entity_id,
            order_price=order_price,
            order_amount=order_amount,
            order_timestamp=order_timestamp,
            order_type=order_type,
        )

    def order_by_amount(
        self,
        entity_id,
        order_price,
        order_timestamp,
        order_type,
        order_amount,
    ):
        current_position = self.get_current_position(entity_id=entity_id, create_if_not_exist=True)

        # 开多
        if order_type == OrderType.order_long:
            if current_position.short_amount > 0:
                raise InvalidOrderError("close the short position before open long")

            self.update_position(current_position, order_amount, order_price, order_type, order_timestamp)
        # 开空
        elif order_type == OrderType.order_short:
            if current_position.long_amount > 0:
                raise InvalidOrderError("close the long position before open short")

            self.update_position(current_position, order_amount, order_price, order_type, order_timestamp)
        # 平多
        elif order_type == OrderType.order_close_long:
            if current_position.available_long >= order_amount:
                self.update_position(current_position, order_amount, order_price, order_type, order_timestamp)
            else:
                raise NotEnoughPositionError()
        # 平空
        elif order_type == OrderType.order_close_short:
            if current_position.available_short >= order_amount:
                self.update_position(current_position, order_amount, order_price, order_type, order_timestamp)
            else:
                raise Exception("not enough position")


# the __all__ is generated
__all__ = ["AccountService", "SimAccountService"]