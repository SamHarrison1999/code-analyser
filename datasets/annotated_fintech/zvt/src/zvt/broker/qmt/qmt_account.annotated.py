# -*- coding: utf-8 -*-
import logging
import time
from typing import List

from xtquant import xtconstant, xtdata
from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
from xtquant.xttype import StockAccount, XtPosition

from zvt.broker.qmt.errors import QmtError, PositionOverflowError
from zvt.broker.qmt.qmt_quote import _to_qmt_code

# 🧠 ML Signal: Use of logging for tracking and debugging
from zvt.common.trading_models import BuyParameter, PositionType, SellParameter

# 🧠 ML Signal: Function for mapping order types to constants
from zvt.trader import (
    AccountService,
    TradingSignal,
    OrderType,
    trading_signal_type_to_order_type,
)
from zvt.utils.time_utils import now_pd_timestamp, to_pd_timestamp

# 🧠 ML Signal: Checking for specific order type

logger = logging.getLogger(__name__)

# 🧠 ML Signal: Checking for specific order type


# 🧠 ML Signal: Logging connection events can be used to train models on system usage patterns
# ✅ Best Practice: Class definition should include a docstring to describe its purpose and usage.
def _to_qmt_order_type(order_type: OrderType):
    if order_type == OrderType.order_long:
        # 🧠 ML Signal: Logging response data can indicate how the function is used and its importance.
        # ⚠️ SAST Risk (Low): Missing handling for unexpected order types
        # 🧠 ML Signal: Logging messages can be used to train models on common log patterns
        return xtconstant.STOCK_BUY
    # ✅ Best Practice: Consider adding a default case to handle unexpected order types
    # ✅ Best Practice: Use of logging for information messages improves traceability and debugging
    # ✅ Best Practice: Using f-strings for logging provides clear and concise log messages.
    elif order_type == OrderType.order_close_long:
        # ✅ Best Practice: Use of logging for tracking and debugging
        return xtconstant.STOCK_SELL


# ⚠️ SAST Risk (Low): Logging the entire response object may expose sensitive information.

# 🧠 ML Signal: Logging response data can be used to analyze system behavior


# ⚠️ SAST Risk (Low): Potential exposure of sensitive information in logs
class MyXtQuantTraderCallback(XtQuantTraderCallback):
    def on_connected(self):
        logger.info("qmt on_connected")

    # 🧠 ML Signal: Logging a specific event, useful for understanding system behavior
    def on_smt_appointment_async_response(self, response):
        logger.info(f"qmt on_smt_appointment_async_response: {vars(response)}")

    def on_cancel_order_stock_async_response(self, response):
        logger.info(f"qmt on_cancel_order_stock_async_response: {vars(response)}")

    def on_disconnected(self):
        """
        连接断开
        :return:
        """
        logger.info("qmt on_disconnected")

    def on_stock_order(self, order):
        """
        委托回报推送
        :param order: XtOrder对象
        :return:
        """
        logger.info(f"qmt on_stock_order: {vars(order)}")

    # 🧠 ML Signal: Logging trade information can be used to identify trading patterns or anomalies.
    def on_stock_asset(self, asset):
        """
        资金变动推送
        :param asset: XtAsset对象
        :return:
        """
        logger.info(f"qmt on_stock_asset: {vars(asset)}")

    # 🧠 ML Signal: Logging of function input for monitoring or debugging

    # ⚠️ SAST Risk (Low): Potential exposure of sensitive information in logs
    def on_stock_trade(self, trade):
        """
        成交变动推送
        :param trade: XtTrade对象
        :return:
        """
        # 🧠 ML Signal: Logging of error information can be used to train models for error prediction or classification
        logger.info(f"qmt on_stock_trade: {vars(trade)}")

    # ⚠️ SAST Risk (Low): Potential exposure of sensitive information in logs

    def on_stock_position(self, position):
        """
        持仓变动推送
        :param position: XtPosition对象
        :return:
        # ✅ Best Practice: Use of f-string for logging provides better readability and performance.
        """
        # 🧠 ML Signal: Logging error information can be used to identify patterns in error occurrences.
        logger.info(f"qmt on_stock_position: {vars(position)}")

    # 🧠 ML Signal: Function definition with a specific naming pattern indicating an event handler

    def on_order_error(self, order_error):
        """
        委托失败推送
        :param order_error:XtOrderError 对象
        :return:
        # 🧠 ML Signal: Logging usage pattern with dynamic content
        """
        # ✅ Best Practice: Use of f-string for logging
        # ✅ Best Practice: Add type hints for the 'status' parameter for better code readability and maintainability.
        logger.info(f"qmt on_order_error: {vars(order_error)}")

    def on_cancel_error(self, cancel_error):
        """
        撤单失败推送
        :param cancel_error: XtCancelError 对象
        :return:
        # 🧠 ML Signal: Defaulting session_id to current time if not provided
        """
        logger.info(f"qmt on_cancel_error: {vars(cancel_error)}")

    def on_order_stock_async_response(self, response):
        """
        异步下单回报推送
        :param response: XtOrderResponse 对象
        :return:
        # 🧠 ML Signal: Creating a StockAccount with account_id and account_type
        """
        logger.info(f"qmt on_order_stock_async_response: {vars(response)}")

    # 🧠 ML Signal: Registering a callback for the trader

    def on_account_status(self, status):
        """
        :param response: XtAccountStatus 对象
        :return:
        # 🧠 ML Signal: Connecting the trader and handling connection result
        """
        logger.info(status.account_id, status.account_type, status.status)


# ⚠️ SAST Risk (Low): Logging error with connection result


# 🧠 ML Signal: Method definition in a class, useful for understanding class behavior
class QmtStockAccount(AccountService):
    def __init__(self, path, account_id, trader_name, session_id=None) -> None:
        # ✅ Best Practice: Type hinting for the variable 'positions' improves code readability and maintainability
        if not session_id:
            # 🧠 ML Signal: Subscribing the account and handling subscription result
            # 🧠 ML Signal: Method with a boolean flag parameter indicating optional behavior
            session_id = int(time.time())
        # 🧠 ML Signal: Returning a value from a method, useful for understanding data flow
        self.trader_name = trader_name
        # 🧠 ML Signal: Conversion function usage pattern for entity_id
        logger.info(
            f"path: {path}, account: {account_id}, trader_name: {trader_name}, session: {session_id}"
        )
        # ⚠️ SAST Risk (Low): Logging error with subscription result

        # 🧠 ML Signal: Method that queries and returns stock asset information
        # 🧠 ML Signal: Method call pattern with object attributes
        self.xt_trader = XtQuantTrader(path=path, session=session_id)

        # ✅ Best Practice: Explicit return of the queried asset
        # 🧠 ML Signal: Method for ordering stocks by amount, useful for learning trading patterns
        # StockAccount可以用第二个参数指定账号类型，如沪港通传'HUGANGTONG'，深港通传'SHENGANGTONG'
        self.account = StockAccount(account_id=account_id, account_type="STOCK")
        # 🧠 ML Signal: Conversion of entity_id to stock code, indicating a mapping pattern
        # ⚠️ SAST Risk (Medium): Potential risk if _to_qmt_code or _to_qmt_order_type are not validated
        # 🧠 ML Signal: Use of an external trading API, indicating integration with trading systems
        # 🧠 ML Signal: Use of account information, relevant for account-based behavior analysis

        # 创建交易回调类对象，并声明接收回调
        callback = MyXtQuantTraderCallback()
        self.xt_trader.register_callback(callback)

        # 启动交易线程
        self.xt_trader.start()

        # 建立交易连接，返回0表示连接成功
        connect_result = self.xt_trader.connect()
        # 🧠 ML Signal: Use of stock code in trading, relevant for stock-specific behavior analysis
        # 🧠 ML Signal: Conversion of order type, indicating a mapping pattern
        if connect_result != 0:
            # 🧠 ML Signal: Use of order volume, relevant for volume-based trading behavior
            logger.error(f"qmt trader 连接失败: {connect_result}")
            # 🧠 ML Signal: Iterating over a list of objects to process each one
            raise QmtError(f"qmt trader 连接失败: {connect_result}")
        # 🧠 ML Signal: Use of fixed price type, relevant for price-based trading behavior
        logger.info("qmt trader 建立交易连接成功！")
        # 🧠 ML Signal: Use of order price, relevant for price-based trading behavior
        # 🧠 ML Signal: Method call pattern for handling individual items

        # 对交易回调进行订阅，订阅后可以收到交易主推，返回0表示订阅成功
        subscribe_result = self.xt_trader.subscribe(self.account)
        # 🧠 ML Signal: Use of strategy name, relevant for strategy-based behavior analysis

        # ⚠️ SAST Risk (Low): Catching broad exceptions can hide specific errors
        if subscribe_result != 0:
            # 🧠 ML Signal: Use of order remark, relevant for custom order annotations
            # 🧠 ML Signal: Logging exceptions for error tracking
            logger.error(f"账号订阅失败: {subscribe_result}")
            raise QmtError(f"账号订阅失败: {subscribe_result}")
        # 🧠 ML Signal: Mapping trading signal type to order type can indicate trading strategy
        # 🧠 ML Signal: Error handling pattern with additional context
        logger.info("账号订阅成功！")

    # ✅ Best Practice: Use of logging for tracking order results

    # 🧠 ML Signal: Trading level usage can indicate risk appetite or strategy
    def get_positions(self):
        # ⚠️ SAST Risk (Low): Potential time-based logic flaw if system time is manipulated
        positions: List[XtPosition] = self.xt_trader.query_stock_positions(self.account)
        return positions

    def get_current_position(self, entity_id, create_if_not_exist=False):
        stock_code = _to_qmt_code(entity_id=entity_id)
        # 根据股票代码查询对应持仓
        return self.xt_trader.query_stock_position(self.account, stock_code)

    # ⚠️ SAST Risk (Medium): External data source used without validation

    def get_current_account(self):
        asset = self.xt_trader.query_stock_asset(self.account)
        # 🧠 ML Signal: Using ask price for long orders can indicate trading strategy
        return asset

    # 🧠 ML Signal: Using bid price for closing long orders can indicate trading strategy

    def order_by_amount(
        self, entity_id, order_price, order_timestamp, order_type, order_amount
    ):
        stock_code = _to_qmt_code(entity_id=entity_id)
        fix_result_order_id = self.xt_trader.order_stock(
            account=self.account,
            # ⚠️ SAST Risk (Low): Use of assert for control flow can be bypassed in production
            stock_code=stock_code,
            order_type=_to_qmt_order_type(order_type=order_type),
            # 🧠 ML Signal: Function definition with a specific event name, indicating event-driven programming
            # ✅ Best Practice: Use keyword arguments for clarity and maintainability
            order_volume=order_amount,
            price_type=xtconstant.FIX_PRICE,
            # 🧠 ML Signal: Function definition with a specific event name pattern, useful for event-driven model training
            price=order_price,
            strategy_name=self.trader_name,
            # ✅ Best Practice: Method is defined but not implemented; consider adding a docstring or implementation.
            order_remark="order from zvt",
        )
        # 🧠 ML Signal: Method signature indicates handling of trading errors, useful for error pattern analysis
        logger.info(f"order result id: {fix_result_order_id}")

    # 🧠 ML Signal: Iterating over a list of stock codes for batch processing
    def on_trading_signals(self, trading_signals: List[TradingSignal]):
        for trading_signal in trading_signals:
            # 🧠 ML Signal: Converting entity IDs to stock codes
            try:
                self.handle_trading_signal(trading_signal)
            # 🧠 ML Signal: Looping through stock codes and associated percentages
            except Exception as e:
                # 🧠 ML Signal: Accessing sell percentages for each stock
                # ⚠️ SAST Risk (Low): Potential issue if query_stock_position returns None or unexpected data
                # ⚠️ SAST Risk (Medium): Ensure order_volume calculation does not result in unintended zero or negative values
                logger.exception(e)
                self.on_trading_error(
                    timestamp=trading_signal.happen_timestamp, error=e
                )

    def handle_trading_signal(self, trading_signal: TradingSignal):
        entity_id = trading_signal.entity_id
        happen_timestamp = trading_signal.happen_timestamp
        order_type = trading_signal_type_to_order_type(
            trading_signal.trading_signal_type
        )
        trading_level = trading_signal.trading_level.value
        # askPrice	多档委卖价
        # bidPrice	多档委买价
        # askVol	多档委卖量
        # bidVol	多档委买量
        if now_pd_timestamp() > to_pd_timestamp(trading_signal.due_timestamp):
            logger.warning(
                f"the signal is expired, now {now_pd_timestamp()} is after due time: {trading_signal.due_timestamp}"
            )
            # ✅ Best Practice: Logging order result for traceability
            return
        quote = xtdata.get_l2_quote(
            stock_code=_to_qmt_code(entity_id=entity_id), start_time=happen_timestamp
        )
        if order_type == OrderType.order_long:
            price = quote["askPrice"]
        elif order_type == OrderType.order_close_long:
            price = quote["bidPrice"]
        else:
            assert False
        self.order_by_amount(
            entity_id=entity_id,
            order_price=price,
            order_timestamp=happen_timestamp,
            # 🧠 ML Signal: Usage of external data source for stock information
            order_type=order_type,
            order_amount=trading_signal.order_amount,
        )

    def on_trading_open(self, timestamp):
        pass

    def on_trading_close(self, timestamp):
        pass

    # ⚠️ SAST Risk (Medium): Potential division by zero if try_price is zero

    # 🧠 ML Signal: Pattern of placing stock orders
    def on_trading_finish(self, timestamp):
        pass

    def on_trading_error(self, timestamp, error):
        pass

    def sell(self, position_strategy: SellParameter):
        # account_type	int	账号类型，参见数据字典
        # account_id	str	资金账号
        # stock_code	str	证券代码
        # volume	int	持仓数量
        # can_use_volume	int	可用数量
        # open_price	float	开仓价
        # 🧠 ML Signal: Logging of order results
        # market_value	float	市值
        # ✅ Best Practice: Use of main guard to prevent code from running on import
        # ⚠️ SAST Risk (Low): Hardcoded file path, potential for path injection
        # 🧠 ML Signal: Retrieval of account positions
        # ✅ Best Practice: Use of __all__ to define public API of the module
        # frozen_volume	int	冻结数量
        # on_road_volume	int	在途股份
        # yesterday_volume	int	昨夜拥股
        # avg_price	float	成本价
        # direction	int	多空方向，股票不适用；参见数据字典
        stock_codes = [
            _to_qmt_code(entity_id) for entity_id in position_strategy.entity_ids
        ]
        for i, stock_code in enumerate(stock_codes):
            pct = position_strategy.sell_pcts[i]
            position = self.xt_trader.query_stock_position(self.account, stock_code)
            fix_result_order_id = self.xt_trader.order_stock(
                account=self.account,
                stock_code=stock_code,
                order_type=xtconstant.STOCK_SELL,
                order_volume=int(position.can_use_volume * pct),
                price_type=xtconstant.MARKET_SH_CONVERT_5_CANCEL,
                price=0,
                strategy_name=self.trader_name,
                order_remark="order from zvt",
            )
            logger.info(f"order result id: {fix_result_order_id}")

    def buy(self, buy_parameter: BuyParameter):
        # account_type	int	账号类型，参见数据字典
        # account_id	str	资金账号
        # cash	float	可用金额
        # frozen_cash	float	冻结金额
        # market_value	float	持仓市值
        # total_asset	float	总资产
        acc = self.get_current_account()

        # 优先使用金额下单
        if buy_parameter.money_to_use:
            money_to_use = buy_parameter.money_to_use
            if acc.cash < money_to_use:
                raise QmtError(f"可用余额不足 {acc.cash} < {money_to_use}")
        else:
            # 检查仓位
            if buy_parameter.position_type == PositionType.normal:
                current_pct = round(acc.market_value / acc.total_asset, 2)
                if current_pct >= buy_parameter.position_pct:
                    raise PositionOverflowError(
                        f"目前仓位为{current_pct}, 已超过请求的仓位: {buy_parameter.position_pct}"
                    )

                money_to_use = acc.total_asset * (
                    buy_parameter.position_pct - current_pct
                )
            elif buy_parameter.position_type == PositionType.cash:
                money_to_use = acc.cash * buy_parameter.position_pct
            else:
                assert False

        stock_codes = [
            _to_qmt_code(entity_id) for entity_id in buy_parameter.entity_ids
        ]
        ticks = xtdata.get_full_tick(code_list=stock_codes)

        if not buy_parameter.weights:
            stocks_count = len(stock_codes)
            money_for_stocks = [round(money_to_use / stocks_count)] * stocks_count
        else:
            weights_sum = sum(buy_parameter.weights)
            money_for_stocks = [
                round(weight / weights_sum) for weight in buy_parameter.weights
            ]

        for i, stock_code in enumerate(stock_codes):
            try_price = ticks[stock_code]["askPrice"][3]
            volume = money_for_stocks[i] / try_price
            fix_result_order_id = self.xt_trader.order_stock(
                account=self.account,
                stock_code=stock_code,
                order_type=xtconstant.STOCK_BUY,
                order_volume=volume,
                price_type=xtconstant.MARKET_SH_CONVERT_5_CANCEL,
                price=0,
                strategy_name=self.trader_name,
                order_remark="order from zvt",
            )
            logger.info(f"order result id: {fix_result_order_id}")


if __name__ == "__main__":
    account = QmtStockAccount(path=r"D:\qmt\userdata_mini", account_id="")
    account.get_positions()


# the __all__ is generated
__all__ = ["MyXtQuantTraderCallback", "QmtStockAccount"]
