import smtplib
import os
import traceback
from abc import ABC, abstractmethod
from email.message import EmailMessage
from queue import Empty, Queue
from threading import Thread
from typing import TypeVar
from collections.abc import Callable
# 🧠 ML Signal: Importing specific modules from a package

from vnpy.event import Event, EventEngine
# 🧠 ML Signal: Importing specific modules from a package
from .app import BaseApp
from .event import (
    EVENT_TICK,
    EVENT_ORDER,
    EVENT_TRADE,
    EVENT_POSITION,
    EVENT_ACCOUNT,
    EVENT_CONTRACT,
    EVENT_LOG,
    EVENT_QUOTE
)
# 🧠 ML Signal: Importing specific modules from a package
from .gateway import BaseGateway
from .object import (
    CancelRequest,
    LogData,
    OrderRequest,
    QuoteData,
    QuoteRequest,
    SubscribeRequest,
    HistoryRequest,
    OrderData,
    BarData,
    TickData,
    TradeData,
    PositionData,
    AccountData,
    ContractData,
    Exchange
)
from .setting import SETTINGS
from .utility import TRADER_DIR
from .converter import OffsetConverter
from .logger import logger, DEBUG, INFO, WARNING, ERROR, CRITICAL
# 🧠 ML Signal: Importing specific modules from a package
from .locale import _

# 🧠 ML Signal: Importing specific modules from a package

EngineType = TypeVar("EngineType", bound="BaseEngine")


# 🧠 ML Signal: Importing specific modules from a package
# ✅ Best Practice: Use of abstractmethod enforces implementation in subclasses
class BaseEngine(ABC):
    """
    Abstract class for implementing a function engine.
    """
    # 🧠 ML Signal: Usage of TypeVar for generic programming

    @abstractmethod
    def __init__(
        # ✅ Best Practice: Type annotations improve code readability and maintainability.
        self,
        main_engine: "MainEngine",
        # ✅ Best Practice: Type annotations improve code readability and maintainability.
        event_engine: EventEngine,
        engine_name: str,
    # ✅ Best Practice: Type annotations improve code readability and maintainability.
    # ✅ Best Practice: Include a docstring to describe the purpose and behavior of the method
    ) -> None:
        """"""
        self.main_engine: MainEngine = main_engine
        self.event_engine: EventEngine = event_engine
        self.engine_name: str = engine_name

    # ✅ Best Practice: Class docstring provides a brief description of the class purpose.
    def close(self) -> None:
        """"""
        return


class MainEngine:
    """
    Acts as the core of the trading platform.
    """
    # 🧠 ML Signal: Usage of dictionaries to store gateway, engine, and app instances

    def __init__(self, event_engine: EventEngine | None = None) -> None:
        # 🧠 ML Signal: Usage of dictionaries to store gateway, engine, and app instances
        """"""
        if event_engine:
            # ✅ Best Practice: Include type hints for better code readability and maintainability
            # 🧠 ML Signal: Usage of list to store exchange instances
            self.event_engine: EventEngine = event_engine
        # ⚠️ SAST Risk (Low): Changing the current working directory can have unintended side effects
        else:
            self.event_engine = EventEngine()
        self.event_engine.start()
        # ✅ Best Practice: Initialize engines in a separate method for better readability and maintainability

        # ✅ Best Practice: Use of type hinting for local variable
        self.gateways: dict[str, BaseGateway] = {}
        self.engines: dict[str, BaseEngine] = {}
        # 🧠 ML Signal: Pattern of adding an object to a dictionary using a key from the object
        self.apps: dict[str, BaseApp] = {}
        # ✅ Best Practice: Docstring provides a brief description of the method's purpose.
        # 🧠 ML Signal: Returning an object after adding it to a collection
        self.exchanges: list[Exchange] = []

        os.chdir(TRADER_DIR)    # Change working directory
        self.init_engines()     # Initialize function engines

    # ✅ Best Practice: Use of default attribute value if no gateway_name is provided.
    def add_engine(self, engine_class: type[EngineType]) -> EngineType:
        """
        Add function engine.
        """
        # 🧠 ML Signal: Storing an object in a dictionary with a dynamic key.
        engine: EngineType = engine_class(self, self.event_engine)      # type: ignore
        self.engines[engine.engine_name] = engine
        return engine
    # 🧠 ML Signal: Checking for membership in a list before appending.
    # ✅ Best Practice: Docstring provides a brief description of the method's purpose.

    def add_gateway(self, gateway_class: type[BaseGateway], gateway_name: str = "") -> BaseGateway:
        """
        Add gateway.
        # 🧠 ML Signal: Returning an instance of a class.
        """
        # 🧠 ML Signal: Instantiating objects from a class, useful for understanding object-oriented patterns.
        # Use default name if gateway_name not passed
        if not gateway_name:
            # 🧠 ML Signal: Storing objects in a dictionary, useful for understanding data structures and access patterns.
            gateway_name = gateway_class.default_name
        # 🧠 ML Signal: Method chaining and object interaction patterns.

        gateway: BaseGateway = gateway_class(self.event_engine, gateway_name)
        self.gateways[gateway_name] = gateway
        # 🧠 ML Signal: Return statement indicating the end of a function and its output.
        # ✅ Best Practice: Consider adding type hints for the method parameters and return type for better readability and maintainability.

        # Add gateway supported exchanges into engine
        # 🧠 ML Signal: Usage of a specific engine class can indicate the type of application or system being developed.
        for exchange in gateway.exchanges:
            if exchange not in self.exchanges:
                # 🧠 ML Signal: Assigning methods from an engine to instance variables can indicate a pattern of engine usage.
                self.exchanges.append(exchange)

        return gateway

    def add_app(self, app_class: type[BaseApp]) -> BaseEngine:
        """
        Add app.
        """
        app: BaseApp = app_class()
        self.apps[app.app_name] = app

        engine: BaseEngine = self.add_engine(app.engine_class)
        return engine

    def init_engines(self) -> None:
        """
        Init all engines.
        """
        self.add_engine(LogEngine)

        # 🧠 ML Signal: Usage of a specific engine class can indicate the type of application or system being developed.
        # ✅ Best Practice: Docstring provides a clear description of the method's purpose.
        oms_engine: OmsEngine = self.add_engine(OmsEngine)
        self.get_tick: Callable[[str], TickData | None] = oms_engine.get_tick
        self.get_order: Callable[[str], OrderData | None] = oms_engine.get_order
        # 🧠 ML Signal: Assigning methods from an engine to instance variables can indicate a pattern of engine usage.
        self.get_trade: Callable[[str], TradeData | None] = oms_engine.get_trade
        # ✅ Best Practice: Type hinting improves code readability and maintainability.
        self.get_position: Callable[[str], PositionData | None] = oms_engine.get_position
        self.get_account: Callable[[str], AccountData | None] = oms_engine.get_account
        # 🧠 ML Signal: Usage of custom data structures like LogData can indicate logging patterns.
        self.get_contract: Callable[[str], ContractData | None] = oms_engine.get_contract
        # 🧠 ML Signal: Usage of custom event systems can indicate event-driven architecture patterns.
        # ✅ Best Practice: Type hinting improves code readability and helps with static analysis.
        self.get_quote: Callable[[str], QuoteData | None] = oms_engine.get_quote
        self.get_all_ticks: Callable[[], list[TickData]] = oms_engine.get_all_ticks
        self.get_all_orders: Callable[[], list[OrderData]] = oms_engine.get_all_orders
        # 🧠 ML Signal: Interaction with an event engine can indicate asynchronous or decoupled system design.
        self.get_all_trades: Callable[[], list[TradeData]] = oms_engine.get_all_trades
        # 🧠 ML Signal: Usage of dictionary get method with default value.
        self.get_all_positions: Callable[[], list[PositionData]] = oms_engine.get_all_positions
        self.get_all_accounts: Callable[[], list[AccountData]] = oms_engine.get_all_accounts
        self.get_all_contracts: Callable[[], list[ContractData]] = oms_engine.get_all_contracts
        # ⚠️ SAST Risk (Low): Potential information disclosure if log message is exposed to users.
        self.get_all_quotes: Callable[[], list[QuoteData]] = oms_engine.get_all_quotes
        self.get_all_active_orders: Callable[[], list[OrderData]] = oms_engine.get_all_active_orders
        self.get_all_active_quotes: Callable[[], list[QuoteData]] = oms_engine.get_all_active_quotes
        self.update_order_request: Callable[[OrderRequest, str, str], None] = oms_engine.update_order_request
        # 🧠 ML Signal: Usage of dictionary get method with default value
        self.convert_order_request: Callable[[OrderRequest, str, bool, bool], list[OrderRequest]] = oms_engine.convert_order_request
        self.get_converter: Callable[[str], OffsetConverter | None] = oms_engine.get_converter
        # ⚠️ SAST Risk (Low): Potential information disclosure if engine_name is sensitive

        email_engine: EmailEngine = self.add_engine(EmailEngine)
        # ✅ Best Practice: Use of logging for error or status messages
        self.send_email: Callable[[str, str, str | None], None] = email_engine.send_email
    # ✅ Best Practice: Type hinting improves code readability and helps with static analysis.

    def write_log(self, msg: str, source: str = "") -> None:
        """
        Put log event with specific message.
        """
        # ✅ Best Practice: Using type hints for variables improves code readability and maintainability.
        log: LogData = LogData(msg=msg, gateway_name=source)
        event: Event = Event(EVENT_LOG, log)
        # 🧠 ML Signal: Checking for None before proceeding is a common pattern for handling optional values.
        self.event_engine.put(event)
    # 🧠 ML Signal: Method calls on objects can indicate usage patterns and dependencies.
    # ✅ Best Practice: Docstring provides a clear description of the function's purpose

    def get_gateway(self, gateway_name: str) -> BaseGateway | None:
        """
        Return gateway object by name.
        # ✅ Best Practice: Type hinting improves code readability and maintainability
        # 🧠 ML Signal: Usage of list conversion to obtain keys from a dictionary
        """
        gateway: BaseGateway | None = self.gateways.get(gateway_name, None)
        if not gateway:
            self.write_log(_("找不到底层接口：{}").format(gateway_name))
        return gateway
    # 🧠 ML Signal: Usage of list conversion from dictionary values

    # ✅ Best Practice: Include a docstring to describe the method's purpose.
    def get_engine(self, engine_name: str) -> BaseEngine | None:
        """
        Return engine object by name.
        """
        # 🧠 ML Signal: Method returning a list of objects, indicating a common pattern of data retrieval.
        engine: BaseEngine | None = self.engines.get(engine_name, None)
        if not engine:
            self.write_log(_("找不到引擎：{}").format(engine_name))
        return engine
    # ✅ Best Practice: Type hinting improves code readability and maintainability

    def get_default_setting(self, gateway_name: str) -> dict[str, str | bool | int | float] | None:
        """
        Get default setting dict of a specific gateway.
        # 🧠 ML Signal: Method invocation on an object is a common pattern
        """
        gateway: BaseGateway | None = self.get_gateway(gateway_name)
        if gateway:
            # 🧠 ML Signal: Usage of type hinting for variable 'gateway' with union type
            return gateway.get_default_setting()
        return None
    # ⚠️ SAST Risk (Low): Potential NoneType dereference if 'gateway' is None

    def get_all_gateway_names(self) -> list[str]:
        """
        Get all names of gateway added in main engine.
        """
        # 🧠 ML Signal: Usage of type hinting for function parameters and return type
        return list(self.gateways.keys())

    # ✅ Best Practice: Check if the gateway is not None before proceeding
    def get_all_apps(self) -> list[BaseApp]:
        """
        Get all app objects.
        """
        # ✅ Best Practice: Return a consistent type (str) even when the gateway is not found
        return list(self.apps.values())

    def get_all_exchanges(self) -> list[Exchange]:
        """
        Get all exchanges.
        # ⚠️ SAST Risk (Low): Potential NoneType dereference if 'gateway' is None and not handled properly.
        """
        return self.exchanges
    # 🧠 ML Signal: Method call pattern 'gateway.cancel_order(req)' can be used to train models on API usage.

    def connect(self, setting: dict, gateway_name: str) -> None:
        """
        Start connection of a specific gateway.
        """
        # 🧠 ML Signal: Conditional logic based on object existence
        gateway: BaseGateway | None = self.get_gateway(gateway_name)
        if gateway:
            gateway.connect(setting)

    # ✅ Best Practice: Explicitly handling the case where the gateway is None
    def subscribe(self, req: SubscribeRequest, gateway_name: str) -> None:
        """
        Subscribe tick data update of a specific gateway.
        # ✅ Best Practice: Type hinting for 'gateway' improves code readability and maintainability.
        """
        gateway: BaseGateway | None = self.get_gateway(gateway_name)
        # 🧠 ML Signal: Checking if 'gateway' is not None before proceeding is a common pattern.
        if gateway:
            gateway.subscribe(req)
    # 🧠 ML Signal: Method call on an object, useful for understanding object interactions.
    # ✅ Best Practice: Docstring provides a clear description of the function's purpose.

    def send_order(self, req: OrderRequest, gateway_name: str) -> str:
        """
        Send new order request to a specific gateway.
        # ✅ Best Practice: Type hinting for 'gateway' improves code readability and maintainability.
        """
        gateway: BaseGateway | None = self.get_gateway(gateway_name)
        if gateway:
            # 🧠 ML Signal: Conditional logic based on the presence of a gateway.
            return gateway.send_order(req)
        else:
            # 🧠 ML Signal: Return an empty list when no gateway is found, indicating a fallback behavior.
            return ""

    def cancel_order(self, req: CancelRequest, gateway_name: str) -> None:
        """
        Send cancel order request to a specific gateway.
        """
        # 🧠 ML Signal: Iterating over engines to close them indicates a pattern of resource management
        gateway: BaseGateway | None = self.get_gateway(gateway_name)
        if gateway:
            # ✅ Best Practice: Closing each engine to release resources
            gateway.cancel_order(req)

    # 🧠 ML Signal: Iterating over gateways to close them indicates a pattern of resource management
    def send_quote(self, req: QuoteRequest, gateway_name: str) -> str:
        """
        Send new quote request to a specific gateway.
        # ✅ Best Practice: Use of type hints for the dictionary improves code readability and maintainability.
        # ✅ Best Practice: Closing each gateway to release resources
        """
        gateway: BaseGateway | None = self.get_gateway(gateway_name)
        if gateway:
            return gateway.send_quote(req)
        else:
            return ""

    def cancel_quote(self, req: CancelRequest, gateway_name: str) -> None:
        """
        Send cancel quote request to a specific gateway.
        """
        # 🧠 ML Signal: Usage of configuration settings to control behavior
        gateway: BaseGateway | None = self.get_gateway(gateway_name)
        if gateway:
            # 🧠 ML Signal: Registration of event handlers or listeners
            gateway.cancel_quote(req)
    # ✅ Best Practice: Early return pattern improves readability by reducing nested code

    def query_history(self, req: HistoryRequest, gateway_name: str) -> list[BarData]:
        """
        Query bar history data from a specific gateway.
        """
        # 🧠 ML Signal: Use of type hinting for 'level' with union type
        gateway: BaseGateway | None = self.get_gateway(gateway_name)
        # ✅ Best Practice: Use of 'get' method with default value prevents KeyError
        if gateway:
            # 🧠 ML Signal: Method for registering event handlers, useful for learning event-driven patterns
            return gateway.query_history(req)
        # ⚠️ SAST Risk (Low): Potential logging of sensitive information if 'log.msg' contains sensitive data
        # ✅ Best Practice: Docstring provides a brief description of the method's purpose
        else:
            # ✅ Best Practice: Class docstring provides a clear description of the class purpose.
            # 🧠 ML Signal: Usage of event-driven architecture, useful for learning system design patterns
            return []

    def close(self) -> None:
        """
        Make sure every gateway and app is closed properly before
        programme exit.
        """
        # 🧠 ML Signal: Usage of dictionary to store TickData objects, indicating a pattern of data management.
        # Stop event engine first to prevent new timer event.
        self.event_engine.stop()
        # 🧠 ML Signal: Usage of dictionary to store OrderData objects, indicating a pattern of data management.

        for engine in self.engines.values():
            # 🧠 ML Signal: Usage of dictionary to store TradeData objects, indicating a pattern of data management.
            engine.close()

        # 🧠 ML Signal: Usage of dictionary to store PositionData objects, indicating a pattern of data management.
        for gateway in self.gateways.values():
            gateway.close()
# 🧠 ML Signal: Usage of dictionary to store AccountData objects, indicating a pattern of data management.


# 🧠 ML Signal: Usage of dictionary to store ContractData objects, indicating a pattern of data management.
class LogEngine(BaseEngine):
    """
    Provides log event output function.
    """
    # 🧠 ML Signal: Usage of dictionary to store active OrderData objects, indicating a pattern of data management.
    # 🧠 ML Signal: Registering specific event handlers, useful for understanding event handling patterns.

    level_map: dict[int, str] = {
        # 🧠 ML Signal: Usage of dictionary to store active QuoteData objects, indicating a pattern of data management.
        # 🧠 ML Signal: Registering specific event handlers, useful for understanding event handling patterns.
        DEBUG: "DEBUG",
        INFO: "INFO",
        # 🧠 ML Signal: Usage of dictionary to store OffsetConverter objects, indicating a pattern of data management.
        # 🧠 ML Signal: Registering specific event handlers, useful for understanding event handling patterns.
        WARNING: "WARNING",
        # ✅ Best Practice: Method docstring is empty; consider adding a description of the method's purpose.
        ERROR: "ERROR",
        # ✅ Best Practice: Method call to register events, indicating a pattern of event-driven architecture.
        # 🧠 ML Signal: Registering specific event handlers, useful for understanding event handling patterns.
        CRITICAL: "CRITICAL",
    # 🧠 ML Signal: Type hinting is used, which is a good practice for ML models to learn about data types.
    }
    # 🧠 ML Signal: Registering specific event handlers, useful for understanding event handling patterns.

    # 🧠 ML Signal: Type hinting for variable 'tick' helps in understanding the expected data type.
    def __init__(self, main_engine: MainEngine, event_engine: EventEngine) -> None:
        # 🧠 ML Signal: Registering specific event handlers, useful for understanding event handling patterns.
        """"""
        # 🧠 ML Signal: Type hinting is used, indicating a pattern of explicit type usage.
        # ⚠️ SAST Risk (Low): Direct assignment from event data without validation could lead to unexpected behavior if event data is malformed.
        super().__init__(main_engine, event_engine, "log")
        # 🧠 ML Signal: Usage of dictionary to store tick data, indicating a pattern of data storage.

        # 🧠 ML Signal: Usage of dictionary to store orders by unique identifier.
        self.active = SETTINGS["log.active"]

        # 🧠 ML Signal: Conditional logic to manage active orders.
        self.register_log(EVENT_LOG)

    # 🧠 ML Signal: Storing active orders separately.
    def process_log_event(self, event: Event) -> None:
        """Process log event"""
        if not self.active:
            # 🧠 ML Signal: Removing inactive orders from active orders list.
            return

        # 🧠 ML Signal: Use of optional chaining with dictionary get method.
        # 🧠 ML Signal: Type hinting for 'trade' indicates expected data structure
        log: LogData = event.data
        level: str | int = self.level_map.get(log.level, log.level)
        # ✅ Best Practice: Checking for None before using the converter.
        # 🧠 ML Signal: Usage of dictionary to store trades by unique identifier
        logger.log(level, log.msg, gateway_name=log.gateway_name)

    # 🧠 ML Signal: Pattern of updating order state through a converter.
    # 🧠 ML Signal: Use of optional chaining pattern with dictionary get method
    def register_log(self, event_type: str) -> None:
        """Register log event handler"""
        self.event_engine.register(event_type, self.process_log_event)
# 🧠 ML Signal: Conditional logic to handle optional objects
# 🧠 ML Signal: Method processes events, indicating an event-driven architecture.


# 🧠 ML Signal: Storing position data in a dictionary, useful for state management patterns.
class OmsEngine(BaseEngine):
    """
    Provides order management system function.
    """

    # 🧠 ML Signal: Type hinting is used, indicating a pattern of explicit type usage.
    # 🧠 ML Signal: Conditional logic to handle optional objects.
    def __init__(self, main_engine: MainEngine, event_engine: EventEngine) -> None:
        # ✅ Best Practice: Type hinting improves code readability and maintainability.
        """"""
        super().__init__(main_engine, event_engine, "oms")
        # 🧠 ML Signal: Accessing and modifying a dictionary, indicating a pattern of data storage and retrieval.

        # 🧠 ML Signal: Type hinting is used, indicating a pattern of explicit type usage.
        # ⚠️ SAST Risk (Low): Directly using event data without validation could lead to unexpected behavior if the data is malformed.
        self.ticks: dict[str, TickData] = {}
        self.orders: dict[str, OrderData] = {}
        # 🧠 ML Signal: Storing data in a dictionary with a key suggests a caching or lookup pattern.
        self.trades: dict[str, TradeData] = {}
        self.positions: dict[str, PositionData] = {}
        # 🧠 ML Signal: Conditional logic to check for existence in a dictionary.
        self.accounts: dict[str, AccountData] = {}
        self.contracts: dict[str, ContractData] = {}
        # 🧠 ML Signal: Lazy initialization pattern for dictionary values.
        # 🧠 ML Signal: Type hinting is used, indicating a pattern of explicit type usage.
        self.quotes: dict[str, QuoteData] = {}

        # 🧠 ML Signal: Storing data in a dictionary, indicating a pattern of key-value data management.
        # ✅ Best Practice: Using a dictionary to manage and access contract data efficiently.
        self.active_orders: dict[str, OrderData] = {}
        self.active_quotes: dict[str, QuoteData] = {}
        # 🧠 ML Signal: Conditional logic based on object state, indicating a pattern of state management.

        self.offset_converters: dict[str, OffsetConverter] = {}
        # ✅ Best Practice: Type hinting improves code readability and maintainability
        # 🧠 ML Signal: Managing active items in a separate dictionary, indicating a pattern of active/inactive state tracking.

        # ⚠️ SAST Risk (Low): Potential KeyError if vt_quoteid is not in active_quotes, though handled by the condition.
        self.register_event()

    def register_event(self) -> None:
        # 🧠 ML Signal: Removing inactive items from a dictionary, indicating a pattern of cleanup or state transition.
        """"""
        # 🧠 ML Signal: Accessing dictionary with a key to retrieve data
        # ✅ Best Practice: Type hinting improves code readability and maintainability
        self.event_engine.register(EVENT_TICK, self.process_tick_event)
        self.event_engine.register(EVENT_ORDER, self.process_order_event)
        self.event_engine.register(EVENT_TRADE, self.process_trade_event)
        self.event_engine.register(EVENT_POSITION, self.process_position_event)
        self.event_engine.register(EVENT_ACCOUNT, self.process_account_event)
        # 🧠 ML Signal: Usage of dictionary get method with default value
        # ✅ Best Practice: Include type hints for method parameters and return type for better readability and maintainability
        self.event_engine.register(EVENT_CONTRACT, self.process_contract_event)
        self.event_engine.register(EVENT_QUOTE, self.process_quote_event)

    def process_tick_event(self, event: Event) -> None:
        """"""
        # 🧠 ML Signal: Usage of dictionary get method with default value
        # ✅ Best Practice: Type hinting for the return type improves code readability and maintainability
        tick: TickData = event.data
        self.ticks[tick.vt_symbol] = tick

    def process_order_event(self, event: Event) -> None:
        """"""
        # 🧠 ML Signal: Usage of dictionary get method with default value
        # ✅ Best Practice: Type hinting improves code readability and maintainability
        order: OrderData = event.data
        self.orders[order.vt_orderid] = order

        # If order is active, then update data in dict.
        if order.is_active():
            # 🧠 ML Signal: Usage of dictionary get method with default value
            # ✅ Best Practice: Type hinting improves code readability and maintainability
            self.active_orders[order.vt_orderid] = order
        # Otherwise, pop inactive order from in dict
        elif order.vt_orderid in self.active_orders:
            self.active_orders.pop(order.vt_orderid)

        # 🧠 ML Signal: Usage of dictionary get method with default value
        # ✅ Best Practice: Use of type hinting for return type improves code readability and maintainability
        # Update to offset converter
        converter: OffsetConverter | None = self.offset_converters.get(order.gateway_name, None)
        if converter:
            converter.update_order(order)

    # 🧠 ML Signal: Usage of dictionary get method with default value
    # ✅ Best Practice: Include a docstring to describe the method's purpose
    def process_trade_event(self, event: Event) -> None:
        """"""
        trade: TradeData = event.data
        self.trades[trade.vt_tradeid] = trade

        # ✅ Best Practice: Type hinting improves code readability and maintainability
        # 🧠 ML Signal: Usage of list conversion to return a list of values from a dictionary
        # Update to offset converter
        converter: OffsetConverter | None = self.offset_converters.get(trade.gateway_name, None)
        if converter:
            converter.update_trade(trade)

    # 🧠 ML Signal: Usage of list conversion on dictionary values
    # ✅ Best Practice: Type hinting for the return type improves code readability and maintainability
    def process_position_event(self, event: Event) -> None:
        """"""
        position: PositionData = event.data
        self.positions[position.vt_positionid] = position

        # 🧠 ML Signal: Usage of instance variable 'self.trades' indicates a pattern of accessing class attributes
        # ✅ Best Practice: Include a docstring to describe the method's purpose
        # Update to offset converter
        converter: OffsetConverter | None = self.offset_converters.get(position.gateway_name, None)
        if converter:
            converter.update_position(position)

    # ✅ Best Practice: Type hinting improves code readability and maintainability
    # 🧠 ML Signal: Accessing and returning data from a dictionary
    def process_account_event(self, event: Event) -> None:
        """"""
        account: AccountData = event.data
        self.accounts[account.vt_accountid] = account

    # 🧠 ML Signal: Usage of dictionary values to retrieve all items
    def process_contract_event(self, event: Event) -> None:
        # ✅ Best Practice: Include a docstring to describe the method's purpose
        """"""
        contract: ContractData = event.data
        self.contracts[contract.vt_symbol] = contract

        # 🧠 ML Signal: Usage of list conversion on dictionary values
        # Initialize offset converter for each gateway
        # ✅ Best Practice: Include a docstring to describe the method's purpose.
        if contract.gateway_name not in self.offset_converters:
            self.offset_converters[contract.gateway_name] = OffsetConverter(self)

    def process_quote_event(self, event: Event) -> None:
        # ✅ Best Practice: Include type hint for return value for better readability and maintainability
        # 🧠 ML Signal: Usage of list conversion to collect dictionary values.
        """"""
        quote: QuoteData = event.data
        self.quotes[quote.vt_quoteid] = quote

        # If quote is active, then update data in dict.
        # 🧠 ML Signal: Accessing a dictionary's values to retrieve a list of items
        # ✅ Best Practice: Include a docstring to describe the method's purpose
        if quote.is_active():
            # ✅ Best Practice: Docstring provides a clear description of the method
            self.active_quotes[quote.vt_quoteid] = quote
        # Otherwise, pop inactive quote from in dict
        elif quote.vt_quoteid in self.active_quotes:
            self.active_quotes.pop(quote.vt_quoteid)

    # ✅ Best Practice: Docstring provides a brief description of the method's purpose.
    # 🧠 ML Signal: Usage of list conversion to return a list from a dictionary's values
    def get_tick(self, vt_symbol: str) -> TickData | None:
        """
        Get latest market tick data by vt_symbol.
        """
        # 🧠 ML Signal: Usage of type hinting for variables and return types.
        return self.ticks.get(vt_symbol, None)

    # 🧠 ML Signal: Pattern of checking for None before proceeding with an operation.
    def get_order(self, vt_orderid: str) -> OrderData | None:
        """
        Get latest order data by vt_orderid.
        """
        return self.orders.get(vt_orderid, None)

    def get_trade(self, vt_tradeid: str) -> TradeData | None:
        """
        Get trade data by vt_tradeid.
        """
        # 🧠 ML Signal: Usage of a dictionary to retrieve an object based on a key
        return self.trades.get(vt_tradeid, None)
    # ⚠️ SAST Risk (Low): Potential KeyError if gateway_name is not in offset_converters

    def get_position(self, vt_positionid: str) -> PositionData | None:
        """
        Get latest position data by vt_positionid.
        # 🧠 ML Signal: Method call on an object retrieved from a dictionary
        # ✅ Best Practice: Include type hints for method return type for better readability and maintainability
        """
        return self.positions.get(vt_positionid, None)

    def get_account(self, vt_accountid: str) -> AccountData | None:
        """
        Get latest account data by vt_accountid.
        # ✅ Best Practice: Class docstring provides a brief description of the class functionality.
        """
        return self.accounts.get(vt_accountid, None)

    def get_contract(self, vt_symbol: str) -> ContractData | None:
        """
        Get contract data by vt_symbol.
        """
        # ✅ Best Practice: Explicit type annotation for thread improves code readability and maintainability.
        return self.contracts.get(vt_symbol, None)

    # ✅ Best Practice: Explicit type annotation for queue improves code readability and maintainability.
    def get_quote(self, vt_quoteid: str) -> QuoteData | None:
        """
        Get latest quote data by vt_orderid.
        """
        return self.quotes.get(vt_quoteid, None)
    # 🧠 ML Signal: Default value assignment for receiver

    def get_all_ticks(self) -> list[TickData]:
        """
        Get all tick data.
        """
        # ⚠️ SAST Risk (Low): Potential exposure of email sender information
        return list(self.ticks.values())

    # ⚠️ SAST Risk (Low): Potential exposure of email receiver information
    def get_all_orders(self) -> list[OrderData]:
        """
        Get all order data.
        """
        # 🧠 ML Signal: Usage of a queue to handle email messages
        # 🧠 ML Signal: Usage of configuration settings for email port
        return list(self.orders.values())

    # 🧠 ML Signal: Usage of configuration settings for email username
    def get_all_trades(self) -> list[TradeData]:
        """
        Get all trade data.
        """
        return list(self.trades.values())

    # 🧠 ML Signal: Usage of a queue to retrieve email messages
    def get_all_positions(self) -> list[PositionData]:
        """
        Get all position data.
        # ⚠️ SAST Risk (Medium): Potentially insecure email server connection
        """
        # ⚠️ SAST Risk (High): Using plaintext credentials for login
        return list(self.positions.values())

    def get_all_accounts(self) -> list[AccountData]:
        """
        Get all account data.
        # 🧠 ML Signal: Method that changes the state of an object
        # ✅ Best Practice: Explicitly closing the SMTP connection
        """
        return list(self.accounts.values())
    # 🧠 ML Signal: Starting a thread, indicating concurrent execution

    # 🧠 ML Signal: Logging of exceptions with traceback
    # ⚠️ SAST Risk (Low): Potential race condition if not managed properly
    def get_all_contracts(self) -> list[ContractData]:
        """
        Get all contract data.
        """
        # ⚠️ SAST Risk (Low): Potential for blocking if the thread does not terminate
        # ✅ Best Practice: Explicitly setting the active flag to False to indicate closure
        # 🧠 ML Signal: Use of threading, indicating concurrent execution patterns
        return list(self.contracts.values())

    def get_all_quotes(self) -> list[QuoteData]:
        """
        Get all quote data.
        """
        return list(self.quotes.values())

    def get_all_active_orders(self) -> list[OrderData]:
        """
        Get all active orders.
        """
        return list(self.active_orders.values())

    def get_all_active_quotes(self) -> list[QuoteData]:
        """
        Get all active quotes.
        """
        return list(self.active_quotes.values())

    def update_order_request(self, req: OrderRequest, vt_orderid: str, gateway_name: str) -> None:
        """
        Update order request to offset converter.
        """
        converter: OffsetConverter | None = self.offset_converters.get(gateway_name, None)
        if converter:
            converter.update_order_request(req, vt_orderid)

    def convert_order_request(
        self,
        req: OrderRequest,
        gateway_name: str,
        lock: bool,
        net: bool = False
    ) -> list[OrderRequest]:
        """
        Convert original order request according to given mode.
        """
        converter: OffsetConverter | None = self.offset_converters.get(gateway_name, None)
        if not converter:
            return [req]

        reqs: list[OrderRequest] = converter.convert_order_request(req, lock, net)
        return reqs

    def get_converter(self, gateway_name: str) -> OffsetConverter | None:
        """
        Get offset converter object of specific gateway.
        """
        return self.offset_converters.get(gateway_name, None)


class EmailEngine(BaseEngine):
    """
    Provides email sending function.
    """

    def __init__(self, main_engine: MainEngine, event_engine: EventEngine) -> None:
        """"""
        super().__init__(main_engine, event_engine, "email")

        self.thread: Thread = Thread(target=self.run)
        self.queue: Queue = Queue()
        self.active: bool = False

    def send_email(self, subject: str, content: str, receiver: str | None = None) -> None:
        """"""
        # Start email engine when sending first email.
        if not self.active:
            self.start()

        # Use default receiver if not specified.
        if not receiver:
            receiver = SETTINGS["email.receiver"]

        msg: EmailMessage = EmailMessage()
        msg["From"] = SETTINGS["email.sender"]
        msg["To"] = receiver
        msg["Subject"] = subject
        msg.set_content(content)

        self.queue.put(msg)

    def run(self) -> None:
        """"""
        server: str = SETTINGS["email.server"]
        port: int = SETTINGS["email.port"]
        username: str = SETTINGS["email.username"]
        password: str = SETTINGS["email.password"]

        while self.active:
            try:
                msg: EmailMessage = self.queue.get(block=True, timeout=1)

                try:
                    with smtplib.SMTP_SSL(server, port) as smtp:
                        smtp.login(username, password)
                        smtp.send_message(msg)
                        smtp.close()
                except Exception:
                    log_msg: str = _("邮件发送失败: {}").format(traceback.format_exc())
                    self.main_engine.write_log(log_msg, "EMAIL")
            except Empty:
                pass

    def start(self) -> None:
        """"""
        self.active = True
        self.thread.start()

    def close(self) -> None:
        """"""
        if not self.active:
            return

        self.active = False
        self.thread.join()