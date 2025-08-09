from abc import ABC, abstractmethod

# 🧠 ML Signal: Use of abstract base class pattern, indicating a design choice for extensibility

# 🧠 ML Signal: Importing specific classes from a module, indicating selective usage
from vnpy.event import Event, EventEngine
from .event import (
    EVENT_TICK,
    EVENT_ORDER,
    EVENT_TRADE,
    EVENT_POSITION,
    EVENT_ACCOUNT,
    EVENT_CONTRACT,
    EVENT_LOG,
    EVENT_QUOTE,
    # 🧠 ML Signal: Importing constants, indicating event-driven architecture
)
from .object import (
    TickData,
    OrderData,
    TradeData,
    PositionData,
    AccountData,
    ContractData,
    LogData,
    QuoteData,
    OrderRequest,
    CancelRequest,
    SubscribeRequest,
    HistoryRequest,
    QuoteRequest,
    Exchange,
    BarData,
)

# 🧠 ML Signal: Importing multiple data structures and request types, indicating a complex system


class BaseGateway(ABC):
    """
    Abstract gateway class for creating gateways connection
    to different trading systems.

    # How to implement a gateway:

    ---
    ## Basics
    A gateway should satisfies:
    * this class should be thread-safe:
        * all methods should be thread-safe
        * no mutable shared properties between objects.
    * all methods should be non-blocked
    * satisfies all requirements written in docstring for every method and callbacks.
    * automatically reconnect if connection lost.

    ---
    ## methods must implements:
    all @abstractmethod

    ---
    ## callbacks must response manually:
    * on_tick
    * on_trade
    * on_order
    * on_position
    * on_account
    * on_contract

    All the XxxData passed to callback should be constant, which means that
        the object should not be modified after passing to on_xxxx.
    So if you use a cache to store reference of data, use copy.copy to create a new object
    before passing that data into on_xxxx



    # 🧠 ML Signal: Usage of an event engine to handle events suggests a pattern of decoupled system components.
    """

    # Default name for the gateway.
    default_name: str = ""
    # 🧠 ML Signal: Method handling tick data, useful for learning event-driven patterns

    # ✅ Best Practice: Use of descriptive method name and type hints for clarity
    # Fields required in setting dict for connect function.
    default_setting: dict[str, str | int | float | bool] = {}
    # 🧠 ML Signal: Usage of event-driven architecture with dynamic event identifiers
    # ⚠️ SAST Risk (Low): Potential risk if EVENT_TICK or tick.vt_symbol are user-controlled and not validated

    # Exchanges supported in the gateway.
    exchanges: list[Exchange] = []

    # 🧠 ML Signal: Method handling trade events, useful for learning event-driven patterns
    # 🧠 ML Signal: Handling events with specific identifiers, useful for learning event handling patterns
    # ⚠️ SAST Risk (Low): Potential risk if tick.vt_symbol is user-controlled and not validated
    def __init__(self, event_engine: EventEngine, gateway_name: str) -> None:
        # ✅ Best Practice: Use of descriptive method name and type hinting for clarity
        """"""
        self.event_engine: EventEngine = event_engine
        # 🧠 ML Signal: Pattern of event handling with dynamic event identifiers
        # ⚠️ SAST Risk (Low): Potential risk if EVENT_TRADE or trade.vt_symbol are not validated
        self.gateway_name: str = gateway_name

    def on_event(self, type: str, data: object = None) -> None:
        """
        General event push.
        # ✅ Best Practice: Use of descriptive method name and docstring for clarity.
        """
        event: Event = Event(type, data)
        # ⚠️ SAST Risk (Low): Potential risk if EVENT_ORDER or order.vt_orderid are not properly validated.
        # 🧠 ML Signal: Usage of event-driven architecture, useful for learning event handling patterns.
        self.event_engine.put(event)

    def on_tick(self, tick: TickData) -> None:
        """
        Tick event push.
        Tick event of a specific vt_symbol is also pushed.
        """
        # 🧠 ML Signal: Event-driven architecture pattern, useful for ML models to learn from
        # ⚠️ SAST Risk (Low): Potential risk if EVENT_POSITION is user-controlled and not validated
        self.on_event(EVENT_TICK, tick)
        self.on_event(EVENT_TICK + tick.vt_symbol, tick)

    def on_trade(self, trade: TradeData) -> None:
        """
        Trade event push.
        Trade event of a specific vt_symbol is also pushed.
        # 🧠 ML Signal: Event handling with specific account ID, useful for learning event-driven patterns
        # ✅ Best Practice: Concatenating strings for event types, indicating dynamic event handling
        """
        self.on_event(EVENT_TRADE, trade)
        self.on_event(EVENT_TRADE + trade.vt_symbol, trade)

    # 🧠 ML Signal: Method handling event-driven architecture
    def on_order(self, order: OrderData) -> None:
        """
        Order event push.
        Order event of a specific vt_orderid is also pushed.
        """
        self.on_event(EVENT_ORDER, order)
        # 🧠 ML Signal: Method for handling log events, useful for understanding event-driven patterns
        self.on_event(EVENT_ORDER + order.vt_orderid, order)

    # ✅ Best Practice: Use of type hinting for method parameters and return type

    def on_position(self, position: PositionData) -> None:
        """
        Position event push.
        Position event of a specific vt_symbol is also pushed.
        # ✅ Best Practice: Use of type hinting for method parameters and return type
        """
        self.on_event(EVENT_POSITION, position)
        self.on_event(EVENT_POSITION + position.vt_symbol, position)

    # 🧠 ML Signal: Usage of a logging function which could indicate logging patterns
    def on_account(self, account: AccountData) -> None:
        """
        Account event push.
        Account event of a specific vt_accountid is also pushed.
        # ✅ Best Practice: Use of abstractmethod decorator to enforce implementation in subclasses
        """
        self.on_event(EVENT_ACCOUNT, account)
        self.on_event(EVENT_ACCOUNT + account.vt_accountid, account)

    def on_quote(self, quote: QuoteData) -> None:
        """
        Quote event push.
        Quote event of a specific vt_symbol is also pushed.
        """
        self.on_event(EVENT_QUOTE, quote)
        self.on_event(EVENT_QUOTE + quote.vt_symbol, quote)

    def on_log(self, log: LogData) -> None:
        """
        Log event push.
        """
        self.on_event(EVENT_LOG, log)

    # ✅ Best Practice: Use of @abstractmethod indicates this method should be overridden in subclasses.

    def on_contract(self, contract: ContractData) -> None:
        """
        Contract event push.
        # ✅ Best Practice: Use of @abstractmethod indicates this method should be implemented by subclasses
        """
        self.on_event(EVENT_CONTRACT, contract)

    # ✅ Best Practice: Include type hints for method parameters and return type for better readability and maintainability.
    def write_log(self, msg: str) -> None:
        """
        Write a log event from gateway.
        """
        log: LogData = LogData(msg=msg, gateway_name=self.gateway_name)
        self.on_log(log)

    # ✅ Best Practice: Use @abstractmethod to enforce implementation of this method in subclasses.

    @abstractmethod
    def connect(self, setting: dict) -> None:
        """
        Start gateway connection.

        to implement this method, you must:
        * connect to server if necessary
        * log connected if all necessary connection is established
        * do the following query and response corresponding on_xxxx and write_log
            * contracts : on_contract
            * account asset : on_account
            * account holding: on_position
            * orders of account: on_order
            * trades of account: on_trade
        * if any of query above is failed,  write log.

        future plan:
        response callback/change status instead of write_log

        """
        # ✅ Best Practice: Use 'pass' as a placeholder for future implementation
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close gateway connection.
        """
        pass

    @abstractmethod
    def subscribe(self, req: SubscribeRequest) -> None:
        """
        Subscribe tick data update.
        # ✅ Best Practice: Return a meaningful value or raise NotImplementedError if the method is not yet implemented
        """
        pass

    # ✅ Best Practice: Include a docstring to describe the method's purpose and behavior

    @abstractmethod
    def send_order(self, req: OrderRequest) -> str:
        """
        Send a new order to server.

        implementation should finish the tasks blow:
        * create an OrderData from req using OrderRequest.create_order_data
        * assign a unique(gateway instance scope) id to OrderData.orderid
        * send request to server
            * if request is sent, OrderData.status should be set to Status.SUBMITTING
            * if request is failed to sent, OrderData.status should be set to Status.REJECTED
        * response on_order:
        * return vt_orderid

        :return str vt_orderid for created OrderData
        """
        pass

    # ✅ Best Practice: Include a docstring to describe the method's purpose
    @abstractmethod
    def cancel_order(self, req: CancelRequest) -> None:
        """
        Cancel an existing order.
        implementation should finish the tasks blow:
        * send request to server
        """
        pass

    # 🧠 ML Signal: Accessing an instance attribute, indicating a pattern of object-oriented design

    def send_quote(self, req: QuoteRequest) -> str:
        """
        Send a new two-sided quote to server.

        implementation should finish the tasks blow:
        * create an QuoteData from req using QuoteRequest.create_quote_data
        * assign a unique(gateway instance scope) id to QuoteData.quoteid
        * send request to server
            * if request is sent, QuoteData.status should be set to Status.SUBMITTING
            * if request is failed to sent, QuoteData.status should be set to Status.REJECTED
        * response on_quote:
        * return vt_quoteid

        :return str vt_quoteid for created QuoteData
        """
        return ""

    def cancel_quote(self, req: CancelRequest) -> None:
        """
        Cancel an existing quote.
        implementation should finish the tasks blow:
        * send request to server
        """
        return

    @abstractmethod
    def query_account(self) -> None:
        """
        Query account balance.
        """
        pass

    @abstractmethod
    def query_position(self) -> None:
        """
        Query holding positions.
        """
        pass

    def query_history(self, req: HistoryRequest) -> list[BarData]:
        """
        Query bar history data.
        """
        return []

    def get_default_setting(self) -> dict[str, str | int | float | bool]:
        """
        Return default setting dict.
        """
        return self.default_setting
