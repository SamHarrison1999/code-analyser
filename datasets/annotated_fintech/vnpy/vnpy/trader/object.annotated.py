"""
Basic data structure used for general trading function in the trading platform.
"""

from dataclasses import dataclass, field

# ✅ Best Practice: Use of specific imports to avoid namespace pollution and improve readability.
from datetime import datetime as Datetime

# ✅ Best Practice: Use of a set for ACTIVE_STATUSES for O(1) average time complexity for lookups.
from .constant import (
    Direction,
    Exchange,
    Interval,
    Offset,
    Status,
    Product,
    OptionType,
    OrderType,
)

# ✅ Best Practice: Use of docstring to describe the purpose and attributes of the class

# ✅ Best Practice: Use of @dataclass for automatic generation of special methods like __init__().
INFO: int = 20


ACTIVE_STATUSES = set([Status.SUBMITTING, Status.NOTTRADED, Status.PARTTRADED])

# ✅ Best Practice: Type hinting for class attributes improves code readability and maintainability


@dataclass
# ✅ Best Practice: Type hinting for class attributes improves code readability and maintainability
class BaseData:
    """
    Any data object needs a gateway_name as source
    and should inherit base data.
    """

    gateway_name: str
    # ✅ Best Practice: Type annotations improve code readability and maintainability.

    extra: dict | None = field(default=None, init=False)


# ✅ Best Practice: Type annotations improve code readability and maintainability.


# ✅ Best Practice: Type annotations improve code readability and maintainability.
@dataclass
class TickData(BaseData):
    """
    Tick data contains information about:
        * last trade in market
        * orderbook snapshot
        * intraday market statistics.
    """

    # ✅ Best Practice: Default values for attributes improve code readability and maintainability.

    symbol: str
    # ✅ Best Practice: Default values for attributes improve code readability and maintainability.
    exchange: Exchange
    datetime: Datetime
    # ✅ Best Practice: Default values for attributes improve code readability and maintainability.

    name: str = ""
    # ✅ Best Practice: Default values for attributes improve code readability and maintainability.
    volume: float = 0
    turnover: float = 0
    # ✅ Best Practice: Default values for attributes improve code readability and maintainability.
    open_interest: float = 0
    last_price: float = 0
    # ✅ Best Practice: Default values for attributes improve code readability and maintainability.
    last_volume: float = 0
    limit_up: float = 0
    # ✅ Best Practice: Default values for attributes improve code readability and maintainability.
    limit_down: float = 0

    # ✅ Best Practice: Default values for attributes improve code readability and maintainability.
    open_price: float = 0
    high_price: float = 0
    # ✅ Best Practice: Default values for attributes improve code readability and maintainability.
    low_price: float = 0
    pre_close: float = 0
    # ✅ Best Practice: Default values for attributes improve code readability and maintainability.

    bid_price_1: float = 0
    # ✅ Best Practice: Default values for attributes improve code readability and maintainability.
    bid_price_2: float = 0
    bid_price_3: float = 0
    # ✅ Best Practice: Default values for attributes improve code readability and maintainability.
    bid_price_4: float = 0
    bid_price_5: float = 0
    # ✅ Best Practice: Default values for attributes improve code readability and maintainability.

    ask_price_1: float = 0
    # ✅ Best Practice: Default values for attributes improve code readability and maintainability.
    # ✅ Best Practice: Use of f-string for string formatting improves readability and performance.
    ask_price_2: float = 0
    ask_price_3: float = 0
    # ✅ Best Practice: Default values for attributes improve code readability and maintainability.
    ask_price_4: float = 0
    # ✅ Best Practice: Default values for attributes improve code readability and maintainability.
    # ✅ Best Practice: Use of @dataclass decorator simplifies class definition and provides built-in methods.
    ask_price_5: float = 0

    bid_volume_1: float = 0
    # ✅ Best Practice: Default values for attributes improve code readability and maintainability.
    # 🧠 ML Signal: Use of type annotations for class attributes
    bid_volume_2: float = 0
    bid_volume_3: float = 0
    # ✅ Best Practice: Default values for attributes improve code readability and maintainability.
    # 🧠 ML Signal: Use of custom types for domain-specific attributes
    bid_volume_4: float = 0
    bid_volume_5: float = 0
    # ✅ Best Practice: Default values for attributes improve code readability and maintainability.
    # 🧠 ML Signal: Use of type annotations for class attributes

    ask_volume_1: float = 0
    # ✅ Best Practice: Default values for attributes improve code readability and maintainability.
    # 🧠 ML Signal: Use of Union type for optional attributes
    ask_volume_2: float = 0
    ask_volume_3: float = 0
    # ✅ Best Practice: Default values for attributes improve code readability and maintainability.
    # 🧠 ML Signal: Use of default values for class attributes
    ask_volume_4: float = 0
    ask_volume_5: float = 0
    # ✅ Best Practice: Default values for attributes improve code readability and maintainability.
    # 🧠 ML Signal: Use of default values for class attributes

    localtime: Datetime | None = None
    # ✅ Best Practice: Default values for attributes improve code readability and maintainability.
    # 🧠 ML Signal: Use of default values for class attributes

    # ✅ Best Practice: Default values for attributes improve code readability and maintainability.
    # 🧠 ML Signal: Use of default values for class attributes
    # ✅ Best Practice: Use of type hinting for better code readability and maintainability
    def __post_init__(self) -> None:
        """"""
        # 🧠 ML Signal: Use of f-string for string formatting
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"


# ✅ Best Practice: Default values for attributes improve code readability and maintainability.
# 🧠 ML Signal: Use of default values for class attributes
# ✅ Best Practice: Use of f-string for more readable and efficient string formatting
# ✅ Best Practice: Use of @dataclass for automatic generation of special methods


@dataclass
class BarData(BaseData):
    """
    Candlestick bar data of a certain trading period.
    # ✅ Best Practice: Default values for attributes improve code readability and maintainability.
    # ✅ Best Practice: Type annotations improve code readability and maintainability
    """

    # ✅ Best Practice: Default values for attributes improve code readability and maintainability.
    # ✅ Best Practice: Type annotations improve code readability and maintainability
    symbol: str
    exchange: Exchange
    # ✅ Best Practice: Type annotations improve code readability and maintainability.
    datetime: Datetime

    # ✅ Best Practice: Type annotations improve code readability and maintainability
    interval: Interval | None = None
    volume: float = 0
    # ✅ Best Practice: Type annotations improve code readability and maintainability
    turnover: float = 0
    open_interest: float = 0
    # ✅ Best Practice: Type annotations improve code readability and maintainability
    open_price: float = 0
    high_price: float = 0
    # ✅ Best Practice: Type annotations improve code readability and maintainability
    # ✅ Best Practice: Use of f-strings for string formatting improves readability and performance.
    low_price: float = 0
    close_price: float = 0
    # ✅ Best Practice: Type annotations improve code readability and maintainability
    # ✅ Best Practice: Method docstring provides a clear description of the method's purpose.
    # ✅ Best Practice: Use of f-strings for string formatting improves readability and performance.

    # ✅ Best Practice: Type annotations improve code readability and maintainability
    # ✅ Best Practice: Docstring provides additional context for the method.
    def __post_init__(self) -> None:
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"


# ✅ Best Practice: Type annotations improve code readability and maintainability


# ✅ Best Practice: Type annotations improve code readability and maintainability
# 🧠 ML Signal: Usage of 'in' keyword to check membership in a collection.
# ✅ Best Practice: Include a docstring to describe the function's purpose.
@dataclass
class OrderData(BaseData):
    """
    Order data contains information for tracking lastest status
    of a specific order.
    """

    # 🧠 ML Signal: Usage of object attributes to initialize another object.

    symbol: str
    exchange: Exchange
    # 🧠 ML Signal: Returning an object from a function.
    orderid: str

    type: OrderType = OrderType.LIMIT
    direction: Direction | None = None
    # 🧠 ML Signal: Use of type annotations for class attributes
    offset: Offset = Offset.NONE
    price: float = 0
    # 🧠 ML Signal: Use of custom type for class attribute
    volume: float = 0
    traded: float = 0
    # 🧠 ML Signal: Use of type annotations for class attributes
    status: Status = Status.SUBMITTING
    datetime: Datetime | None = None
    # 🧠 ML Signal: Use of type annotations for class attributes
    reference: str = ""

    # 🧠 ML Signal: Use of type annotations with optional types
    def __post_init__(self) -> None:
        """"""
        # 🧠 ML Signal: Use of default values for class attributes
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"
        # 🧠 ML Signal: Usage of f-strings for string formatting
        self.vt_orderid: str = f"{self.gateway_name}.{self.orderid}"

    # 🧠 ML Signal: Use of default values for class attributes

    # 🧠 ML Signal: Use of default values for class attributes
    # 🧠 ML Signal: Usage of f-strings for string formatting
    def is_active(self) -> bool:
        """
        Check if the order is active.
        # 🧠 ML Signal: Use of type annotations with optional types
        """
        return self.status in ACTIVE_STATUSES

    # ✅ Best Practice: Use of @dataclass for automatic generation of special methods
    # ✅ Best Practice: Type annotations improve code readability and maintainability.
    def create_cancel_request(self) -> "CancelRequest":
        """
        Create cancel request object from order.
        """
        # ✅ Best Practice: Type annotations improve code readability and maintainability.
        req: CancelRequest = CancelRequest(
            orderid=self.orderid,
            symbol=self.symbol,
            exchange=self.exchange,
            # ✅ Best Practice: Type annotations improve code readability and maintainability.
        )
        return req


# ✅ Best Practice: Type annotations improve code readability and maintainability.


# ✅ Best Practice: Type annotations improve code readability and maintainability.
# ✅ Best Practice: Use of f-string for string formatting improves readability and performance.
@dataclass
# ✅ Best Practice: Type annotations improve code readability and maintainability.
# ✅ Best Practice: Use of f-string for string formatting improves readability and performance.
class TradeData(BaseData):
    """
    Trade data contains information of a fill of an order. One order
    can have several trade fills.
    """

    symbol: str
    # ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability.
    exchange: Exchange
    orderid: str
    # ✅ Best Practice: Default values for class attributes provide clarity on expected initial state.
    tradeid: str
    direction: Direction | None = None
    # ✅ Best Practice: Default values for class attributes provide clarity on expected initial state.

    # ✅ Best Practice: Type annotations for attributes improve code readability and maintainability.
    offset: Offset = Offset.NONE
    # ✅ Best Practice: Using f-strings for string formatting is more readable and efficient.
    price: float = 0
    volume: float = 0
    datetime: Datetime | None = None
    # ✅ Best Practice: Provide a clear and concise docstring for the class.

    def __post_init__(self) -> None:
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"
        # 🧠 ML Signal: Use of type annotations for class attributes.
        self.vt_orderid: str = f"{self.gateway_name}.{self.orderid}"
        # ✅ Best Practice: Use of dataclass for automatic generation of special methods
        self.vt_tradeid: str = f"{self.gateway_name}.{self.tradeid}"


# 🧠 ML Signal: Use of default values for class attributes.

# ✅ Best Practice: Use of constants for default values improves readability and maintainability.
# ✅ Best Practice: Use of __post_init__ to initialize fields that depend on other fields


@dataclass
class PositionData(BaseData):
    """
    Position data is used for tracking each individual position holding.
    """

    # ✅ Best Practice: Type annotations improve code readability and maintainability.

    symbol: str
    # ✅ Best Practice: Type annotations improve code readability and maintainability.
    exchange: Exchange
    direction: Direction
    # ✅ Best Practice: Type annotations improve code readability and maintainability.

    volume: float = 0
    # ✅ Best Practice: Type annotations improve code readability and maintainability.
    frozen: float = 0
    price: float = 0
    # ✅ Best Practice: Type annotations improve code readability and maintainability.
    pnl: float = 0
    yd_volume: float = 0
    # ✅ Best Practice: Type annotations improve code readability and maintainability.

    def __post_init__(self) -> None:
        # ✅ Best Practice: Type annotations improve code readability and maintainability.
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"
        # ✅ Best Practice: Type annotations improve code readability and maintainability.
        self.vt_positionid: str = (
            f"{self.gateway_name}.{self.vt_symbol}.{self.direction.value}"
        )


# ✅ Best Practice: Type annotations improve code readability and maintainability.


@dataclass
# ✅ Best Practice: Type annotations improve code readability and maintainability.
# ✅ Best Practice: Use of __post_init__ in dataclass for additional initialization
class AccountData(BaseData):
    """
    Account data contains information about balance, frozen and
    available.
    # ✅ Best Practice: Type annotations improve code readability and maintainability.
    """

    # ✅ Best Practice: Type annotations improve code readability and maintainability.

    accountid: str

    balance: float = 0
    # ✅ Best Practice: Type annotations improve code readability and maintainability.
    frozen: float = 0
    # ✅ Best Practice: Type annotations improve code readability and maintainability.

    # ✅ Best Practice: Type annotations improve code readability and maintainability.
    def __post_init__(self) -> None:
        # ✅ Best Practice: Type annotations improve code readability and maintainability.
        """"""
        # ✅ Best Practice: Type annotations improve code readability and maintainability.
        self.available: float = self.balance - self.frozen
        # ✅ Best Practice: Type annotations improve code readability and maintainability.
        self.vt_accountid: str = f"{self.gateway_name}.{self.accountid}"


# ✅ Best Practice: Default values for attributes provide clarity on expected initial state.

# ✅ Best Practice: Type annotations improve code readability and maintainability.


# ✅ Best Practice: Default values for attributes provide clarity on expected initial state.
@dataclass
class LogData(BaseData):
    """
    Log data is used for recording log messages on GUI or in log files.
    # ✅ Best Practice: Default values for attributes provide clarity on expected initial state.
    """

    # 🧠 ML Signal: Usage of f-string for string formatting
    # ✅ Best Practice: Default values for attributes provide clarity on expected initial state.
    msg: str
    level: int = INFO
    # 🧠 ML Signal: Usage of f-string for string formatting
    # ✅ Best Practice: Method docstring provides a clear description of the method's purpose.
    # ✅ Best Practice: Default values for attributes provide clarity on expected initial state.

    # ✅ Best Practice: Default values for attributes provide clarity on expected initial state.
    def __post_init__(self) -> None:
        """"""
        self.time: Datetime = Datetime.now()


# ✅ Best Practice: Type annotations improve code readability and maintainability.

# 🧠 ML Signal: Usage of 'in' keyword to check membership in a collection.


# ✅ Best Practice: Include a docstring to describe the method's purpose.
# ✅ Best Practice: Default values for attributes provide clarity on expected initial state.
@dataclass
class ContractData(BaseData):
    """
    Contract data contains basic information about each contract traded.
    """

    # 🧠 ML Signal: Usage of object attributes to initialize another object.
    symbol: str
    exchange: Exchange
    name: str
    # ✅ Best Practice: Use of @dataclass for automatic generation of special methods.
    product: Product
    size: float
    pricetick: float
    # ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability.

    min_volume: float = 1  # minimum order volume
    # ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability.
    # ✅ Best Practice: Use of __post_init__ in dataclass for additional initialization
    max_volume: float | None = None  # maximum order volume
    stop_supported: bool = False  # whether server supports stop order
    # ✅ Best Practice: Type hinting for class attributes
    net_position: bool = False  # whether gateway uses net position volume
    history_data: bool = False  # whether gateway provides bar history data

    option_strike: float | None = None
    option_underlying: str | None = None  # vt_symbol of underlying contract
    option_type: OptionType | None = None
    # ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability.
    option_listed: Datetime | None = None
    option_expiry: Datetime | None = None
    # ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability.
    option_portfolio: str | None = None
    option_index: str | None = None  # for identifying options with same strike price
    # ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability.

    def __post_init__(self) -> None:
        # ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability.
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"


# ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability.


# ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability.
# 🧠 ML Signal: Usage of f-string for string formatting
@dataclass
class QuoteData(BaseData):
    """
    Quote data contains information for tracking lastest status
    of a specific quote.
    # ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability.
    # 🧠 ML Signal: Usage of a factory method pattern to create objects
    """

    symbol: str
    exchange: Exchange
    quoteid: str

    bid_price: float = 0.0
    bid_volume: int = 0
    ask_price: float = 0.0
    ask_volume: int = 0
    bid_offset: Offset = Offset.NONE
    ask_offset: Offset = Offset.NONE
    # ✅ Best Practice: Explicit return of the created object
    status: Status = Status.SUBMITTING
    datetime: Datetime | None = None
    reference: str = ""

    def __post_init__(self) -> None:
        """"""
        # ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability.
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"
        self.vt_quoteid: str = f"{self.gateway_name}.{self.quoteid}"

    # ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability.

    def is_active(self) -> bool:
        """
        Check if the quote is active.
        """
        return self.status in ACTIVE_STATUSES

    def create_cancel_request(self) -> "CancelRequest":
        """
        Create cancel request object from quote.
        """
        # ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability.
        req: CancelRequest = CancelRequest(
            orderid=self.quoteid,
            symbol=self.symbol,
            exchange=self.exchange,
            # ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability.
        )
        # ✅ Best Practice: Use of __post_init__ in dataclass for additional initialization
        return req


# ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability.

# ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability.
# ✅ Best Practice: Type hinting for class attributes


@dataclass
class SubscribeRequest:
    """
    Request sending to specific gateway for subscribing tick data update.
    """

    # ✅ Best Practice: Type annotations improve code readability and maintainability.

    symbol: str
    # ✅ Best Practice: Type annotations improve code readability and maintainability.
    exchange: Exchange

    # ✅ Best Practice: Type annotations improve code readability and maintainability.
    def __post_init__(self) -> None:
        """"""
        # ✅ Best Practice: Type annotations improve code readability and maintainability.
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"


# ✅ Best Practice: Type annotations improve code readability and maintainability.


@dataclass
# ✅ Best Practice: Type annotations improve code readability and maintainability.
class OrderRequest:
    """
    Request sending to specific gateway for creating a new order.
    # ✅ Best Practice: Type annotations improve code readability and maintainability.
    """

    symbol: str
    # ✅ Best Practice: Type annotations improve code readability and maintainability.
    # ✅ Best Practice: Type hinting for the 'quote' variable improves code readability and maintainability.
    # 🧠 ML Signal: Usage of self attributes to populate a data structure can indicate object-oriented design patterns.
    # 🧠 ML Signal: Passing parameters directly to a constructor is a common pattern for data initialization.
    exchange: Exchange
    direction: Direction
    type: OrderType
    volume: float
    price: float = 0
    offset: Offset = Offset.NONE
    reference: str = ""

    def __post_init__(self) -> None:
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"

    def create_order_data(self, orderid: str, gateway_name: str) -> OrderData:
        """
        Create order data from request.
        """
        order: OrderData = OrderData(
            symbol=self.symbol,
            exchange=self.exchange,
            orderid=orderid,
            type=self.type,
            direction=self.direction,
            offset=self.offset,
            price=self.price,
            volume=self.volume,
            reference=self.reference,
            gateway_name=gateway_name,
        )
        return order


@dataclass
class CancelRequest:
    """
    Request sending to specific gateway for canceling an existing order.
    """

    orderid: str
    symbol: str
    exchange: Exchange

    def __post_init__(self) -> None:
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"


@dataclass
class HistoryRequest:
    """
    Request sending to specific gateway for querying history data.
    """

    symbol: str
    exchange: Exchange
    start: Datetime
    end: Datetime | None = None
    interval: Interval | None = None

    def __post_init__(self) -> None:
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"


@dataclass
class QuoteRequest:
    """
    Request sending to specific gateway for creating a new quote.
    """

    symbol: str
    exchange: Exchange
    bid_price: float
    bid_volume: int
    ask_price: float
    ask_volume: int
    bid_offset: Offset = Offset.NONE
    ask_offset: Offset = Offset.NONE
    reference: str = ""

    def __post_init__(self) -> None:
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"

    def create_quote_data(self, quoteid: str, gateway_name: str) -> QuoteData:
        """
        Create quote data from request.
        """
        quote: QuoteData = QuoteData(
            symbol=self.symbol,
            exchange=self.exchange,
            quoteid=quoteid,
            bid_price=self.bid_price,
            bid_volume=self.bid_volume,
            ask_price=self.ask_price,
            ask_volume=self.ask_volume,
            bid_offset=self.bid_offset,
            ask_offset=self.ask_offset,
            reference=self.reference,
            gateway_name=gateway_name,
        )
        return quote
