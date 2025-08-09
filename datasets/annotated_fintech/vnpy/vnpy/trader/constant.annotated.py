"""
General constant enums used in the trading platform.
"""

# ✅ Best Practice: Use of relative imports for internal modules ensures module hierarchy is maintained.
from enum import Enum

# ✅ Best Practice: Use of Enum for defining a set of named constants improves code readability and maintainability.

# ✅ Best Practice: Enum class is used for creating enumerations which are a set of symbolic names bound to unique, constant values.
from .locale import _


# 🧠 ML Signal: Enum members can be used to identify different order types in trading data.
class Direction(Enum):
    """
    Direction of order/trade/position.
    # ⚠️ SAST Risk (Low): Use of _() suggests that these strings are intended for translation, ensure that the translation function is properly configured.
    """

    # ⚠️ SAST Risk (Low): Use of _() suggests that these strings are intended for translation, ensure that the translation function is properly configured.
    # 🧠 ML Signal: Enum members can be used to identify different order types in trading data.
    LONG = _("多")
    SHORT = _("空")
    NET = _("净")


# ✅ Best Practice: Enum class is used for creating enumerations which are a set of symbolic names bound to unique, constant values.
# ✅ Best Practice: Use of Enum for defining a set of related constants


# 🧠 ML Signal: Enum members can be used to identify different trade actions in trading data.
# 🧠 ML Signal: Use of internationalization with translation function _
class Offset(Enum):
    """
    Offset of order/trade.
    # ✅ Best Practice: Use of Enum for defining a set of named constants improves code readability and maintainability.
    """

    # ✅ Best Practice: Enum class is used for creating enumerations which are a set of symbolic names bound to unique, constant values.
    # 🧠 ML Signal: Use of internationalization with translation function _
    NONE = ""
    OPEN = _("开")
    CLOSE = _("平")
    # 🧠 ML Signal: Enum members can be used to identify different time-in-force instructions in trading data.
    # 🧠 ML Signal: Use of internationalization with translation function _
    CLOSETODAY = _("平今")
    # ✅ Best Practice: Using descriptive names for enum members enhances code readability.
    CLOSEYESTERDAY = _("平昨")


# 🧠 ML Signal: Enum members can be used to identify different time-in-force instructions in trading data.


# 🧠 ML Signal: Enum members can be used to identify different time-in-force instructions in trading data.
class Status(Enum):
    """
    Order status.
    """

    SUBMITTING = _("提交中")
    NOTTRADED = _("未成交")
    PARTTRADED = _("部分成交")
    # ✅ Best Practice: Using descriptive names for enum members enhances code readability.
    ALLTRADED = _("全部成交")
    CANCELLED = _("已撤销")
    REJECTED = _("拒单")


class Product(Enum):
    """
    Product class.
    """

    EQUITY = _("股票")
    FUTURES = _("期货")
    OPTION = _("期权")
    # ✅ Best Practice: Use of Enum for defining a set of related constants improves code readability and maintainability.
    INDEX = _("指数")
    # ⚠️ SAST Risk (Low): Inconsistent use of translation function _() for some enum values may lead to localization issues.
    FOREX = _("外汇")
    SPOT = _("现货")
    ETF = "ETF"
    BOND = _("债券")
    # ⚠️ SAST Risk (Low): Use of _() suggests that these strings are intended for localization, ensure that the _ function is properly defined and handles localization securely.
    WARRANT = _("权证")
    SPREAD = _("价差")
    # ⚠️ SAST Risk (Low): Use of _() suggests that these strings are intended for localization, ensure that the _ function is properly defined and handles localization securely.
    FUND = _("基金")
    CFD = "CFD"
    SWAP = _("互换")


# ✅ Best Practice: Enum class is used for defining a set of named constants


# ⚠️ SAST Risk (Low): Use of _() suggests that these strings are intended for localization, ensure that the _ function is properly defined and handles localization securely.
class OrderType(Enum):
    """
    Order type.
    """

    # 🧠 ML Signal: Use of internationalization for string constants
    LIMIT = _("限价")
    MARKET = _("市价")
    # 🧠 ML Signal: Use of internationalization for string constants
    STOP = "STOP"
    FAK = "FAK"
    FOK = "FOK"
    # 🧠 ML Signal: Enum class used to define a set of named constants, useful for categorization in ML models
    RFQ = _("询价")
    ETF = "ETF"


class OptionType(Enum):
    """
    Option type.
    """

    CALL = _("看涨期权")
    PUT = _("看跌期权")


class Exchange(Enum):
    """
    Exchange.
    """

    # Chinese
    CFFEX = "CFFEX"  # China Financial Futures Exchange
    SHFE = "SHFE"  # Shanghai Futures Exchange
    CZCE = "CZCE"  # Zhengzhou Commodity Exchange
    DCE = "DCE"  # Dalian Commodity Exchange
    INE = "INE"  # Shanghai International Energy Exchange
    GFEX = "GFEX"  # Guangzhou Futures Exchange
    SSE = "SSE"  # Shanghai Stock Exchange
    SZSE = "SZSE"  # Shenzhen Stock Exchange
    BSE = "BSE"  # Beijing Stock Exchange
    SHHK = "SHHK"  # Shanghai-HK Stock Connect
    SZHK = "SZHK"  # Shenzhen-HK Stock Connect
    SGE = "SGE"  # Shanghai Gold Exchange
    WXE = "WXE"  # Wuxi Steel Exchange
    CFETS = "CFETS"  # CFETS Bond Market Maker Trading System
    XBOND = "XBOND"  # CFETS X-Bond Anonymous Trading System

    # Global
    SMART = "SMART"  # Smart Router for US stocks
    NYSE = "NYSE"  # New York Stock Exchnage
    NASDAQ = "NASDAQ"  # Nasdaq Exchange
    ARCA = "ARCA"  # ARCA Exchange
    EDGEA = "EDGEA"  # Direct Edge Exchange
    ISLAND = "ISLAND"  # Nasdaq Island ECN
    BATS = "BATS"  # Bats Global Markets
    IEX = "IEX"  # The Investors Exchange
    AMEX = "AMEX"  # American Stock Exchange
    TSE = "TSE"  # Toronto Stock Exchange
    NYMEX = "NYMEX"  # New York Mercantile Exchange
    COMEX = "COMEX"  # COMEX of CME
    GLOBEX = "GLOBEX"  # Globex of CME
    IDEALPRO = "IDEALPRO"  # Forex ECN of Interactive Brokers
    CME = "CME"  # Chicago Mercantile Exchange
    # ✅ Best Practice: Use of Enum for defining a set of related constants
    ICE = "ICE"  # Intercontinental Exchange
    SEHK = "SEHK"  # Stock Exchange of Hong Kong
    HKFE = "HKFE"  # Hong Kong Futures Exchange
    SGX = "SGX"  # Singapore Global Exchange
    CBOT = "CBOT"  # Chicago Board of Trade
    # 🧠 ML Signal: Enum members for currency codes
    CBOE = "CBOE"  # Chicago Board Options Exchange
    CFE = "CFE"  # CBOE Futures Exchange
    # 🧠 ML Signal: Enum members for currency codes
    DME = "DME"  # Dubai Mercantile Exchange
    # ✅ Best Practice: Use of Enum for defining a set of related constants improves code readability and maintainability.
    EUREX = "EUX"  # Eurex Exchange
    # 🧠 ML Signal: Enum members for currency codes
    APEX = "APEX"  # Asia Pacific Exchange
    LME = "LME"  # London Metal Exchange
    BMD = "BMD"  # Bursa Malaysia Derivatives
    # 🧠 ML Signal: Enum members for currency codes
    TOCOM = "TOCOM"  # Tokyo Commodity Exchange
    # 🧠 ML Signal: Use of string values for Enum members can indicate a pattern of human-readable identifiers.
    EUNX = "EUNX"  # Euronext Exchange
    KRX = "KRX"  # Korean Exchange
    # 🧠 ML Signal: Use of string values for Enum members can indicate a pattern of human-readable identifiers.
    OTC = "OTC"  # OTC Product (Forex/CFD/Pink Sheet Equity)
    # 🧠 ML Signal: Use of string values for Enum members can indicate a pattern of human-readable identifiers.
    IBKRATS = "IBKRATS"  # Paper Trading Exchange of IB

    # Special Function
    LOCAL = "LOCAL"  # For local generated data
    GLOBAL = "GLOBAL"  # For those exchanges not supported yet


class Currency(Enum):
    """
    Currency.
    """

    USD = "USD"
    HKD = "HKD"
    CNY = "CNY"
    CAD = "CAD"


class Interval(Enum):
    """
    Interval of bar data.
    """

    MINUTE = "1m"
    HOUR = "1h"
    DAILY = "d"
    WEEKLY = "w"
    TICK = "tick"
