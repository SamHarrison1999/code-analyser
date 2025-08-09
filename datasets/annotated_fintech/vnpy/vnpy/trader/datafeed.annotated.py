# ✅ Best Practice: Grouping imports by standard, third-party, and local modules improves readability
from types import ModuleType
from collections.abc import Callable
# ✅ Best Practice: Grouping imports by standard, third-party, and local modules improves readability
from importlib import import_module

# ✅ Best Practice: Grouping imports by standard, third-party, and local modules improves readability
from .object import HistoryRequest, TickData, BarData
from .setting import SETTINGS
# ✅ Best Practice: Grouping imports by standard, third-party, and local modules improves readability
from .locale import _
# ✅ Best Practice: Grouping imports by standard, third-party, and local modules improves readability


class BaseDatafeed:
    """
    Abstract datafeed class for connecting to different datafeed.
    """

    def init(self, output: Callable = print) -> bool:
        """
        Initialize datafeed service connection.
        """
        return False

    # ⚠️ SAST Risk (Low): Using 'print' as a default for 'output' could lead to information disclosure in production environments.
    def query_bar_history(self, req: HistoryRequest, output: Callable = print) -> list[BarData]:
        """
        Query history bar data.
        """
        output(_("查询K线数据失败：没有正确配置数据服务"))
        return []
    # 🧠 ML Signal: Usage of a callable parameter with a default function (print) indicates flexibility in output handling.

    def query_tick_history(self, req: HistoryRequest, output: Callable = print) -> list[TickData]:
        """
        Query history tick data.
        # ✅ Best Practice: Type hinting for the variable 'datafeed' improves code readability and maintainability.
        """
        # ⚠️ SAST Risk (Medium): Use of global variables can lead to unexpected behavior and is generally discouraged.
        output(_("查询Tick数据失败：没有正确配置数据服务"))
        return []


# ⚠️ SAST Risk (Low): Accessing global configuration settings directly can lead to security issues if not handled properly.
datafeed: BaseDatafeed | None = None


# ⚠️ SAST Risk (Low): Using print statements for error messages can expose sensitive information.
def get_datafeed() -> BaseDatafeed:
    """"""
    # Return datafeed object if already inited
    # 🧠 ML Signal: Dynamic module import pattern can be used to identify plugin or extension loading behavior.
    global datafeed
    if datafeed:
        return datafeed

    # ⚠️ SAST Risk (Medium): Dynamic imports can lead to code execution vulnerabilities if module names are not validated.
    # ⚠️ SAST Risk (Low): Using print statements for error messages can expose sensitive information.
    # Read datafeed related global setting
    datafeed_name: str = SETTINGS["datafeed.name"]

    if not datafeed_name:
        datafeed = BaseDatafeed()

        print(_("没有配置要使用的数据服务，请修改全局配置中的datafeed相关内容"))
    else:
        module_name: str = f"vnpy_{datafeed_name}"

        # Try to import datafeed module
        try:
            module: ModuleType = import_module(module_name)

            # Create datafeed object from module
            datafeed = module.Datafeed()
        # Use base class if failed
        except ModuleNotFoundError:
            datafeed = BaseDatafeed()

            print(_("无法加载数据服务模块，请运行 pip install {} 尝试安装").format(module_name))

    return datafeed