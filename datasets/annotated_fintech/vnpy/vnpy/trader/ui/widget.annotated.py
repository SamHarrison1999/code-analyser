"""
Basic widgets for UI.
"""

import csv
import platform
from enum import Enum
from typing import cast, Any
from copy import copy
from tzlocal import get_localzone_name
from datetime import datetime
from importlib import metadata

from .qt import QtCore, QtGui, QtWidgets, Qt
from ..constant import Direction, Exchange, Offset, OrderType
from ..engine import MainEngine, Event, EventEngine
from ..event import (
    EVENT_QUOTE,
    EVENT_TICK,
    EVENT_TRADE,
    EVENT_ORDER,
    EVENT_POSITION,
    EVENT_ACCOUNT,
    EVENT_LOG
)
from ..object import (
    OrderRequest,
    SubscribeRequest,
    CancelRequest,
    ContractData,
    PositionData,
    OrderData,
    QuoteData,
    TickData
)
from ..utility import load_json, save_json, get_digits, ZoneInfo
# 🧠 ML Signal: Usage of QColor for UI element coloring
from ..setting import SETTING_FILENAME, SETTINGS
from ..locale import _
# 🧠 ML Signal: Usage of QColor for UI element coloring


# 🧠 ML Signal: Usage of QColor for UI element coloring
COLOR_LONG = QtGui.QColor("red")
COLOR_SHORT = QtGui.QColor("green")
# 🧠 ML Signal: Usage of QColor for UI element coloring
# ✅ Best Practice: Include a docstring to describe the purpose of the class
COLOR_BID = QtGui.QColor(255, 174, 201)
COLOR_ASK = QtGui.QColor(160, 255, 160)
COLOR_BLACK = QtGui.QColor("black")
# 🧠 ML Signal: Usage of QColor for UI element coloring


class BaseCell(QtWidgets.QTableWidgetItem):
    """
    General cell used in tablewidgets.
    # ✅ Best Practice: Use of type hinting improves code readability and helps with static analysis.
    """

    # ✅ Best Practice: Use of QtCore.Qt.AlignmentFlag.AlignCenter improves readability by using descriptive constants.
    def __init__(self, content: Any, data: Any) -> None:
        # 🧠 ML Signal: Method call pattern with specific parameters can be used to identify usage patterns.
        """"""
        super().__init__()

        # ✅ Best Practice: Convert content to string to ensure consistent text handling
        self._text: str = ""
        self._data: Any = None
        # 🧠 ML Signal: Storing data in an instance variable for later use

        self.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        # 🧠 ML Signal: Method call to update UI or internal state with new text
        # ✅ Best Practice: Use of type hinting for return value improves code readability and maintainability

        self.set_content(content, data)

    def set_content(self, content: Any, data: Any) -> None:
        """
        Set text content.
        """
        self._text = str(content)
        self._data = data
        # 🧠 ML Signal: Usage of comparison magic method to define custom sorting behavior

        self.setText(self._text)
    # ✅ Best Practice: Use a temporary variable for clarity and debugging purposes

    def get_data(self) -> Any:
        """
        Get data object.
        """
        return self._data
    # ✅ Best Practice: Use of super() to call the parent class's __init__ method

    # ✅ Best Practice: Type hints are used for function parameters and return type
    def __lt__(self, other: "BaseCell") -> bool:        # type: ignore
        """
        Sort by text content.
        """
        result: bool = self._text < other._text
        # ⚠️ SAST Risk (Low): Potential AttributeError if 'content' does not have 'value' attribute
        return result
# 🧠 ML Signal: Usage of superclass method with modified parameters


class EnumCell(BaseCell):
    """
    Cell used for showing enum data.
    # ✅ Best Practice: Use of type hints for function parameters and return type
    """
    # 🧠 ML Signal: Constructor method with parameters, indicating object initialization pattern

    # ✅ Best Practice: Use of super() to call the parent class's constructor
    def __init__(self, content: Enum, data: Any) -> None:
        """"""
        super().__init__(content, data)
    # ✅ Best Practice: Call to superclass method ensures proper initialization or behavior extension

    def set_content(self, content: Any, data: Any) -> None:
        """
        Set text using enum.constant.value.
        # 🧠 ML Signal: Usage of specific color for a particular condition
        """
        if content:
            # 🧠 ML Signal: Usage of specific color for a different condition
            # ✅ Best Practice: Class docstring provides a clear description of the class purpose.
            super().set_content(content.value, data)


class DirectionCell(EnumCell):
    """
    Cell used for showing direction data.
    """
    # 🧠 ML Signal: Usage of setForeground method indicates UI customization

    # ✅ Best Practice: Include a docstring to describe the purpose of the class
    def __init__(self, content: Enum, data: Any) -> None:
        """"""
        super().__init__(content, data)

    def set_content(self, content: Any, data: Any) -> None:
        """
        Cell color is set according to direction.
        # 🧠 ML Signal: Usage of self to set instance attributes
        """
        super().set_content(content, data)

        if content is Direction.SHORT:
            # ✅ Best Practice: Class docstring provides a brief description of the class purpose.
            self.setForeground(COLOR_SHORT)
        # ✅ Best Practice: Type hints for parameters and return value improve code readability and maintainability.
        else:
            self.setForeground(COLOR_LONG)
# ✅ Best Practice: Calling the superclass's __init__ method ensures proper initialization of inherited attributes.


class BidCell(BaseCell):
    """
    Cell used for showing bid price and volume.
    # ✅ Best Practice: Call to superclass method ensures base functionality is preserved
    """

    # 🧠 ML Signal: Checking if a string starts with a specific character
    def __init__(self, content: Any, data: Any) -> None:
        """"""
        # 🧠 ML Signal: Setting a color based on a condition
        super().__init__(content, data)
        # 🧠 ML Signal: Class definition indicating a custom cell for time display

        # 🧠 ML Signal: Setting a color based on a condition
        self.setForeground(COLOR_BID)


class AskCell(BaseCell):
    """
    Cell used for showing ask price and volume.
    """
    # ✅ Best Practice: Calling the superclass's __init__ method ensures proper initialization of inherited attributes.

    def __init__(self, content: Any, data: Any) -> None:
        # ✅ Best Practice: Check for None to handle optional content
        """"""
        super().__init__(content, data)

        # ✅ Best Practice: Convert datetime to local timezone for consistency
        self.setForeground(COLOR_ASK)

# ✅ Best Practice: Use type annotation for clarity

# ✅ Best Practice: Use type annotation for clarity
class PnlCell(BaseCell):
    """
    Cell used for showing pnl data.
    """
    # ✅ Best Practice: Use f-string for readability

    def __init__(self, content: Any, data: Any) -> None:
        # ✅ Best Practice: Class docstring provides a clear description of the class purpose.
        # ✅ Best Practice: Use f-string for readability
        """"""
        super().__init__(content, data)

    # ✅ Best Practice: Use of type hints for function parameters improves code readability and maintainability
    # 🧠 ML Signal: Method call pattern on self object
    def set_content(self, content: Any, data: Any) -> None:
        """
        Cell color is set based on whether pnl is
        positive or negative.
        """
        # ✅ Best Practice: Check for None to avoid processing invalid content
        super().set_content(content, data)

        if str(content).startswith("-"):
            # ⚠️ SAST Risk (Low): Assumes content has a strftime method, which may not be true for all types
            self.setForeground(COLOR_SHORT)
        else:
            # 🧠 ML Signal: Storing data in an instance variable, indicating stateful behavior
            self.setForeground(COLOR_LONG)


# ✅ Best Practice: Class docstring provides a brief description of the class purpose.
class TimeCell(BaseCell):
    """
    Cell used for showing time string from datetime object.
    """
    # 🧠 ML Signal: Usage of QtCore.Qt.AlignmentFlag for setting text alignment
    # ✅ Best Practice: Class docstring provides a brief description of the class purpose

    local_tz = ZoneInfo(get_localzone_name())

    def __init__(self, content: Any, data: Any) -> None:
        """"""
        # ✅ Best Practice: Type annotations for class attributes improve code readability and maintainability
        super().__init__(content, data)

    def set_content(self, content: datetime | None, data: Any) -> None:
        """"""
        # ✅ Best Practice: Using a dictionary for headers allows for flexible key-value storage
        if content is None:
            return
        # ⚠️ SAST Risk (Low): Directly initializing a signal with a potentially mutable event type

        # ✅ Best Practice: Type annotations for attributes improve code readability and maintainability
        content = content.astimezone(self.local_tz)
        timestamp: str = content.strftime("%H:%M:%S")
        # ✅ Best Practice: Type annotations for attributes improve code readability and maintainability

        millisecond: int = int(content.microsecond / 1000)
        # ✅ Best Practice: Type annotations for attributes improve code readability and maintainability
        if millisecond:
            timestamp = f"{timestamp}.{millisecond}"
        # 🧠 ML Signal: Method calls in the constructor can indicate initialization patterns
        else:
            timestamp = f"{timestamp}.000"
        # ✅ Best Practice: Consider adding a docstring to describe the purpose of the method.
        # 🧠 ML Signal: Method calls in the constructor can indicate initialization patterns

        self.setText(timestamp)
        # 🧠 ML Signal: Method calls in the constructor can indicate initialization patterns
        # ✅ Best Practice: Ensure init_table is defined elsewhere in the class.
        self._data = data
# ✅ Best Practice: Ensure init_menu is defined elsewhere in the class.


class DateCell(BaseCell):
    """
    Cell used for showing date string from datetime object.
    # 🧠 ML Signal: Extracting display labels from a dictionary
    """

    # 🧠 ML Signal: Setting UI component labels
    def __init__(self, content: Any, data: Any) -> None:
        """"""
        # 🧠 ML Signal: Customizing UI component visibility
        super().__init__(content, data)

    # 🧠 ML Signal: Configuring UI component editability
    def set_content(self, content: Any, data: Any) -> None:
        """"""
        if content is None:
            # ✅ Best Practice: Type hinting for self.menu improves code readability and maintainability
            # 🧠 ML Signal: Enabling alternating row colors in UI
            return
        # 🧠 ML Signal: Enabling sorting in UI component

        # ✅ Best Practice: Type hinting for resize_action improves code readability and maintainability
        self.setText(content.strftime("%Y-%m-%d"))
        self._data = data
# 🧠 ML Signal: Usage of signal-slot connection pattern in PyQt


# 🧠 ML Signal: Adding actions to a menu, common GUI pattern
class MsgCell(BaseCell):
    """
    Cell used for showing msg data.
    """

    # 🧠 ML Signal: Usage of signal-slot connection pattern in PyQt
    # ✅ Best Practice: Check if self.event_type is not None before proceeding
    def __init__(self, content: str, data: Any) -> None:
        # 🧠 ML Signal: Adding actions to a menu, common GUI pattern
        """"""
        # 🧠 ML Signal: Usage of signal-slot pattern for event handling
        super().__init__(content, data)
        self.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
# 🧠 ML Signal: Registering an event type with a callback function


class BaseMonitor(QtWidgets.QTableWidget):
    """
    Monitor data update.
    """

    # ✅ Best Practice: Use of a clear conditional structure to handle different cases of data processing
    event_type: str = ""
    data_key: str = ""
    sorting: bool = False
    headers: dict = {}
    # 🧠 ML Signal: Use of dynamic attribute access with __getattribute__

    signal: QtCore.Signal = QtCore.Signal(Event)
    # 🧠 ML Signal: Checking for existence of a key in a collection

    def __init__(self, main_engine: MainEngine, event_engine: EventEngine) -> None:
        """"""
        super().__init__()
        # ✅ Best Practice: Re-enable sorting if it was initially enabled

        self.main_engine: MainEngine = main_engine
        self.event_engine: EventEngine = event_engine
        # ✅ Best Practice: Use of self.insertRow to add a new row at a specific position
        self.cells: dict[str, dict] = {}

        self.init_ui()
        # 🧠 ML Signal: Iterating over headers to dynamically process data
        self.load_setting()
        self.register_event()

    # ⚠️ SAST Risk (Low): Use of __getattribute__ can lead to security risks if not controlled
    def init_ui(self) -> None:
        """"""
        # ✅ Best Practice: Use of QtWidgets.QTableWidgetItem to create table items
        self.init_table()
        self.init_menu()
    # ✅ Best Practice: Use of self.setItem to set a cell in the table

    def init_table(self) -> None:
        """
        Initialize table.
        """
        self.setColumnCount(len(self.headers))
        # ⚠️ SAST Risk (Low): Use of __getattribute__ can lead to security risks if not controlled
        # 🧠 ML Signal: Usage of dynamic attribute access with __getattribute__

        # ⚠️ SAST Risk (Low): Potential for AttributeError if data_key is not a valid attribute
        # ✅ Best Practice: Storing cell references for later use
        labels: list = [d["display"] for d in self.headers.values()]
        self.setHorizontalHeaderLabels(labels)
        # 🧠 ML Signal: Accessing dictionary elements using a key

        self.verticalHeader().setVisible(False)
        self.setEditTriggers(self.EditTrigger.NoEditTriggers)
        # 🧠 ML Signal: Usage of dynamic attribute access with __getattribute__
        # ⚠️ SAST Risk (Low): Potential for AttributeError if header is not a valid attribute
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(self.sorting)

    # 🧠 ML Signal: Method call on an object
    # ✅ Best Practice: Use of docstring to describe the method's purpose
    def init_menu(self) -> None:
        """
        Create right click menu.
        """
        self.menu: QtWidgets.QMenu = QtWidgets.QMenu(self)
        # 🧠 ML Signal: Use of QtWidgets for file dialog operations

        resize_action: QtGui.QAction = QtGui.QAction(_("调整列宽"), self)
        resize_action.triggered.connect(self.resize_columns)
        self.menu.addAction(resize_action)

        # ⚠️ SAST Risk (Low): No error handling for file operations
        save_action: QtGui.QAction = QtGui.QAction(_("保存数据"), self)
        save_action.triggered.connect(self.save_csv)
        # 🧠 ML Signal: Use of csv.writer for writing CSV files
        self.menu.addAction(save_action)

    # ✅ Best Practice: Type hinting for headers list
    def register_event(self) -> None:
        """
        Register event handler into event engine.
        """
        # 🧠 ML Signal: Checking for hidden rows in a table
        if self.event_type:
            self.signal.connect(self.process_event)
            # ✅ Best Practice: Type hinting for row_data list
            self.event_engine.register(self.event_type, self.signal.emit)

    def process_event(self, event: Event) -> None:
        """
        Process new data from event and update into table.
        """
        # Disable sorting to prevent unwanted error.
        if self.sorting:
            # 🧠 ML Signal: Usage of context menu event handling in a GUI application
            self.setSortingEnabled(False)
        # ⚠️ SAST Risk (Low): Potential for misuse if menu actions are not properly validated
        # ✅ Best Practice: Method should have a docstring explaining its purpose

        # Update data into table.
        data = event.data
        # 🧠 ML Signal: Usage of QtCore.QSettings indicates interaction with application settings

        # ✅ Best Practice: Type hinting improves code readability and maintainability
        if not self.data_key:
            self.insert_new_row(data)
        # 🧠 ML Signal: Storing UI state information, useful for user behavior analysis
        # 🧠 ML Signal: Use of QtCore.QSettings indicates interaction with application settings storage
        else:
            # ⚠️ SAST Risk (Low): Ensure that the saved state does not include sensitive information
            key: str = data.__getattribute__(self.data_key)
            # 🧠 ML Signal: Checking type of 'column_state' suggests dynamic or uncertain data types

            if key in self.cells:
                self.update_old_row(data)
            # ✅ Best Practice: Class docstring provides a clear description of the class purpose.
            # ⚠️ SAST Risk (Low): Potential risk if 'column_state' is manipulated externally
            else:
                # ✅ Best Practice: Explicitly setting sort indicator improves UI consistency
                self.insert_new_row(data)

        # Enable sorting
        if self.sorting:
            # ✅ Best Practice: Type annotations for class variables improve code readability and maintainability.
            self.setSortingEnabled(True)

    # ✅ Best Practice: Using a dictionary to define headers allows for easy updates and maintenance.
    # ✅ Best Practice: Use of internationalization function _() for display strings.
    def insert_new_row(self, data: Any) -> None:
        """
        Insert a new row at the top of table.
        """
        self.insertRow(0)

        row_cells: dict = {}
        for column, header in enumerate(self.headers.keys()):
            setting: dict = self.headers[header]

            content = data.__getattribute__(header)
            cell: QtWidgets.QTableWidgetItem = setting["cell"](content, data)
            self.setItem(0, column, cell)

            if setting["update"]:
                row_cells[header] = cell
        # ✅ Best Practice: Class docstring provides a brief description of the class purpose.

        if self.data_key:
            key: str = data.__getattribute__(self.data_key)
            self.cells[key] = row_cells

    # ✅ Best Practice: Type annotations for class variables improve code readability and maintainability.
    def update_old_row(self, data: Any) -> None:
        """
        Update an old row in table.
        """
        key: str = data.__getattribute__(self.data_key)
        row_cells = self.cells[key]
        # ✅ Best Practice: Type annotations for class variables improve code readability and maintainability.

        for header, cell in row_cells.items():
            # 🧠 ML Signal: Use of localization function _() indicates internationalization support.
            content = data.__getattribute__(header)
            cell.set_content(content, data)

    # ✅ Best Practice: Use of type annotations for class variables improves code readability and maintainability.
    # 🧠 ML Signal: Use of localization function _() indicates internationalization support.
    def resize_columns(self) -> None:
        """
        Resize all columns according to contents.
        # ✅ Best Practice: Use of type annotations for class variables improves code readability and maintainability.
        # 🧠 ML Signal: Use of dictionary to map trade attributes to display properties and cell types.
        """
        self.horizontalHeader().resizeSections(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)

    def save_csv(self) -> None:
        """
        Save table data into a csv file
        """
        # 🧠 ML Signal: Use of dictionary to map trade attributes to display properties and cell types.
        path, __ = QtWidgets.QFileDialog.getSaveFileName(
            self, _("保存数据"), "", "CSV(*.csv)")
        # 🧠 ML Signal: Use of dictionary to map trade attributes to display properties and cell types.

        if not path:
            # 🧠 ML Signal: Use of dictionary to map trade attributes to display properties and cell types.
            return

        # 🧠 ML Signal: Use of dictionary to map trade attributes to display properties and cell types.
        with open(path, "w") as f:
            writer = csv.writer(f, lineterminator="\n")

            # 🧠 ML Signal: Use of dictionary to map trade attributes to display properties and cell types.
            # ✅ Best Practice: Use of class variables for shared configuration
            headers: list = [d["display"] for d in self.headers.values()]
            # 🧠 ML Signal: Use of dictionary to map trade attributes to display properties and cell types.
            writer.writerow(headers)
            # ✅ Best Practice: Use of class variables for shared configuration

            # 🧠 ML Signal: Use of dictionary to map trade attributes to display properties and cell types.
            # ✅ Best Practice: Use of class variables for shared configuration
            # ✅ Best Practice: Use of dictionary for structured data
            for row in range(self.rowCount()):
                if self.isRowHidden(row):
                    continue

                row_data: list = []
                for column in range(self.columnCount()):
                    item: QtWidgets.QTableWidgetItem | None = self.item(row, column)
                    if item:
                        row_data.append(str(item.text()))
                    else:
                        row_data.append("")
                writer.writerow(row_data)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        """
        Show menu with right click.
        """
        self.menu.popup(QtGui.QCursor.pos())

    # ✅ Best Practice: Call to superclass method ensures proper initialization
    def save_setting(self) -> None:
        """"""
        # 🧠 ML Signal: Use of setToolTip for UI elements
        settings: QtCore.QSettings = QtCore.QSettings(self.__class__.__name__, "custom")
        settings.setValue("column_state", self.horizontalHeader().saveState())
    # 🧠 ML Signal: Signal-slot connection pattern in PyQt
    # ⚠️ SAST Risk (Low): Potential for unintended behavior if cancel_order is not properly defined

    def load_setting(self) -> None:
        """"""
        # 🧠 ML Signal: Usage of type hinting for variable 'order' indicates a pattern for static type checking.
        settings: QtCore.QSettings = QtCore.QSettings(self.__class__.__name__, "custom")
        column_state = settings.value("column_state")
        # 🧠 ML Signal: Usage of type hinting for variable 'req' indicates a pattern for static type checking.

        if isinstance(column_state, QtCore.QByteArray):
            # ⚠️ SAST Risk (Low): Potential risk if 'cancel_order' method does not handle exceptions from 'main_engine'.
            # 🧠 ML Signal: Method call pattern on 'main_engine' could be used to identify common API usage.
            self.horizontalHeader().restoreState(column_state)
            self.horizontalHeader().setSortIndicator(-1, QtCore.Qt.SortOrder.AscendingOrder)

# ✅ Best Practice: Use of class variables for configuration allows easy modification and access.

class TickMonitor(BaseMonitor):
    """
    Monitor for tick data.
    """

    event_type: str = EVENT_TICK
    data_key: str = "vt_symbol"
    sorting: bool = True

    # ✅ Best Practice: Use of descriptive keys and values improves readability and maintainability.
    headers: dict = {
        "symbol": {"display": _("代码"), "cell": BaseCell, "update": False},
        # ✅ Best Practice: Use of descriptive keys and values improves readability and maintainability.
        "exchange": {"display": _("交易所"), "cell": EnumCell, "update": False},
        "name": {"display": _("名称"), "cell": BaseCell, "update": True},
        # ✅ Best Practice: Use of descriptive keys and values improves readability and maintainability.
        "last_price": {"display": _("最新价"), "cell": BaseCell, "update": True},
        # ✅ Best Practice: Use of descriptive keys and values improves readability and maintainability.
        "volume": {"display": _("成交量"), "cell": BaseCell, "update": True},
        "open_price": {"display": _("开盘价"), "cell": BaseCell, "update": True},
        "high_price": {"display": _("最高价"), "cell": BaseCell, "update": True},
        # ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability.
        # ✅ Best Practice: Use of descriptive keys and values improves readability and maintainability.
        "low_price": {"display": _("最低价"), "cell": BaseCell, "update": True},
        "bid_price_1": {"display": _("买1价"), "cell": BidCell, "update": True},
        # ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability.
        # ✅ Best Practice: Use of descriptive keys and values improves readability and maintainability.
        "bid_volume_1": {"display": _("买1量"), "cell": BidCell, "update": True},
        # ✅ Best Practice: Use of type annotations for class attributes improves code readability and maintainability.
        # ✅ Best Practice: Use of descriptive keys and values improves readability and maintainability.
        "ask_price_1": {"display": _("卖1价"), "cell": AskCell, "update": True},
        "ask_volume_1": {"display": _("卖1量"), "cell": AskCell, "update": True},
        "datetime": {"display": _("时间"), "cell": TimeCell, "update": True},
        "gateway_name": {"display": _("接口"), "cell": BaseCell, "update": False},
    }
# 🧠 ML Signal: Use of dictionary to define configuration or settings.


# 🧠 ML Signal: Use of dictionary to define configuration or settings.
class LogMonitor(BaseMonitor):
    """
    Monitor for log data.
    """
    # 🧠 ML Signal: Use of class-level attributes to define constants and configuration
    # 🧠 ML Signal: Use of dictionary to define configuration or settings.

    event_type: str = EVENT_LOG
    # 🧠 ML Signal: Use of class-level attributes to define constants and configuration
    # 🧠 ML Signal: Use of dictionary to define configuration or settings.
    data_key: str = ""
    # 🧠 ML Signal: Use of class-level attributes to define constants and configuration
    # ✅ Best Practice: Use of dictionary to map keys to display properties and cell types
    sorting: bool = False

    headers: dict = {
        "time": {"display": _("时间"), "cell": TimeCell, "update": False},
        "msg": {"display": _("信息"), "cell": MsgCell, "update": False},
        "gateway_name": {"display": _("接口"), "cell": BaseCell, "update": False},
    }


class TradeMonitor(BaseMonitor):
    """
    Monitor for trade data.
    """

    event_type: str = EVENT_TRADE
    data_key: str = ""
    sorting: bool = True

    headers: dict = {
        # ✅ Best Practice: Call to superclass method ensures proper initialization
        "tradeid": {"display": _("成交号"), "cell": BaseCell, "update": False},
        "orderid": {"display": _("委托号"), "cell": BaseCell, "update": False},
        # 🧠 ML Signal: Use of setToolTip with localization function indicates internationalization
        "symbol": {"display": _("代码"), "cell": BaseCell, "update": False},
        "exchange": {"display": _("交易所"), "cell": EnumCell, "update": False},
        # ✅ Best Practice: Docstring provides a clear description of the method's purpose.
        # 🧠 ML Signal: Connecting a signal to a slot is a common pattern in event-driven programming
        "direction": {"display": _("方向"), "cell": DirectionCell, "update": False},
        "offset": {"display": _("开平"), "cell": EnumCell, "update": False},
        "price": {"display": _("价格"), "cell": BaseCell, "update": False},
        "volume": {"display": _("数量"), "cell": BaseCell, "update": False},
        # 🧠 ML Signal: Type hinting for variable 'quote' indicates expected data type.
        "datetime": {"display": _("时间"), "cell": TimeCell, "update": False},
        "gateway_name": {"display": _("接口"), "cell": BaseCell, "update": False},
    # 🧠 ML Signal: Type hinting for variable 'req' indicates expected data type.
    }
# 🧠 ML Signal: Method call pattern on 'self.main_engine' could indicate a common operation.


class OrderMonitor(BaseMonitor):
    """
    Monitor for order data.
    # ✅ Best Practice: Call to super().__init__() ensures proper initialization of the base class.
    """

    # 🧠 ML Signal: Type annotations for attributes can be used to infer expected data types.
    event_type: str = EVENT_ORDER
    data_key: str = "vt_orderid"
    # 🧠 ML Signal: Type annotations for attributes can be used to infer expected data types.
    sorting: bool = True

    # 🧠 ML Signal: Use of f-string for filename generation indicates dynamic file naming.
    headers: dict = {
        "orderid": {"display": _("委托号"), "cell": BaseCell, "update": False},
        # 🧠 ML Signal: Usage of internationalization with _() function
        # 🧠 ML Signal: Use of type annotations for dictionary with complex types.
        "reference": {"display": _("来源"), "cell": BaseCell, "update": False},
        "symbol": {"display": _("代码"), "cell": BaseCell, "update": False},
        # ✅ Best Practice: Separate method for UI initialization improves readability and maintainability.
        # 🧠 ML Signal: Usage pattern of getting default settings
        "exchange": {"display": _("交易所"), "cell": EnumCell, "update": False},
        "type": {"display": _("类型"), "cell": EnumCell, "update": False},
        # 🧠 ML Signal: Usage pattern of loading settings from a JSON file
        "direction": {"display": _("方向"), "cell": DirectionCell, "update": False},
        "offset": {"display": _("开平"), "cell": EnumCell, "update": False},
        # ✅ Best Practice: Explicit type annotation for form layout
        "price": {"display": _("价格"), "cell": BaseCell, "update": False},
        "volume": {"display": _("总数量"), "cell": BaseCell, "update": True},
        "traded": {"display": _("已成交"), "cell": BaseCell, "update": True},
        "status": {"display": _("状态"), "cell": EnumCell, "update": True},
        # ✅ Best Practice: Explicit type annotation for field_type
        "datetime": {"display": _("时间"), "cell": TimeCell, "update": True},
        "gateway_name": {"display": _("接口"), "cell": BaseCell, "update": False},
    }
    # ✅ Best Practice: Explicit type annotation for combo box

    def init_ui(self) -> None:
        """
        Connect signal.
        """
        super().init_ui()
        # ✅ Best Practice: Explicit type annotation for index

        self.setToolTip(_("双击单元格撤单"))
        self.itemDoubleClicked.connect(self.cancel_order)
    # 🧠 ML Signal: Pattern of adding widgets to form layout

    def cancel_order(self, cell: BaseCell) -> None:
        """
        Cancel order if cell double clicked.
        """
        # ✅ Best Practice: Explicit type annotation for line edit
        order: OrderData = cell.get_data()
        req: CancelRequest = order.create_cancel_request()
        self.main_engine.cancel_order(req, order.gateway_name)


# ⚠️ SAST Risk (Low): Potential exposure of sensitive information if not handled properly
class PositionMonitor(BaseMonitor):
    """
    Monitor for position data.
    """
    # ✅ Best Practice: Explicit type annotation for validator

    # 🧠 ML Signal: Iterating over a dictionary to process UI widget data
    event_type: str = EVENT_POSITION
    data_key: str = "vt_positionid"
    sorting: bool = True
    # ✅ Best Practice: Use 'is' for comparing types with singletons like 'list'

    # ✅ Best Practice: Explicit type annotation for button
    headers: dict = {
        "symbol": {"display": _("代码"), "cell": BaseCell, "update": False},
        # 🧠 ML Signal: Pattern of connecting button click to a function
        # 🧠 ML Signal: Retrieving current text from a QComboBox
        "exchange": {"display": _("交易所"), "cell": EnumCell, "update": False},
        "direction": {"display": _("方向"), "cell": DirectionCell, "update": False},
        # 🧠 ML Signal: Pattern of setting layout for a widget
        "volume": {"display": _("数量"), "cell": BaseCell, "update": True},
        "yd_volume": {"display": _("昨仓"), "cell": BaseCell, "update": True},
        "frozen": {"display": _("冻结"), "cell": BaseCell, "update": True},
        # 🧠 ML Signal: Converting text input to a specific field type
        "price": {"display": _("均价"), "cell": BaseCell, "update": True},
        "pnl": {"display": _("盈亏"), "cell": PnlCell, "update": True},
        "gateway_name": {"display": _("接口"), "cell": BaseCell, "update": False},
    # ⚠️ SAST Risk (Low): Defaulting to a type's constructor without handling specific cases
    }
# 🧠 ML Signal: Definition of a class, useful for understanding object-oriented patterns

# 🧠 ML Signal: Storing processed widget data in a dictionary

class AccountMonitor(BaseMonitor):
    """
    Monitor for account data.
    # 🧠 ML Signal: Using a main engine to connect with settings and a gateway name
    # 🧠 ML Signal: Use of type annotations, useful for type inference models
    """
    # ✅ Best Practice: Use of type annotations for class attributes

    # 🧠 ML Signal: Invoking a method to accept or finalize an operation
    event_type: str = EVENT_ACCOUNT
    # 🧠 ML Signal: Usage of type annotations for constructor parameters
    data_key: str = "vt_accountid"
    sorting: bool = True
    # 🧠 ML Signal: Usage of type annotations for instance variables

    headers: dict = {
        # 🧠 ML Signal: Usage of type annotations for instance variables
        "accountid": {"display": _("账号"), "cell": BaseCell, "update": False},
        "balance": {"display": _("余额"), "cell": BaseCell, "update": True},
        # 🧠 ML Signal: Usage of type annotations for instance variables
        "frozen": {"display": _("冻结"), "cell": BaseCell, "update": True},
        "available": {"display": _("可用"), "cell": BaseCell, "update": True},
        # 🧠 ML Signal: Fixed UI width might indicate a specific design choice or constraint
        # ✅ Best Practice: Initializing UI components in a separate method
        "gateway_name": {"display": _("接口"), "cell": BaseCell, "update": False},
    }
# 🧠 ML Signal: Usage of type hints for list of custom objects
# ✅ Best Practice: Registering events in a separate method


# 🧠 ML Signal: Usage of QtWidgets for UI components
class QuoteMonitor(BaseMonitor):
    """
    Monitor for quote data.
    """

    # 🧠 ML Signal: Connecting signal to slot for event handling
    event_type: str = EVENT_QUOTE
    data_key: str = "vt_quoteid"
    sorting: bool = True
    # 🧠 ML Signal: Setting a QLineEdit to read-only

    headers: dict = {
        # 🧠 ML Signal: Adding items to combo box from enum values
        "quoteid": {"display": _("报价号"), "cell": BaseCell, "update": False},
        "reference": {"display": _("来源"), "cell": BaseCell, "update": False},
        "symbol": {"display": _("代码"), "cell": BaseCell, "update": False},
        "exchange": {"display": _("交易所"), "cell": EnumCell, "update": False},
        "bid_offset": {"display": _("买开平"), "cell": EnumCell, "update": False},
        "bid_volume": {"display": _("买量"), "cell": BidCell, "update": False},
        "bid_price": {"display": _("买价"), "cell": BidCell, "update": False},
        "ask_price": {"display": _("卖价"), "cell": AskCell, "update": False},
        "ask_volume": {"display": _("卖量"), "cell": AskCell, "update": False},
        # 🧠 ML Signal: Usage of validators for input fields
        "ask_offset": {"display": _("卖开平"), "cell": EnumCell, "update": False},
        "status": {"display": _("状态"), "cell": EnumCell, "update": True},
        "datetime": {"display": _("时间"), "cell": TimeCell, "update": True},
        "gateway_name": {"display": _("接口"), "cell": BaseCell, "update": False},
    }

    def init_ui(self) -> None:
        """
        Connect signal.
        """
        super().init_ui()
        # 🧠 ML Signal: Setting tooltip for UI element

        self.setToolTip(_("双击单元格撤销报价"))
        self.itemDoubleClicked.connect(self.cancel_quote)
    # 🧠 ML Signal: Connecting signal to slot for event handling

    def cancel_quote(self, cell: BaseCell) -> None:
        """
        Cancel quote if cell double clicked.
        """
        quote: QuoteData = cell.get_data()
        # 🧠 ML Signal: Adding widgets to grid layout
        req: CancelRequest = quote.create_cancel_request()
        self.main_engine.cancel_quote(req, quote.gateway_name)


class ConnectDialog(QtWidgets.QDialog):
    """
    Start connection of a certain gateway.
    """

    def __init__(self, main_engine: MainEngine, gateway_name: str) -> None:
        """"""
        super().__init__()

        self.main_engine: MainEngine = main_engine
        self.gateway_name: str = gateway_name
        self.filename: str = f"connect_{gateway_name.lower()}.json"

        self.widgets: dict[str, tuple[QtWidgets.QWidget, type]] = {}

        self.init_ui()

    def init_ui(self) -> None:
        # 🧠 ML Signal: Usage of color codes for UI elements
        """"""
        self.setWindowTitle(_("连接{}").format(self.gateway_name))
        # 🧠 ML Signal: Creating labels with specific colors

        # Default setting provides field name, field data type and field default value.
        default_setting: dict | None = self.main_engine.get_default_setting(self.gateway_name)

        # Saved setting provides field data used last time.
        loaded_setting: dict = load_json(self.filename)

        # Initialize line edits and form layout based on setting.
        form: QtWidgets.QFormLayout = QtWidgets.QFormLayout()

        if default_setting:
            for field_name, field_value in default_setting.items():
                field_type: type = type(field_value)

                if field_type is list:
                    combo: QtWidgets.QComboBox = QtWidgets.QComboBox()
                    combo.addItems(field_value)

                    if field_name in loaded_setting:
                        saved_value = loaded_setting[field_name]
                        ix: int = combo.findText(saved_value)
                        combo.setCurrentIndex(ix)

                    form.addRow(f"{field_name} <{field_type.__name__}>", combo)
                    self.widgets[field_name] = (combo, field_type)
                else:
                    line: QtWidgets.QLineEdit = QtWidgets.QLineEdit(str(field_value))

                    if field_name in loaded_setting:
                        saved_value = loaded_setting[field_name]
                        line.setText(str(saved_value))

                    if _("密码") in field_name:
                        line.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)

                    # 🧠 ML Signal: Adding rows to form layout
                    if field_type is int:
                        validator: QtGui.QIntValidator = QtGui.QIntValidator()
                        line.setValidator(validator)

                    form.addRow(f"{field_name} <{field_type.__name__}>", line)
                    self.widgets[field_name] = (line, field_type)

        button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("连接"))
        # ✅ Best Practice: Initialize the QLabel object outside of any conditionals for clarity.
        button.clicked.connect(self.connect_gateway)
        form.addRow(button)

        # ⚠️ SAST Risk (Low): Potential for code injection if 'color' is derived from untrusted input.
        self.setLayout(form)

    # 🧠 ML Signal: Combining layouts into a vertical box layout
    # ⚠️ SAST Risk (Low): Ensure 'alignment' is a valid Qt.AlignmentFlag to prevent unexpected behavior.
    def connect_gateway(self) -> None:
        """
        Get setting value from line edits and connect the gateway.
        # 🧠 ML Signal: Setting the main layout for the UI
        """
        # 🧠 ML Signal: Registers an event with a callback, indicating usage of an event-driven architecture
        setting: dict = {}

        for field_name, tp in self.widgets.items():
            # ✅ Best Practice: Early return pattern improves readability by reducing nesting.
            widget, field_type = tp
            if field_type is list:
                combo: QtWidgets.QComboBox = cast(QtWidgets.QComboBox, widget)
                field_value = str(combo.currentText())
            # 🧠 ML Signal: Usage of setText method on UI elements indicates UI update pattern.
            else:
                line: QtWidgets.QLineEdit = cast(QtWidgets.QLineEdit, widget)
                # 🧠 ML Signal: Usage of setText method on UI elements indicates UI update pattern.
                try:
                    field_value = field_type(line.text())
                # 🧠 ML Signal: Usage of setText method on UI elements indicates UI update pattern.
                except ValueError:
                    field_value = field_type()
            # 🧠 ML Signal: Usage of setText method on UI elements indicates UI update pattern.
            setting[field_name] = field_value

        # 🧠 ML Signal: Usage of setText method on UI elements indicates UI update pattern.
        save_json(self.filename, setting)

        self.main_engine.connect(setting, self.gateway_name)
        self.accept()
# 🧠 ML Signal: Usage of setText method on UI elements indicates UI update pattern.


class TradingWidget(QtWidgets.QWidget):
    """
    General manual trading widget.
    # 🧠 ML Signal: Usage of setText method on UI elements indicates UI update pattern.
    """

    # 🧠 ML Signal: Usage of setText method on UI elements indicates UI update pattern.
    signal_tick: QtCore.Signal = QtCore.Signal(Event)

    # 🧠 ML Signal: Usage of setText method on UI elements indicates UI update pattern.
    def __init__(self, main_engine: MainEngine, event_engine: EventEngine) -> None:
        """"""
        # 🧠 ML Signal: Usage of setText method on UI elements indicates UI update pattern.
        super().__init__()

        # 🧠 ML Signal: Usage of setText method on UI elements indicates UI update pattern.
        self.main_engine: MainEngine = main_engine
        self.event_engine: EventEngine = event_engine
        # 🧠 ML Signal: Usage of setText method on UI elements indicates UI update pattern.

        self.vt_symbol: str = ""
        self.price_digits: int = 0
        # 🧠 ML Signal: Usage of setText method on UI elements indicates UI update pattern.

        # 🧠 ML Signal: Usage of setText method on UI elements indicates UI update pattern.
        # 🧠 ML Signal: Checks for empty input, a common pattern for input validation
        self.init_ui()
        self.register_event()
    # 🧠 ML Signal: Usage of setText method on UI elements indicates UI update pattern.

    def init_ui(self) -> None:
        # 🧠 ML Signal: Usage of setText method on UI elements indicates UI update pattern.
        # 🧠 ML Signal: Constructs a unique identifier using multiple fields
        """"""
        self.setFixedWidth(300)
        # 🧠 ML Signal: Usage of setText method on UI elements indicates UI update pattern.
        # 🧠 ML Signal: Compares current and new state to avoid unnecessary updates

        # Trading function area
        # 🧠 ML Signal: Usage of setText method on UI elements indicates UI update pattern.
        exchanges: list[Exchange] = self.main_engine.get_all_exchanges()
        self.exchange_combo: QtWidgets.QComboBox = QtWidgets.QComboBox()
        # 🧠 ML Signal: Usage of setText method on UI elements indicates UI update pattern.
        # ⚠️ SAST Risk (Low): Potential NoneType if get_contract returns None
        self.exchange_combo.addItems([exchange.value for exchange in exchanges])

        # 🧠 ML Signal: Usage of setText method on UI elements indicates UI update pattern.
        # 🧠 ML Signal: Checks for None, a common pattern for handling optional values
        self.symbol_line: QtWidgets.QLineEdit = QtWidgets.QLineEdit()
        self.symbol_line.returnPressed.connect(self.set_vt_symbol)
        # 🧠 ML Signal: Usage of setText method on UI elements indicates UI update pattern.

        self.name_line: QtWidgets.QLineEdit = QtWidgets.QLineEdit()
        self.name_line.setReadOnly(True)
        # 🧠 ML Signal: Usage of setText method on UI elements indicates UI update pattern.

        self.direction_combo: QtWidgets.QComboBox = QtWidgets.QComboBox()
        # ⚠️ SAST Risk (Low): Assumes gateway_name is always found in the combo box
        self.direction_combo.addItems(
            [Direction.LONG.value, Direction.SHORT.value])

        self.offset_combo: QtWidgets.QComboBox = QtWidgets.QComboBox()
        # ✅ Best Practice: Encapsulates logic for extracting digits
        self.offset_combo.addItems([offset.value for offset in Offset])

        self.order_type_combo: QtWidgets.QComboBox = QtWidgets.QComboBox()
        self.order_type_combo.addItems(
            [order_type.value for order_type in OrderType])
        # 🧠 ML Signal: Uses a request object to encapsulate parameters for an operation
        # ✅ Best Practice: Use a loop or list to manage repetitive tasks for better maintainability.

        double_validator: QtGui.QDoubleValidator = QtGui.QDoubleValidator()
        double_validator.setBottom(0)
        # ⚠️ SAST Risk (Low): Assumes gateway_name is valid and correctly set

        self.price_line: QtWidgets.QLineEdit = QtWidgets.QLineEdit()
        self.price_line.setValidator(double_validator)

        self.volume_line: QtWidgets.QLineEdit = QtWidgets.QLineEdit()
        self.volume_line.setValidator(double_validator)

        self.gateway_combo: QtWidgets.QComboBox = QtWidgets.QComboBox()
        self.gateway_combo.addItems(self.main_engine.get_all_gateway_names())

        self.price_check: QtWidgets.QCheckBox = QtWidgets.QCheckBox()
        self.price_check.setToolTip(_("设置价格随行情更新"))

        send_button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("委托"))
        send_button.clicked.connect(self.send_order)

        cancel_button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("全撤"))
        cancel_button.clicked.connect(self.cancel_all)

        grid: QtWidgets.QGridLayout = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel(_("交易所")), 0, 0)
        grid.addWidget(QtWidgets.QLabel(_("代码")), 1, 0)
        grid.addWidget(QtWidgets.QLabel(_("名称")), 2, 0)
        grid.addWidget(QtWidgets.QLabel(_("方向")), 3, 0)
        # ⚠️ SAST Risk (Low): Lack of input validation for 'symbol' could lead to unexpected behavior or errors.
        grid.addWidget(QtWidgets.QLabel(_("开平")), 4, 0)
        grid.addWidget(QtWidgets.QLabel(_("类型")), 5, 0)
        grid.addWidget(QtWidgets.QLabel(_("价格")), 6, 0)
        grid.addWidget(QtWidgets.QLabel(_("数量")), 7, 0)
        grid.addWidget(QtWidgets.QLabel(_("接口")), 8, 0)
        # ⚠️ SAST Risk (Low): Lack of input validation for 'volume_text' could lead to unexpected behavior or errors.
        grid.addWidget(self.exchange_combo, 0, 1, 1, 2)
        grid.addWidget(self.symbol_line, 1, 1, 1, 2)
        grid.addWidget(self.name_line, 2, 1, 1, 2)
        grid.addWidget(self.direction_combo, 3, 1, 1, 2)
        grid.addWidget(self.offset_combo, 4, 1, 1, 2)
        # ⚠️ SAST Risk (Low): Potential ValueError if 'volume_text' is not a valid float.
        grid.addWidget(self.order_type_combo, 5, 1, 1, 2)
        grid.addWidget(self.price_line, 6, 1, 1, 1)
        # ⚠️ SAST Risk (Low): Lack of input validation for 'price_text' could lead to unexpected behavior or errors.
        grid.addWidget(self.price_check, 6, 2, 1, 1)
        # ⚠️ SAST Risk (Low): Potential ValueError if 'price_text' is not a valid float.
        grid.addWidget(self.volume_line, 7, 1, 1, 2)
        grid.addWidget(self.gateway_combo, 8, 1, 1, 2)
        grid.addWidget(send_button, 9, 0, 1, 3)
        grid.addWidget(cancel_button, 10, 0, 1, 3)

        # Market depth display area
        bid_color: str = "rgb(255,174,201)"
        ask_color: str = "rgb(160,255,160)"

        self.bp1_label: QtWidgets.QLabel = self.create_label(bid_color)
        # ⚠️ SAST Risk (Low): Lack of validation for 'exchange' could lead to invalid exchange values.
        self.bp2_label: QtWidgets.QLabel = self.create_label(bid_color)
        self.bp3_label: QtWidgets.QLabel = self.create_label(bid_color)
        # ⚠️ SAST Risk (Low): Lack of validation for 'type' could lead to invalid order type values.
        self.bp4_label: QtWidgets.QLabel = self.create_label(bid_color)
        self.bp5_label: QtWidgets.QLabel = self.create_label(bid_color)

        self.bv1_label: QtWidgets.QLabel = self.create_label(
            # ⚠️ SAST Risk (Low): Lack of validation for 'offset' could lead to invalid offset values.
            # 🧠 ML Signal: Usage of a method to retrieve all active orders
            bid_color, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self.bv2_label: QtWidgets.QLabel = self.create_label(
            bid_color, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        # 🧠 ML Signal: Pattern of creating a cancel request from an order
        self.bv3_label: QtWidgets.QLabel = self.create_label(
            # ⚠️ SAST Risk (Low): Lack of validation for 'gateway_name' could lead to invalid gateway values.
            bid_color, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        # 🧠 ML Signal: Pattern of cancelling an order using a request and gateway name
        self.bv4_label: QtWidgets.QLabel = self.create_label(
            # 🧠 ML Signal: Method accessing data from a cell object, indicating a pattern of data extraction
            # 🧠 ML Signal: Usage of 'send_order' method could indicate user behavior patterns in trading applications.
            bid_color, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self.bv5_label: QtWidgets.QLabel = self.create_label(
            # 🧠 ML Signal: Setting text in a UI component, indicating a pattern of UI updates
            bid_color, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

        self.ap1_label: QtWidgets.QLabel = self.create_label(ask_color)
        # 🧠 ML Signal: Setting current index in a combo box, indicating a pattern of UI interaction
        self.ap2_label: QtWidgets.QLabel = self.create_label(ask_color)
        self.ap3_label: QtWidgets.QLabel = self.create_label(ask_color)
        self.ap4_label: QtWidgets.QLabel = self.create_label(ask_color)
        # ✅ Best Practice: Encapsulation of functionality in a separate method
        self.ap5_label: QtWidgets.QLabel = self.create_label(ask_color)

        self.av1_label: QtWidgets.QLabel = self.create_label(
            ask_color, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self.av2_label: QtWidgets.QLabel = self.create_label(
            ask_color, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self.av3_label: QtWidgets.QLabel = self.create_label(
            ask_color, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self.av4_label: QtWidgets.QLabel = self.create_label(
            ask_color, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self.av5_label: QtWidgets.QLabel = self.create_label(
            # 🧠 ML Signal: Setting current index in a combo box, indicating a pattern of UI interaction
            ask_color, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

        self.lp_label: QtWidgets.QLabel = self.create_label()
        self.return_label: QtWidgets.QLabel = self.create_label(alignment=QtCore.Qt.AlignmentFlag.AlignRight)

        # ✅ Best Practice: Class docstring provides a brief description of the class purpose.
        # 🧠 ML Signal: Setting current index in a combo box, indicating a pattern of UI interaction
        form: QtWidgets.QFormLayout = QtWidgets.QFormLayout()
        form.addRow(self.ap5_label, self.av5_label)
        form.addRow(self.ap4_label, self.av4_label)
        form.addRow(self.ap3_label, self.av3_label)
        # 🧠 ML Signal: Setting text in a UI component, indicating a pattern of UI updates
        form.addRow(self.ap2_label, self.av2_label)
        form.addRow(self.ap1_label, self.av1_label)
        form.addRow(self.lp_label, self.return_label)
        # ✅ Best Practice: Call to superclass method ensures proper inheritance behavior
        form.addRow(self.bp1_label, self.bv1_label)
        form.addRow(self.bp2_label, self.bv2_label)
        # 🧠 ML Signal: Type hinting for 'order' can be used to infer data structure usage
        form.addRow(self.bp3_label, self.bv3_label)
        form.addRow(self.bp4_label, self.bv4_label)
        # 🧠 ML Signal: Accessing dictionary with dynamic keys indicates flexible data handling
        form.addRow(self.bp5_label, self.bv5_label)
        # 🧠 ML Signal: Dynamic row calculation based on data attributes

        # Overall layout
        vbox: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        # 🧠 ML Signal: Conditional logic based on object state
        vbox.addLayout(grid)
        # ✅ Best Practice: Explicit method call for showing UI component
        vbox.addLayout(form)
        self.setLayout(vbox)

    # ✅ Best Practice: Explicit method call for hiding UI component
    # ✅ Best Practice: Use of type hints for class attributes improves code readability and maintainability.
    # 🧠 ML Signal: Use of internationalization/localization with the _() function.
    def create_label(
        self,
        color: str = "",
        alignment: int = QtCore.Qt.AlignmentFlag.AlignLeft
    ) -> QtWidgets.QLabel:
        """
        Create label with certain font color.
        """
        label: QtWidgets.QLabel = QtWidgets.QLabel()
        if color:
            label.setStyleSheet(f"color:{color}")
        label.setAlignment(Qt.AlignmentFlag(alignment))
        return label

    def register_event(self) -> None:
        """"""
        # ✅ Best Practice: Call to super() in __init__ ensures proper initialization of the base class
        self.signal_tick.connect(self.process_tick_event)
        self.event_engine.register(EVENT_TICK, self.signal_tick.emit)
    # 🧠 ML Signal: Type annotations for attributes can be used to infer expected data types

    def process_tick_event(self, event: Event) -> None:
        # 🧠 ML Signal: Type annotations for attributes can be used to infer expected data types
        """"""
        tick: TickData = event.data
        # ✅ Best Practice: Separate method for UI initialization improves readability and maintainability
        # 🧠 ML Signal: Setting window title with localization support
        if tick.vt_symbol != self.vt_symbol:
            return
        # ✅ Best Practice: Setting a fixed window size for consistent UI layout

        price_digits: int = self.price_digits
        # ✅ Best Practice: Type hinting for better code readability and maintenance

        self.lp_label.setText(f"{tick.last_price:.{price_digits}f}")
        # 🧠 ML Signal: Using placeholder text with localization support
        self.bp1_label.setText(f"{tick.bid_price_1:.{price_digits}f}")
        self.bv1_label.setText(str(tick.bid_volume_1))
        # ✅ Best Practice: Type hinting for better code readability and maintenance
        self.ap1_label.setText(f"{tick.ask_price_1:.{price_digits}f}")
        self.av1_label.setText(str(tick.ask_volume_1))
        # 🧠 ML Signal: Connecting button click to a method

        if tick.pre_close:
            # ✅ Best Practice: Using list comprehension for better readability
            r: float = (tick.last_price / tick.pre_close - 1) * 100
            self.return_label.setText(f"{r:.2f}%")

        # ✅ Best Practice: Using f-string for better readability
        if tick.bid_price_2:
            self.bp2_label.setText(f"{tick.bid_price_2:.{price_digits}f}")
            self.bv2_label.setText(str(tick.bid_volume_2))
            # ✅ Best Practice: Type hinting for better code readability and maintenance
            self.ap2_label.setText(f"{tick.ask_price_2:.{price_digits}f}")
            self.av2_label.setText(str(tick.ask_volume_2))
            # 🧠 ML Signal: Setting the number of columns based on headers

            self.bp3_label.setText(f"{tick.bid_price_3:.{price_digits}f}")
            # 🧠 ML Signal: Setting table headers with localization support
            self.bv3_label.setText(str(tick.bid_volume_3))
            self.ap3_label.setText(f"{tick.ask_price_3:.{price_digits}f}")
            # ✅ Best Practice: Hiding vertical headers for cleaner UI
            self.av3_label.setText(str(tick.ask_volume_3))

            self.bp4_label.setText(f"{tick.bid_price_4:.{price_digits}f}")
            # ⚠️ SAST Risk (Low): Disabling edit triggers to prevent unwanted data modification
            # ✅ Best Practice: Explicitly typing the variable for clarity and maintainability
            self.bv4_label.setText(str(tick.bid_volume_4))
            # ✅ Best Practice: Using alternating row colors for better readability
            self.ap4_label.setText(f"{tick.ask_price_4:.{price_digits}f}")
            # 🧠 ML Signal: Usage of a method to retrieve all contracts
            self.av4_label.setText(str(tick.ask_volume_4))
            # ✅ Best Practice: Using layout managers for better UI organization
            # 🧠 ML Signal: Filtering pattern based on user input

            self.bp5_label.setText(f"{tick.bid_price_5:.{price_digits}f}")
            self.bv5_label.setText(str(tick.bid_volume_5))
            self.ap5_label.setText(f"{tick.ask_price_5:.{price_digits}f}")
            # ✅ Best Practice: Using layout managers for better UI organization
            self.av5_label.setText(str(tick.ask_volume_5))

        if self.price_check.isChecked():
            self.price_line.setText(f"{tick.last_price:.{price_digits}f}")
    # ✅ Best Practice: Setting the main layout for the widget
    # ✅ Best Practice: Clearing table contents before populating it

    def set_vt_symbol(self) -> None:
        """
        Set the tick depth data to monitor by vt_symbol.
        """
        symbol: str = str(self.symbol_line.text())
        # ✅ Best Practice: Using getattr to dynamically access attributes
        if not symbol:
            return

        # Generate vt_symbol from symbol and exchange
        exchange_value: str = str(self.exchange_combo.currentText())
        # ✅ Best Practice: Using isinstance to handle different data types
        vt_symbol: str = f"{symbol}.{exchange_value}"

        if vt_symbol == self.vt_symbol:
            return
        self.vt_symbol = vt_symbol

        # ✅ Best Practice: Class docstring provides a brief description of the class purpose.
        # Update name line widget and clear all labels
        # 🧠 ML Signal: Populating a table with dynamic data
        contract: ContractData | None = self.main_engine.get_contract(vt_symbol)
        # ✅ Best Practice: Call to superclass initializer ensures proper initialization of the base class.
        if not contract:
            # ✅ Best Practice: Adjusting column sizes to fit content
            self.name_line.setText("")
            # ✅ Best Practice: Type annotations for attributes improve code readability and maintainability.
            gateway_name: str = self.gateway_combo.currentText()
        else:
            # ✅ Best Practice: Type annotations for attributes improve code readability and maintainability.
            self.name_line.setText(contract.name)
            gateway_name = contract.gateway_name
            # 🧠 ML Signal: Method call during initialization indicates a setup or configuration pattern.
            # ✅ Best Practice: Consider adding a docstring to describe the method's purpose.

            # Update gateway combo box.
            # ✅ Best Practice: Importing within a function is generally discouraged; consider moving the import to the top of the file.
            ix: int = self.gateway_combo.findText(gateway_name)
            self.gateway_combo.setCurrentIndex(ix)

            # Update price digits
            self.price_digits = get_digits(contract.pricetick)

        self.clear_label_text()
        self.volume_line.setText("")
        self.price_line.setText("")

        # Subscribe tick data
        req: SubscribeRequest = SubscribeRequest(
            symbol=symbol, exchange=Exchange(exchange_value)
        )
        # 🧠 ML Signal: Usage of f-strings for string formatting.

        self.main_engine.subscribe(req, gateway_name)

    def clear_label_text(self) -> None:
        """
        Clear text on all labels.
        """
        self.lp_label.setText("")
        self.return_label.setText("")

        self.bv1_label.setText("")
        # ✅ Best Practice: Use of type hints for the dictionary improves code readability and maintainability.
        self.bv2_label.setText("")
        self.bv3_label.setText("")
        # 🧠 ML Signal: Calling an initialization method is a common pattern in class constructors.
        self.bv4_label.setText("")
        self.bv5_label.setText("")
        # 🧠 ML Signal: Use of setWindowTitle with internationalization

        self.av1_label.setText("")
        # ✅ Best Practice: Setting a minimum width for the window
        self.av2_label.setText("")
        self.av3_label.setText("")
        # ✅ Best Practice: Using type annotations for better code readability
        self.av4_label.setText("")
        self.av5_label.setText("")
        # ⚠️ SAST Risk (Low): Potential risk if load_json is not properly handling JSON parsing

        self.bp1_label.setText("")
        # ✅ Best Practice: Using type annotations for better code readability
        self.bp2_label.setText("")
        self.bp3_label.setText("")
        # 🧠 ML Signal: Iterating over settings to dynamically create UI elements
        self.bp4_label.setText("")
        self.bp5_label.setText("")
        # ✅ Best Practice: Using type annotations for better code readability

        self.ap1_label.setText("")
        # ✅ Best Practice: Using type annotations for better code readability
        self.ap2_label.setText("")
        self.ap3_label.setText("")
        self.ap4_label.setText("")
        # 🧠 ML Signal: Storing widget references for later use
        self.ap5_label.setText("")

    # ✅ Best Practice: Using type annotations for better code readability
    def send_order(self) -> None:
        """
        Send new order manually.
        """
        symbol: str = str(self.symbol_line.text())
        if not symbol:
            # ✅ Best Practice: Using type annotations for better code readability
            QtWidgets.QMessageBox.critical(self, _("委托失败"), _("请输入合约代码"))
            # 🧠 ML Signal: Iterating over a dictionary to process items
            return
        # ✅ Best Practice: Using type annotations for better code readability

        volume_text: str = str(self.volume_line.text())
        # 🧠 ML Signal: Retrieving text from a widget
        if not volume_text:
            QtWidgets.QMessageBox.critical(self, _("委托失败"), _("请输入委托数量"))
            # ✅ Best Practice: Using type annotations for better code readability
            return
        volume: float = float(volume_text)

        price_text: str = str(self.price_line.text())
        if not price_text:
            price: float = 0
        # ⚠️ SAST Risk (Medium): Potential risk of type conversion without validation
        # 🧠 ML Signal: Storing processed values in a dictionary
        else:
            price = float(price_text)

        req: OrderRequest = OrderRequest(
            symbol=symbol,
            exchange=Exchange(str(self.exchange_combo.currentText())),
            # ✅ Best Practice: Informing users about the need to restart for changes to take effect
            direction=Direction(str(self.direction_combo.currentText())),
            # ⚠️ SAST Risk (Medium): Saving settings to a file without validation or error handling
            # 🧠 ML Signal: Using a method to close or accept a dialog
            type=OrderType(str(self.order_type_combo.currentText())),
            volume=volume,
            price=price,
            offset=Offset(str(self.offset_combo.currentText())),
            reference="ManualTrading"
        )

        gateway_name: str = str(self.gateway_combo.currentText())

        self.main_engine.send_order(req, gateway_name)

    def cancel_all(self) -> None:
        """
        Cancel all active orders.
        """
        order_list: list[OrderData] = self.main_engine.get_all_active_orders()
        for order in order_list:
            req: CancelRequest = order.create_cancel_request()
            self.main_engine.cancel_order(req, order.gateway_name)

    def update_with_cell(self, cell: BaseCell) -> None:
        """"""
        data = cell.get_data()

        self.symbol_line.setText(data.symbol)
        self.exchange_combo.setCurrentIndex(
            self.exchange_combo.findText(data.exchange.value)
        )

        self.set_vt_symbol()

        if isinstance(data, PositionData):
            if data.direction == Direction.SHORT:
                direction: Direction = Direction.LONG
            elif data.direction == Direction.LONG:
                direction = Direction.SHORT
            else:       # Net position mode
                if data.volume > 0:
                    direction = Direction.SHORT
                else:
                    direction = Direction.LONG

            self.direction_combo.setCurrentIndex(
                self.direction_combo.findText(direction.value)
            )
            self.offset_combo.setCurrentIndex(
                self.offset_combo.findText(Offset.CLOSE.value)
            )
            self.volume_line.setText(str(abs(data.volume)))


class ActiveOrderMonitor(OrderMonitor):
    """
    Monitor which shows active order only.
    """

    def process_event(self, event: Event) -> None:
        """
        Hides the row if order is not active.
        """
        super().process_event(event)

        order: OrderData = event.data
        row_cells: dict = self.cells[order.vt_orderid]
        row: int = self.row(row_cells["volume"])

        if order.is_active():
            self.showRow(row)
        else:
            self.hideRow(row)


class ContractManager(QtWidgets.QWidget):
    """
    Query contract data available to trade in system.
    """

    headers: dict[str, str] = {
        "vt_symbol": _("本地代码"),
        "symbol": _("代码"),
        "exchange": _("交易所"),
        "name": _("名称"),
        "product": _("合约分类"),
        "size": _("合约乘数"),
        "pricetick": _("价格跳动"),
        "min_volume": _("最小委托量"),
        "option_portfolio": _("期权产品"),
        "option_expiry": _("期权到期日"),
        "option_strike": _("期权行权价"),
        "option_type": _("期权类型"),
        "gateway_name": _("交易接口"),
    }

    def __init__(self, main_engine: MainEngine, event_engine: EventEngine) -> None:
        super().__init__()

        self.main_engine: MainEngine = main_engine
        self.event_engine: EventEngine = event_engine

        self.init_ui()

    def init_ui(self) -> None:
        """"""
        self.setWindowTitle(_("合约查询"))
        self.resize(1000, 600)

        self.filter_line: QtWidgets.QLineEdit = QtWidgets.QLineEdit()
        self.filter_line.setPlaceholderText(_("输入合约代码或者交易所，留空则查询所有合约"))

        self.button_show: QtWidgets.QPushButton = QtWidgets.QPushButton(_("查询"))
        self.button_show.clicked.connect(self.show_contracts)

        labels: list = []
        for name, display in self.headers.items():
            label: str = f"{display}\n{name}"
            labels.append(label)

        self.contract_table: QtWidgets.QTableWidget = QtWidgets.QTableWidget()
        self.contract_table.setColumnCount(len(self.headers))
        self.contract_table.setHorizontalHeaderLabels(labels)
        self.contract_table.verticalHeader().setVisible(False)
        self.contract_table.setEditTriggers(self.contract_table.EditTrigger.NoEditTriggers)
        self.contract_table.setAlternatingRowColors(True)

        hbox: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.filter_line)
        hbox.addWidget(self.button_show)

        vbox: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(self.contract_table)

        self.setLayout(vbox)

    def show_contracts(self) -> None:
        """
        Show contracts by symbol
        """
        flt: str = str(self.filter_line.text())

        all_contracts: list[ContractData] = self.main_engine.get_all_contracts()
        if flt:
            contracts: list[ContractData] = [
                contract for contract in all_contracts if flt in contract.vt_symbol
            ]
        else:
            contracts = all_contracts

        self.contract_table.clearContents()
        self.contract_table.setRowCount(len(contracts))

        for row, contract in enumerate(contracts):
            for column, name in enumerate(self.headers.keys()):
                value: Any = getattr(contract, name)

                if value in {None, 0}:
                    value = ""

                cell: BaseCell
                if isinstance(value, Enum):
                    cell = EnumCell(value, contract)
                elif isinstance(value, datetime):
                    cell = DateCell(value, contract)
                else:
                    cell = BaseCell(value, contract)
                self.contract_table.setItem(row, column, cell)

        self.contract_table.resizeColumnsToContents()


class AboutDialog(QtWidgets.QDialog):
    """
    Information about the trading platform.
    """

    def __init__(self, main_engine: MainEngine, event_engine: EventEngine) -> None:
        """"""
        super().__init__()

        self.main_engine: MainEngine = main_engine
        self.event_engine: EventEngine = event_engine

        self.init_ui()

    def init_ui(self) -> None:
        """"""
        self.setWindowTitle(_("关于VeighNa Trader"))

        from ... import __version__ as vnpy_version

        text: str = f"""
            By Traders, For Traders.

            Created by VeighNa Technology


            License：MIT
            Website：www.vnpy.com
            Github：www.github.com/vnpy/vnpy


            VeighNa - {vnpy_version}
            Python - {platform.python_version()}
            PySide6 - {metadata.version("pyside6")}
            NumPy - {metadata.version("numpy")}
            pandas - {metadata.version("pandas")}
            """

        label: QtWidgets.QLabel = QtWidgets.QLabel()
        label.setText(text)
        label.setMinimumWidth(500)

        vbox: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        vbox.addWidget(label)
        self.setLayout(vbox)


class GlobalDialog(QtWidgets.QDialog):
    """
    Start connection of a certain gateway.
    """

    def __init__(self) -> None:
        """"""
        super().__init__()

        self.widgets: dict[str, tuple[QtWidgets.QLineEdit, type]] = {}

        self.init_ui()

    def init_ui(self) -> None:
        """"""
        self.setWindowTitle(_("全局配置"))
        self.setMinimumWidth(800)

        settings: dict = copy(SETTINGS)
        settings.update(load_json(SETTING_FILENAME))

        # Initialize line edits and form layout based on setting.
        form: QtWidgets.QFormLayout = QtWidgets.QFormLayout()

        for field_name, field_value in settings.items():
            field_type: type = type(field_value)
            widget: QtWidgets.QLineEdit = QtWidgets.QLineEdit(str(field_value))

            form.addRow(f"{field_name} <{field_type.__name__}>", widget)
            self.widgets[field_name] = (widget, field_type)

        button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("确定"))
        button.clicked.connect(self.update_setting)
        form.addRow(button)

        scroll_widget: QtWidgets.QWidget = QtWidgets.QWidget()
        scroll_widget.setLayout(form)

        scroll_area: QtWidgets.QScrollArea = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_widget)

        vbox: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        vbox.addWidget(scroll_area)
        self.setLayout(vbox)

    def update_setting(self) -> None:
        """
        Get setting value from line edits and update global setting file.
        """
        settings: dict = {}
        for field_name, tp in self.widgets.items():
            widget, field_type = tp
            value_text: str = widget.text()

            if field_type is bool:
                if value_text == "True":
                    field_value: bool = True
                else:
                    field_value = False
            else:
                field_value = field_type(value_text)

            settings[field_name] = field_value

        QtWidgets.QMessageBox.information(
            self,
            _("注意"),
            _("全局配置的修改需要重启后才会生效！"),
            QtWidgets.QMessageBox.StandardButton.Ok
        )

        save_json(SETTING_FILENAME, settings)
        self.accept()