"""
Implements main window of the trading platform.
"""
# ✅ Best Practice: Grouping standard library imports at the top improves readability and maintainability.

from types import ModuleType
# ✅ Best Practice: Grouping standard library imports at the top improves readability and maintainability.
import webbrowser
from functools import partial
# ✅ Best Practice: Grouping standard library imports at the top improves readability and maintainability.
from importlib import import_module
from typing import TypeVar
# ✅ Best Practice: Grouping standard library imports at the top improves readability and maintainability.
from collections.abc import Callable

# ✅ Best Practice: Grouping standard library imports at the top improves readability and maintainability.
import vnpy
# ✅ Best Practice: Grouping standard library imports at the top improves readability and maintainability.
# ✅ Best Practice: Grouping local application/library specific imports separately improves readability and maintainability.
from vnpy.event import EventEngine

from .qt import QtCore, QtGui, QtWidgets
from .widget import (
    BaseMonitor,
    TickMonitor,
    OrderMonitor,
    TradeMonitor,
    PositionMonitor,
    AccountMonitor,
    LogMonitor,
    ActiveOrderMonitor,
    ConnectDialog,
    ContractManager,
    TradingWidget,
    # ✅ Best Practice: Grouping local application/library specific imports separately improves readability and maintainability.
    AboutDialog,
    GlobalDialog
)
from ..engine import MainEngine, BaseApp
# 🧠 ML Signal: Class definition for a GUI application, indicating a pattern for UI-based applications
from ..utility import get_icon_path, TRADER_DIR
from ..locale import _


WidgetType = TypeVar("WidgetType", bound="QtWidgets.QWidget")

# ✅ Best Practice: Grouping local application/library specific imports separately improves readability and maintainability.

class MainWindow(QtWidgets.QMainWindow):
    """
    Main window of the trading platform.
    # ✅ Best Practice: Grouping local application/library specific imports separately improves readability and maintainability.
    """
    # ✅ Best Practice: Use of type hints for dictionary with specific key-value types

    # 🧠 ML Signal: Use of TypeVar indicates generic programming, which can be a signal for type inference models.
    def __init__(self, main_engine: MainEngine, event_engine: EventEngine) -> None:
        # ✅ Best Practice: Use of type hints for dictionary with specific key-value types
        """"""
        super().__init__()
        # 🧠 ML Signal: Method names starting with 'init_' suggest initialization patterns
        # 🧠 ML Signal: Initialization of UI components in a dedicated method

        self.main_engine: MainEngine = main_engine
        # 🧠 ML Signal: UI setup methods often follow a specific sequence
        self.event_engine: EventEngine = event_engine

        # 🧠 ML Signal: UI setup methods often follow a specific sequence
        self.window_title: str = _("VeighNa Trader 社区版 - {}   [{}]").format(vnpy.__version__, TRADER_DIR)

        # 🧠 ML Signal: UI setup methods often follow a specific sequence
        self.widgets: dict[str, QtWidgets.QWidget] = {}
        # 🧠 ML Signal: Loading settings is a common pattern in UI initialization
        # 🧠 ML Signal: Usage of a method to create and initialize dock widgets
        self.monitors: dict[str, BaseMonitor] = {}

        self.init_ui()
    # 🧠 ML Signal: Usage of a method to create and initialize dock widgets

    def init_ui(self) -> None:
        """"""
        self.setWindowTitle(self.window_title)
        self.init_dock()
        self.init_toolbar()
        # 🧠 ML Signal: Usage of a method to create and initialize dock widgets
        self.init_menu()
        self.load_window_setting("custom")

    # 🧠 ML Signal: Usage of a method to create and initialize dock widgets
    def init_dock(self) -> None:
        """"""
        self.trading_widget, trading_dock = self.create_dock(
            # 🧠 ML Signal: Usage of a method to create and initialize dock widgets
            TradingWidget, _("交易"), QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
        )
        tick_widget, tick_dock = self.create_dock(
            TickMonitor, _("行情"), QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )
        order_widget, order_dock = self.create_dock(
            # 🧠 ML Signal: Usage of a method to create and initialize dock widgets
            OrderMonitor, _("委托"), QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )
        active_widget, active_dock = self.create_dock(
            # 🧠 ML Signal: Usage of a method to create and initialize dock widgets
            ActiveOrderMonitor, _("活动"), QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )
        trade_widget, trade_dock = self.create_dock(
            TradeMonitor, _("成交"), QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        # 🧠 ML Signal: Usage of a method to create and initialize dock widgets
        )
        log_widget, log_dock = self.create_dock(
            # 🧠 ML Signal: Usage of QtWidgets for GUI menu creation
            LogMonitor, _("日志"), QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
        )
        # 🧠 ML Signal: Usage of tabifying dock widgets
        # ✅ Best Practice: Explicitly setting native menu bar to False for cross-platform consistency
        account_widget, account_dock = self.create_dock(
            AccountMonitor, _("资金"), QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
        # 🧠 ML Signal: Saving window settings
        # 🧠 ML Signal: Adding a system menu with localization support
        )
        position_widget, position_dock = self.create_dock(
            # ⚠️ SAST Risk (Low): Potential for unhandled exceptions if the connection fails
            # 🧠 ML Signal: Dynamic retrieval of gateway names
            # 🧠 ML Signal: Use of partial to bind function arguments
            PositionMonitor, _("持仓"), QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
        )

        self.tabifyDockWidget(active_dock, order_dock)

        self.save_window_setting("default")
        # 🧠 ML Signal: Adding actions to the menu with icons and callbacks

        tick_widget.itemDoubleClicked.connect(self.trading_widget.update_with_cell)
        position_widget.itemDoubleClicked.connect(self.trading_widget.update_with_cell)

    def init_menu(self) -> None:
        """"""
        bar: QtWidgets.QMenuBar = self.menuBar()
        # 🧠 ML Signal: Adding a separator in the menu
        bar.setNativeMenuBar(False)     # for mac and linux
        # 🧠 ML Signal: Adding an exit action to the system menu

        # System menu
        sys_menu: QtWidgets.QMenu = bar.addMenu(_("系统"))

        gateway_names: list = self.main_engine.get_all_gateway_names()
        for name in gateway_names:
            func: Callable = partial(self.connect_gateway, name)
            # 🧠 ML Signal: Adding an application menu with localization support
            self.add_action(
                sys_menu,
                # 🧠 ML Signal: Dynamic retrieval of all applications
                _("连接{}").format(name),
                # ⚠️ SAST Risk (Medium): Dynamic import of modules can lead to code execution risks
                get_icon_path(__file__, "connect.ico"),
                func
            )

        sys_menu.addSeparator()

        self.add_action(
            # 🧠 ML Signal: Creating a QAction for configuration
            # ⚠️ SAST Risk (Medium): Use of getattr can lead to attribute access risks
            # 🧠 ML Signal: Use of partial to bind function arguments
            # 🧠 ML Signal: Adding actions to the menu with icons and callbacks
            sys_menu,
            _("退出"),
            get_icon_path(__file__, "exit.ico"),
            self.close
        )

        # 🧠 ML Signal: Connecting QAction to a slot
        # 🧠 ML Signal: Adding action to the menu bar
        # 🧠 ML Signal: Adding a help menu with localization support
        # App menu
        app_menu: QtWidgets.QMenu = bar.addMenu(_("功能"))

        all_apps: list[BaseApp] = self.main_engine.get_all_apps()
        for app in all_apps:
            ui_module: ModuleType = import_module(app.app_module + ".ui")
            # 🧠 ML Signal: Adding actions to the help menu with icons and callbacks
            widget_class: type[QtWidgets.QWidget] = getattr(ui_module, app.widget_name)

            func = partial(self.open_widget, widget_class, app.app_name)

            self.add_action(app_menu, app.display_name, app.icon_name, func, True)

        # Global setting editor
        action: QtGui.QAction = QtGui.QAction(_("配置"), self)
        action.triggered.connect(self.edit_global_setting)
        bar.addAction(action)

        # Help menu
        help_menu: QtWidgets.QMenu = bar.addMenu(_("帮助"))

        self.add_action(
            # ✅ Best Practice: Type hinting for self.toolbar improves code readability and maintainability
            help_menu,
            _("查询合约"),
            # 🧠 ML Signal: Use of setObjectName with a localized string indicates internationalization support
            get_icon_path(__file__, "contract.ico"),
            partial(self.open_widget, ContractManager, "contract"),
            # ✅ Best Practice: Disabling floatable and movable properties for a toolbar can improve UI consistency
            True
        )

        # ✅ Best Practice: Type hinting for variable 'w' improves code readability
        self.add_action(
            help_menu,
            # ✅ Best Practice: Using a variable for size improves maintainability and readability
            _("还原窗口"),
            get_icon_path(__file__, "restore.ico"),
            # ✅ Best Practice: Type hinting for layout improves code readability and maintainability
            # ✅ Best Practice: Setting spacing for layout improves UI consistency
            self.restore_window_setting
        )

        self.add_action(
            help_menu,
            _("测试邮件"),
            # 🧠 ML Signal: Adding toolbar to a specific area indicates a fixed UI layout pattern
            get_icon_path(__file__, "email.ico"),
            self.send_test_email
        # ✅ Best Practice: Type hinting improves code readability and maintainability.
        )

        self.add_action(
            # ✅ Best Practice: Explicit type declaration for icon improves code clarity.
            help_menu,
            _("社区论坛"),
            # ✅ Best Practice: Explicit type declaration for action improves code clarity.
            get_icon_path(__file__, "forum.ico"),
            self.open_forum,
            # ⚠️ SAST Risk (Low): Ensure 'func' is a safe callable to avoid executing arbitrary code.
            True
        # 🧠 ML Signal: Pattern of adding actions to a menu, useful for UI behavior modeling.
        )

        self.add_action(
            help_menu,
            _("关于"),
            get_icon_path(__file__, "about.ico"),
            # 🧠 ML Signal: Conditional logic for adding actions to a toolbar, useful for UI behavior modeling.
            partial(self.open_widget, AboutDialog, "about"),
        )

    # 🧠 ML Signal: Usage of type hinting for function parameters and return type
    def init_toolbar(self) -> None:
        """"""
        # 🧠 ML Signal: Checking instance type to conditionally store in a dictionary
        self.toolbar: QtWidgets.QToolBar = QtWidgets.QToolBar(self)
        self.toolbar.setObjectName(_("工具栏"))
        self.toolbar.setFloatable(False)
        # 🧠 ML Signal: Creating a QDockWidget with a specific name
        self.toolbar.setMovable(False)

        # ✅ Best Practice: Explicitly setting the widget for the dock
        # Set button size
        w: int = 40
        # ✅ Best Practice: Setting an object name for the dock widget
        size = QtCore.QSize(w, w)
        # ✅ Best Practice: Setting features for the dock widget to enhance user interaction
        self.toolbar.setIconSize(size)

        # Set button spacing
        # 🧠 ML Signal: Adding the dock widget to a specific area
        # 🧠 ML Signal: Type hinting is used, indicating a pattern of explicit type usage
        layout: QtWidgets.QLayout | None = self.toolbar.layout()
        # ✅ Best Practice: Use of type hinting for gateway_name improves code readability and maintainability
        if layout:
            # 🧠 ML Signal: Returning a tuple of widget and dock
            layout.setSpacing(10)
        # 🧠 ML Signal: Instantiation of a dialog object, indicating a UI interaction pattern
        # ✅ Best Practice: Explicit type declaration for dialog variable enhances code clarity

        self.addToolBar(QtCore.Qt.ToolBarArea.LeftToolBarArea, self.toolbar)

    # ✅ Best Practice: Use of QMessageBox for user confirmation is a good practice for critical actions.
    # 🧠 ML Signal: Execution of a dialog, indicating a pattern of user interaction
    # ⚠️ SAST Risk (Low): Potential for blocking call if dialog.exec() is a modal dialog
    def add_action(
        self,
        menu: QtWidgets.QMenu,
        action_name: str,
        icon_name: str,
        func: Callable,
        toolbar: bool = False
    ) -> None:
        # 🧠 ML Signal: User confirmation pattern for application exit.
        """"""
        icon: QtGui.QIcon = QtGui.QIcon(icon_name)
        # 🧠 ML Signal: Iterating over widgets to close them is a common pattern in GUI applications.

        action: QtGui.QAction = QtGui.QAction(action_name, self)
        action.triggered.connect(func)
        # 🧠 ML Signal: Iterating over monitors to save settings is a common pattern in applications with user settings.
        action.setIcon(icon)

        menu.addAction(action)
        # ✅ Best Practice: Saving window settings before closing is a good practice for user experience.

        # ✅ Best Practice: Include type hints for function parameters and return type for better readability and maintainability.
        if toolbar:
            # ⚠️ SAST Risk (Low): Ensure main_engine.close() handles exceptions internally to prevent application crash.
            self.toolbar.addAction(action)

    def create_dock(
        self,
        # 🧠 ML Signal: Usage of dictionary get method with default value to handle missing keys.
        widget_class: type[WidgetType],
        name: str,
        area: QtCore.Qt.DockWidgetArea
    # 🧠 ML Signal: Pattern of instantiating a class with specific arguments.
    ) -> tuple[WidgetType, QtWidgets.QDockWidget]:
        """
        Initialize a dock widget.
        """
        # 🧠 ML Signal: Conditional logic to handle different widget types.
        widget: WidgetType = widget_class(self.main_engine, self.event_engine)      # type: ignore
        # ⚠️ SAST Risk (Low): Blocking call with exec() can freeze the application if not handled properly.
        if isinstance(widget, BaseMonitor):
            self.monitors[name] = widget

        # ✅ Best Practice: Type hinting for 'settings' improves code readability and maintainability.
        dock: QtWidgets.QDockWidget = QtWidgets.QDockWidget(name)
        # 🧠 ML Signal: Pattern of showing a widget in a GUI application.
        dock.setWidget(widget)
        # 🧠 ML Signal: Usage of QSettings to persist application state can be a pattern for ML models.
        dock.setObjectName(name)
        dock.setFeatures(dock.DockWidgetFeature.DockWidgetFloatable | dock.DockWidgetFeature.DockWidgetMovable)
        # 🧠 ML Signal: Saving geometry settings is a common pattern for applications with GUI.
        self.addDockWidget(area, dock)
        return widget, dock

    # 🧠 ML Signal: Use of QtCore.QSettings indicates a pattern of saving/loading application settings
    def connect_gateway(self, gateway_name: str) -> None:
        """
        Open connect dialog for gateway connection.
        """
        dialog: ConnectDialog = ConnectDialog(self.main_engine, gateway_name)
        # ✅ Best Practice: Checking the type of 'state' ensures that the method calls are safe
        dialog.exec()

    # ⚠️ SAST Risk (Low): Potential risk if 'state' or 'geometry' are tampered with, leading to unexpected behavior
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """
        Call main engine close function before exit.
        # ⚠️ SAST Risk (Low): Potential risk if 'geometry' is tampered with, leading to unexpected behavior
        # ✅ Best Practice: Use of a descriptive method name enhances code readability.
        """
        reply = QtWidgets.QMessageBox.question(
            # 🧠 ML Signal: Calling a method with a specific string argument can indicate a common usage pattern.
            self,
            # 🧠 ML Signal: Method calls that change UI state can be indicative of user interaction patterns.
            _("退出"),
            _("确认退出？"),
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            # 🧠 ML Signal: Method for sending emails, useful for detecting communication patterns
            QtWidgets.QMessageBox.StandardButton.No,
        # ⚠️ SAST Risk (Low): Potential for misuse if email content is not properly validated
        )

        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            # ⚠️ SAST Risk (Low): Opening a URL without validation can lead to security risks if the URL is user-controlled.
            for widget in self.widgets.values():
                # 🧠 ML Signal: Usage of webbrowser to open URLs can indicate user interaction patterns.
                widget.close()
            # ✅ Best Practice: Add a docstring to describe the purpose and behavior of the function

            for monitor in self.monitors.values():
                monitor.save_setting()
            # 🧠 ML Signal: Use of a dialog pattern, which may indicate a GUI application
            # 🧠 ML Signal: Use of exec method, which may indicate execution of a dialog or command

            self.save_window_setting("custom")

            self.main_engine.close()

            event.accept()
        else:
            event.ignore()

    def open_widget(self, widget_class: type[QtWidgets.QWidget], name: str) -> None:
        """
        Open contract manager.
        """
        widget: QtWidgets.QWidget | None = self.widgets.get(name, None)
        if not widget:
            widget = widget_class(self.main_engine, self.event_engine)      # type: ignore
            self.widgets[name] = widget

        if isinstance(widget, QtWidgets.QDialog):
            widget.exec()
        else:
            widget.show()

    def save_window_setting(self, name: str) -> None:
        """
        Save current window size and state by trader path and setting name.
        """
        settings: QtCore.QSettings = QtCore.QSettings(self.window_title, name)
        settings.setValue("state", self.saveState())
        settings.setValue("geometry", self.saveGeometry())

    def load_window_setting(self, name: str) -> None:
        """
        Load previous window size and state by trader path and setting name.
        """
        settings: QtCore.QSettings = QtCore.QSettings(self.window_title, name)
        state = settings.value("state")
        geometry = settings.value("geometry")

        if isinstance(state, QtCore.QByteArray):
            self.restoreState(state)
            self.restoreGeometry(geometry)

    def restore_window_setting(self) -> None:
        """
        Restore window to default setting.
        """
        self.load_window_setting("default")
        self.showMaximized()

    def send_test_email(self) -> None:
        """
        Sending a test email.
        """
        self.main_engine.send_email("VeighNa Trader", "testing", None)

    def open_forum(self) -> None:
        """
        """
        webbrowser.open("https://www.vnpy.com/forum/")

    def edit_global_setting(self) -> None:
        """
        """
        dialog: GlobalDialog = GlobalDialog()
        dialog.exec()