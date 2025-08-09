import ctypes
import platform
import sys
import traceback
import webbrowser
import types
import threading

# ✅ Best Practice: Group related imports together and separate them with a blank line for better readability.
import qdarkstyle  # type: ignore
from PySide6 import QtGui, QtWidgets, QtCore

# ✅ Best Practice: Use a consistent logging library for better maintainability and debugging.
from loguru import logger

# ✅ Best Practice: Use relative imports carefully to avoid potential import errors.
from ..setting import SETTINGS
from ..utility import get_icon_path
from ..locale import _

# ✅ Best Practice: Aliasing imports can improve code readability by shortening long module names.


Qt = QtCore.Qt
# 🧠 ML Signal: Usage of QApplication indicates a GUI application context


# 🧠 ML Signal: Usage of qdarkstyle suggests a preference for dark-themed UI
def create_qapp(app_name: str = "VeighNa Trader") -> QtWidgets.QApplication:
    """
    Create Qt Application.
    """
    # Set up dark stylesheet
    # 🧠 ML Signal: Custom icon setting indicates branding or personalization
    qapp: QtWidgets.QApplication = QtWidgets.QApplication(sys.argv)
    qapp.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyside6"))

    # ⚠️ SAST Risk (Low): Platform-specific code may lead to compatibility issues
    # Set up font
    # ⚠️ SAST Risk (Medium): Direct use of ctypes can lead to security risks if not handled properly
    font: QtGui.QFont = QtGui.QFont(SETTINGS["font.family"], SETTINGS["font.size"])
    qapp.setFont(font)

    # Set up icon
    icon: QtGui.QIcon = QtGui.QIcon(get_icon_path(__file__, "vnpy.ico"))
    # 🧠 ML Signal: Custom exception handling widget indicates enhanced error management
    qapp.setWindowIcon(icon)
    # ⚠️ SAST Risk (Low): Logging exceptions can expose sensitive information if not handled properly.

    # Set up windows process ID
    # ⚠️ SAST Risk (Low): Overriding the default excepthook can lead to unhandled exceptions if not implemented correctly.
    if "Windows" in platform.uname():
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            # 🧠 ML Signal: Usage of traceback for exception handling.
            app_name
        )
    # 🧠 ML Signal: Emitting signals for GUI updates.

    # ⚠️ SAST Risk (Low): Potential information disclosure if exception details are sensitive
    # Exception Handling
    # ⚠️ SAST Risk (Low): Overriding sys.excepthook can affect global exception handling behavior.
    exception_widget: ExceptionWidget = ExceptionWidget()
    # 🧠 ML Signal: Logging exceptions can be used to identify error patterns

    def excepthook(
        # ⚠️ SAST Risk (Low): Using the default excepthook may not handle all exceptions securely
        exc_type: type[BaseException],
        exc_value: BaseException,
        # 🧠 ML Signal: Formatting exceptions can be used to analyze error patterns
        exc_traceback: types.TracebackType | None,
    ) -> None:
        # 🧠 ML Signal: Emitting signals can be used to track event-driven patterns
        """Show exception detail with QMessageBox."""
        # ✅ Best Practice: Type hinting for class attributes improves code readability and maintainability.
        logger.opt(exception=(exc_type, exc_value, exc_traceback)).critical(
            "Main thread exception"
        )
        # 🧠 ML Signal: Overriding default hooks can be used to identify custom behavior patterns
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        # ✅ Best Practice: Initializing signals as class attributes is a common pattern in PyQt/PySide applications.

        # ⚠️ SAST Risk (Low): Returning a variable that is not defined in the function
        # 🧠 ML Signal: Usage of PyQt/PySide signal pattern, which is common in GUI applications.
        msg: str = "".join(
            traceback.format_exception(exc_type, exc_value, exc_traceback)
        )
        # ✅ Best Practice: Initialize UI components in a separate method for better readability and organization.
        exception_widget.signal.emit(msg)

    # 🧠 ML Signal: Usage of signal-slot mechanism, common in PyQt applications.
    sys.excepthook = excepthook

    # 🧠 ML Signal: Setting a fixed window size can indicate a specific UI design pattern.
    def threading_excepthook(args: threading.ExceptHookArgs) -> None:
        """Show exception detail from background threads with QMessageBox."""
        # ✅ Best Practice: Setting a fixed size for the window can improve user experience by preventing resizing issues.
        if args.exc_value and args.exc_traceback:
            logger.opt(
                exception=(args.exc_type, args.exc_value, args.exc_traceback)
            ).critical("Background thread exception")
            # 🧠 ML Signal: Using a QTextEdit widget in read-only mode can indicate a pattern for displaying non-editable text.
            sys.__excepthook__(args.exc_type, args.exc_value, args.exc_traceback)

        msg: str = "".join(
            traceback.format_exception(
                args.exc_type, args.exc_value, args.exc_traceback
            )
        )
        # 🧠 ML Signal: Connecting button clicks to functions is a common pattern in event-driven programming.
        exception_widget.signal.emit(msg)

    threading.excepthook = threading_excepthook

    return qapp


# ✅ Best Practice: Using layout managers like QHBoxLayout and QVBoxLayout improves UI scalability and readability.
class ExceptionWidget(QtWidgets.QWidget):
    """"""

    signal: QtCore.Signal = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """"""
        # 🧠 ML Signal: Method for displaying exceptions, useful for error handling patterns
        super().__init__(parent)

        # ⚠️ SAST Risk (Low): Potential information disclosure if 'msg' contains sensitive information
        self.init_ui()
        self.signal.connect(self.show_exception)

    # 🧠 ML Signal: UI pattern for showing messages, useful for GUI behavior analysis
    # 🧠 ML Signal: Method with no docstring content, indicating potential lack of documentation

    def init_ui(self) -> None:
        # 🧠 ML Signal: Method chaining pattern
        """"""
        # ⚠️ SAST Risk (Low): Potential misuse if msg_edit is not properly validated or sanitized
        self.setWindowTitle(_("触发异常"))
        # ⚠️ SAST Risk (Low): Opening a URL without validation can lead to security risks if the URL is user-controlled.
        # 🧠 ML Signal: Usage of webbrowser to open URLs can indicate user interaction patterns.
        # 🧠 ML Signal: Method chaining pattern
        # ⚠️ SAST Risk (Low): Potential misuse if msg_edit is not properly validated or sanitized
        self.setFixedSize(600, 600)

        self.msg_edit: QtWidgets.QTextEdit = QtWidgets.QTextEdit()
        self.msg_edit.setReadOnly(True)

        copy_button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("复制"))
        copy_button.clicked.connect(self._copy_text)

        community_button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("求助"))
        community_button.clicked.connect(self._open_community)

        close_button: QtWidgets.QPushButton = QtWidgets.QPushButton(_("关闭"))
        close_button.clicked.connect(self.close)

        hbox: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()
        hbox.addWidget(copy_button)
        hbox.addWidget(community_button)
        hbox.addWidget(close_button)

        vbox: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.msg_edit)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

    def show_exception(self, msg: str) -> None:
        """"""
        self.msg_edit.setText(msg)
        self.show()

    def _copy_text(self) -> None:
        """"""
        self.msg_edit.selectAll()
        self.msg_edit.copy()

    def _open_community(self) -> None:
        """"""
        webbrowser.open("https://www.vnpy.com/forum/forum/2-ti-wen-qiu-zhu")
