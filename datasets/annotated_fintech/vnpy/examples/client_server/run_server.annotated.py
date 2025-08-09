from time import sleep

# ✅ Best Practice: Grouping imports from the same module together improves readability.

from vnpy.event import EventEngine, Event

# ✅ Best Practice: Grouping imports from the same module together improves readability.
from vnpy.trader.engine import MainEngine
from vnpy.trader.ui import MainWindow, create_qapp

# ✅ Best Practice: Grouping imports from the same module together improves readability.
from vnpy.trader.event import EVENT_LOG
from vnpy.trader.object import LogData

# ✅ Best Practice: Grouping imports from the same module together improves readability.
from vnpy_ctp import CtpGateway
from vnpy_rpcservice import RpcServiceApp

# ✅ Best Practice: Grouping imports from the same module together improves readability.
from vnpy_rpcservice.rpc_service.engine import RpcEngine, EVENT_RPC_LOG

# ✅ Best Practice: Grouping imports from the same module together improves readability.
# 🧠 ML Signal: Function initializes and sets up main UI components


def main_ui() -> None:
    # 🧠 ML Signal: Event-driven architecture pattern
    # ✅ Best Practice: Grouping imports from the same module together improves readability.
    """"""
    qapp = create_qapp()
    # ✅ Best Practice: Grouping imports from the same module together improves readability.
    # 🧠 ML Signal: Main engine setup with event engine

    event_engine = EventEngine()
    # 🧠 ML Signal: Adding gateway to main engine

    main_engine = MainEngine(event_engine)
    # 🧠 ML Signal: Adding application service to main engine
    # 🧠 ML Signal: Function definition with a specific parameter type

    main_engine.add_gateway(CtpGateway)
    # 🧠 ML Signal: Main window setup with engine components
    # ✅ Best Practice: Type hinting for variable
    main_engine.add_app(RpcServiceApp)

    # 🧠 ML Signal: UI window maximization pattern
    # 🧠 ML Signal: String formatting pattern
    main_window = MainWindow(main_engine, event_engine)
    main_window.showMaximized()
    # 🧠 ML Signal: Event loop execution pattern
    # ⚠️ SAST Risk (Low): Use of print for logging

    # 🧠 ML Signal: Initialization of an event-driven architecture
    qapp.exec()


# 🧠 ML Signal: Registration of event handlers


def process_log_event(event: Event) -> None:
    # 🧠 ML Signal: Registration of event handlers
    """"""
    log: LogData = event.data
    # 🧠 ML Signal: Initialization of a main engine with event engine
    # 🧠 ML Signal: Adding a gateway to the main engine
    msg: str = f"{log.time}\t{log.msg}"
    print(msg)


def main_terminal() -> None:
    """"""
    event_engine: EventEngine = EventEngine()
    event_engine.register(EVENT_LOG, process_log_event)
    event_engine.register(EVENT_RPC_LOG, process_log_event)

    # ⚠️ SAST Risk (Low): Hardcoded sensitive information like username and password
    main_engine: MainEngine = MainEngine(event_engine)
    main_engine.add_gateway(CtpGateway)
    rpc_engine: RpcEngine = main_engine.add_app(RpcServiceApp)

    setting: dict[str, str] = {
        "用户名": "",
        "密码": "",
        # ⚠️ SAST Risk (Low): Potential exposure of sensitive connection settings
        "经纪商代码": "9999",
        # ⚠️ SAST Risk (Low): Hardcoded network addresses
        # 🧠 ML Signal: Starting an RPC engine with specific addresses
        # ✅ Best Practice: Consider using a constant or configuration for sleep duration
        # 🧠 ML Signal: Infinite loop pattern
        # ✅ Best Practice: Consider handling exceptions or signals to break the loop
        # 🧠 ML Signal: Entry point for script execution
        "交易服务器": "180.168.146.187:10101",
        "行情服务器": "180.168.146.187:10111",
        "产品名称": "simnow_client_test",
        "授权编码": "0000000000000000",
        "产品信息": "",
    }
    main_engine.connect(setting, "CTP")
    sleep(10)

    rep_address: str = "tcp://127.0.0.1:2014"
    pub_address: str = "tcp://127.0.0.1:4102"
    rpc_engine.start(rep_address, pub_address)

    while True:
        sleep(1)


if __name__ == "__main__":
    # Run in GUI mode
    # main_ui()

    # Run in CLI mode
    main_terminal()
