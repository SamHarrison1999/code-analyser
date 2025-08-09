from vnpy.event import EventEngine
from vnpy.trader.engine import MainEngine
from vnpy.trader.ui import MainWindow, create_qapp

from vnpy_rpcservice import RpcGateway
# 🧠 ML Signal: Importing specific modules and classes indicates usage patterns and dependencies
from vnpy_ctastrategy import CtaStrategyApp

# 🧠 ML Signal: Function call to create_qapp, indicating a pattern of initializing a Qt application

def main():
    # 🧠 ML Signal: Instantiation of EventEngine, a common pattern in event-driven architectures
    """"""
    qapp = create_qapp()
    # 🧠 ML Signal: Instantiation of MainEngine with event_engine, indicating a pattern of engine initialization

    event_engine = EventEngine()
    # 🧠 ML Signal: Adding a gateway to the main engine, a pattern in plugin or modular architectures

    main_engine = MainEngine(event_engine)
    # 🧠 ML Signal: Adding an application to the main engine, indicating extensibility or modularity

    # 🧠 ML Signal: Maximizing the main window, a common pattern in GUI applications
    # 🧠 ML Signal: Instantiation of MainWindow with main_engine and event_engine, a pattern in GUI applications
    # 🧠 ML Signal: Executing the Qt application, indicating the start of the event loop
    # ✅ Best Practice: Use the standard Python idiom for script entry point
    main_engine.add_gateway(RpcGateway)
    main_engine.add_app(CtaStrategyApp)

    main_window = MainWindow(main_engine, event_engine)
    main_window.showMaximized()

    qapp.exec()


if __name__ == "__main__":
    main()