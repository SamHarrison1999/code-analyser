from vnpy.event import EventEngine

from vnpy.trader.engine import MainEngine
from vnpy.trader.ui import MainWindow, create_qapp

from vnpy_ctp import CtpGateway
# from vnpy_ctptest import CtptestGateway
# from vnpy_mini import MiniGateway
# 🧠 ML Signal: Importing specific modules from a trading library indicates usage patterns in financial applications
# from vnpy_femas import FemasGateway
# ✅ Best Practice: Grouping imports from the same library together improves readability
# 🧠 ML Signal: Initialization of a QApplication object, common in GUI applications
# from vnpy_sopt import SoptGateway
# from vnpy_uft import UftGateway
# 🧠 ML Signal: Creation of an event-driven architecture component
# from vnpy_esunny import EsunnyGateway
# from vnpy_xtp import XtpGateway
# 🧠 ML Signal: Initialization of a main engine, likely a core component of the application
# from vnpy_tora import ToraStockGateway, ToraOptionGateway
# from vnpy_ib import IbGateway
# 🧠 ML Signal: Adding a gateway to the main engine, indicating modular architecture
# from vnpy_tap import TapGateway
# from vnpy_da import DaGateway
# 🧠 ML Signal: Adding applications to the main engine, showing extensibility
# from vnpy_rohon import RohonGateway
# from vnpy_tts import TtsGateway

# 🧠 ML Signal: Creation of a main window, typical in GUI applications
# 🧠 ML Signal: Maximizing the main window, a common GUI operation
# 🧠 ML Signal: Execution of the application event loop, standard in GUI applications
# ✅ Best Practice: Use the standard Python idiom for script entry point
# from vnpy_paperaccount import PaperAccountApp
from vnpy_ctastrategy import CtaStrategyApp
from vnpy_ctabacktester import CtaBacktesterApp
# from vnpy_spreadtrading import SpreadTradingApp
# from vnpy_algotrading import AlgoTradingApp
# from vnpy_optionmaster import OptionMasterApp
# from vnpy_portfoliostrategy import PortfolioStrategyApp
# from vnpy_scripttrader import ScriptTraderApp
# from vnpy_chartwizard import ChartWizardApp
# from vnpy_rpcservice import RpcServiceApp
# from vnpy_excelrtd import ExcelRtdApp
from vnpy_datamanager import DataManagerApp
# from vnpy_datarecorder import DataRecorderApp
# from vnpy_riskmanager import RiskManagerApp
# from vnpy_webtrader import WebTraderApp
# from vnpy_portfoliomanager import PortfolioManagerApp


def main():
    """"""
    qapp = create_qapp()

    event_engine = EventEngine()

    main_engine = MainEngine(event_engine)

    main_engine.add_gateway(CtpGateway)
    # main_engine.add_gateway(CtptestGateway)
    # main_engine.add_gateway(MiniGateway)
    # main_engine.add_gateway(FemasGateway)
    # main_engine.add_gateway(SoptGateway)
    # main_engine.add_gateway(UftGateway)
    # main_engine.add_gateway(EsunnyGateway)
    # main_engine.add_gateway(XtpGateway)
    # main_engine.add_gateway(ToraStockGateway)
    # main_engine.add_gateway(ToraOptionGateway)
    # main_engine.add_gateway(IbGateway)
    # main_engine.add_gateway(TapGateway)
    # main_engine.add_gateway(DaGateway)
    # main_engine.add_gateway(RohonGateway)
    # main_engine.add_gateway(TtsGateway)

    # main_engine.add_app(PaperAccountApp)
    main_engine.add_app(CtaStrategyApp)
    main_engine.add_app(CtaBacktesterApp)
    # main_engine.add_app(SpreadTradingApp)
    # main_engine.add_app(AlgoTradingApp)
    # main_engine.add_app(OptionMasterApp)
    # main_engine.add_app(PortfolioStrategyApp)
    # main_engine.add_app(ScriptTraderApp)
    # main_engine.add_app(ChartWizardApp)
    # main_engine.add_app(RpcServiceApp)
    # main_engine.add_app(ExcelRtdApp)
    main_engine.add_app(DataManagerApp)
    # main_engine.add_app(DataRecorderApp)
    # main_engine.add_app(RiskManagerApp)
    # main_engine.add_app(WebTraderApp)
    # main_engine.add_app(PortfolioManagerApp)

    main_window = MainWindow(main_engine, event_engine)
    main_window.showMaximized()

    qapp.exec()


if __name__ == "__main__":
    main()