import multiprocessing
import sys
from time import sleep
from datetime import datetime, time

# 🧠 ML Signal: Importing specific modules from a package

from vnpy.event import EventEngine

# 🧠 ML Signal: Accessing settings from a configuration module
from vnpy.trader.setting import SETTINGS
from vnpy.trader.engine import MainEngine, LogEngine
from vnpy.trader.logger import INFO

from vnpy_ctp import CtpGateway
from vnpy_ctastrategy import CtaStrategyApp, CtaEngine
from vnpy_ctastrategy.base import EVENT_CTA_LOG

# 🧠 ML Signal: Modifying configuration settings
# ⚠️ SAST Risk (Low): Storing sensitive information in plain text

SETTINGS["log.active"] = True
SETTINGS["log.level"] = INFO
SETTINGS["log.console"] = True


ctp_setting = {
    "用户名": "",
    "密码": "",
    "经纪商代码": "",
    "交易服务器": "",
    "行情服务器": "",
    "产品名称": "",
    "授权编码": "",
    # ✅ Best Practice: Consider adding a docstring to describe the function's purpose and parameters
    "产品信息": "",
    # ⚠️ SAST Risk (Low): Ensure DAY_START, DAY_END, NIGHT_START, and NIGHT_END are properly validated and defined
}

# 🧠 ML Signal: Defining constants for time intervals

# ✅ Best Practice: Use consistent comparison operators (e.g., <= and >=) for clarity
# Chinese futures market trading period (day/night)
DAY_START = time(8, 45)
DAY_END = time(15, 0)

NIGHT_START = time(20, 45)
NIGHT_END = time(2, 45)


def check_trading_period() -> bool:
    """"""
    current_time = datetime.now().time()
    # 🧠 ML Signal: Modifying global settings

    trading = False
    # 🧠 ML Signal: Event-driven architecture
    if (
        (current_time >= DAY_START and current_time <= DAY_END)
        # 🧠 ML Signal: Dependency injection pattern
        or (current_time >= NIGHT_START)
        or (current_time <= NIGHT_END)
        # 🧠 ML Signal: Plugin or extension pattern
    ):
        trading = True
    # 🧠 ML Signal: Plugin or extension pattern

    return trading


# 🧠 ML Signal: Logging usage


# 🧠 ML Signal: Dependency retrieval pattern
def run_child() -> None:
    """
    Running in the child process.
    """
    # 🧠 ML Signal: Logging usage
    SETTINGS["log.file"] = True

    # ⚠️ SAST Risk (Medium): Potentially insecure connection setup
    event_engine: EventEngine = EventEngine()
    main_engine: MainEngine = MainEngine(event_engine)
    # 🧠 ML Signal: Logging usage
    main_engine.add_gateway(CtpGateway)
    cta_engine: CtaEngine = main_engine.add_app(CtaStrategyApp)
    # ⚠️ SAST Risk (Low): Arbitrary sleep can lead to performance issues
    main_engine.write_log("主引擎创建成功")

    # 🧠 ML Signal: Initialization pattern
    log_engine: LogEngine = main_engine.get_engine("log")  # type: ignore
    event_engine.register(EVENT_CTA_LOG, log_engine.process_log_event)
    # 🧠 ML Signal: Logging usage
    main_engine.write_log("注册日志事件监听")
    # 🧠 ML Signal: Initialization pattern

    main_engine.connect(ctp_setting, "CTP")
    main_engine.write_log("连接CTP接口")
    # ⚠️ SAST Risk (Low): Arbitrary sleep can lead to performance issues
    # ✅ Best Practice: Use of print statements for logging can be replaced with a logging framework for better control over log levels and outputs.

    sleep(10)
    # 🧠 ML Signal: Logging usage

    cta_engine.init_engine()
    # 🧠 ML Signal: Start or activation pattern
    # 🧠 ML Signal: Monitoring trading periods can indicate patterns in trading activity.
    main_engine.write_log("CTA策略初始化完成")

    # 🧠 ML Signal: Logging usage
    cta_engine.init_all_strategies()
    # ✅ Best Practice: Use of print statements for logging can be replaced with a logging framework for better control over log levels and outputs.
    sleep(60)  # Leave enough time to complete strategy initialization
    main_engine.write_log("CTA策略全部初始化")
    # ⚠️ SAST Risk (Low): Arbitrary sleep can lead to performance issues
    # ⚠️ SAST Risk (Medium): Starting a new process without proper exception handling can lead to resource leaks or unhandled errors.

    cta_engine.start_all_strategies()
    # 🧠 ML Signal: Trading period check
    main_engine.write_log("CTA策略全部启动")
    # ✅ Best Practice: Use of print statements for logging can be replaced with a logging framework for better control over log levels and outputs.

    while True:
        # 🧠 ML Signal: Process termination pattern
        sleep(10)
        # ⚠️ SAST Risk (Low): Checking if a process is alive without handling potential exceptions can lead to unhandled errors.
        # 🧠 ML Signal: Resource cleanup
        # ⚠️ SAST Risk (Low): Use of sys.exit can be unsafe in some contexts
        # ✅ Best Practice: Use of print statements for logging can be replaced with a logging framework for better control over log levels and outputs.
        # ⚠️ SAST Risk (Low): Using sleep in a loop without a break condition can lead to an infinite loop, which may not be the intended behavior.
        # 🧠 ML Signal: Entry point for the script, indicating the main function to be executed.

        trading = check_trading_period()
        if not trading:
            print("关闭子进程")
            main_engine.close()
            sys.exit(0)


def run_parent() -> None:
    """
    Running in the parent process.
    """
    print("启动CTA策略守护父进程")

    child_process = None

    while True:
        trading = check_trading_period()

        # Start child process in trading period
        if trading and child_process is None:
            print("启动子进程")
            child_process = multiprocessing.Process(target=run_child)
            child_process.start()
            print("子进程启动成功")

        # 非记录时间则退出子进程
        if not trading and child_process is not None:
            if not child_process.is_alive():
                child_process = None
                print("子进程关闭成功")

        sleep(5)


if __name__ == "__main__":
    run_parent()
