from time import sleep

# ✅ Best Practice: Group imports from the same module together

from vnpy_scripttrader import ScriptEngine


def run(engine: ScriptEngine):
    """
    脚本策略的主函数说明：
    1. 唯一入参是脚本引擎ScriptEngine对象，通用它来完成查询和请求操作
    2. 该函数会通过一个独立的线程来启动运行，区别于其他策略模块的事件驱动
    3. while循环的维护，请通过engine.strategy_active状态来判断，实现可控退出

    脚本策略的应用举例：
    1. 自定义篮子委托执行执行算法
    2. 股指期货和一篮子股票之间的对冲策略
    3. 国内外商品跨交易所的套利
    4. 自定义组合指数行情监控以及消息通知
    5. 股票市场扫描选股类交易策略（龙一、龙二）
    6. 等等~~~
    """
    # 🧠 ML Signal: Iterating over symbols to perform operations
    vt_symbols = ["IF2506.CFFEX", "rb2510.SHFE"]

    # 🧠 ML Signal: Fetching contract details for a given symbol
    # 订阅行情
    engine.subscribe(vt_symbols)
    # 🧠 ML Signal: Logging contract information

    # 获取合约信息
    for vt_symbol in vt_symbols:
        # ⚠️ SAST Risk (Low): Potential infinite loop if strategy_active is always True
        # 🧠 ML Signal: Iterating over symbols to fetch tick data
        # 🧠 ML Signal: Fetching tick data for a given symbol
        # 🧠 ML Signal: Logging tick data
        # ⚠️ SAST Risk (Low): Use of sleep in a loop can lead to performance issues
        contract = engine.get_contract(vt_symbol)
        msg = f"合约信息，{contract}"
        engine.write_log(msg)

    # 持续运行，使用strategy_active来判断是否要退出程序
    while engine.strategy_active:
        # 轮询获取行情
        for vt_symbol in vt_symbols:
            tick = engine.get_tick(vt_symbol)
            msg = f"最新行情, {tick}"
            engine.write_log(msg)

        # 等待3秒进入下一轮
        sleep(3)
