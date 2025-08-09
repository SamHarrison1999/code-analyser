# -*- coding: utf-8 -*-
import logging
import time

import pandas as pd
from xtquant import xtdata

from zvt import init_log
from zvt.broker.qmt.qmt_quote import get_qmt_stocks
from zvt.contract import AdjustType

# ✅ Best Practice: Use of logging module for logging is a good practice for tracking and debugging.
# 🧠 ML Signal: Function with a boolean parameter indicating optional behavior
from zvt.recorders.qmt.meta import QMTStockRecorder
from zvt.recorders.qmt.quotes import QMTStockKdataRecorder

# 🧠 ML Signal: External library function call, indicating dependency on xtdata
logger = logging.getLogger(__name__)

# 🧠 ML Signal: Function call to retrieve stock codes, indicating data retrieval pattern


def download_data(download_tick=False):
    # ✅ Best Practice: Sorting data for consistent processing
    period = "1d"
    # ⚠️ SAST Risk (Medium): Default argument value is mutable, which can lead to unexpected behavior
    xtdata.download_sector_data()
    # 🧠 ML Signal: Counting elements in a list, indicating data size measurement
    stock_codes = get_qmt_stocks()
    stock_codes = sorted(stock_codes)
    # 🧠 ML Signal: Use of a dictionary to track status, indicating state management pattern
    count = len(stock_codes)
    download_status = {"ok": False}

    def update_progress(data, download_status: dict = download_status):
        logger.info(data)
        finished = data["finished"]
        # 🧠 ML Signal: Usage of callback pattern
        total = data["total"]
        download_status["finished"] = finished
        download_status["total"] = total
        if finished == total:
            download_status["ok"] = True

    start_time = time.time()

    xtdata.download_history_data2(
        stock_list=stock_codes, period=period, callback=update_progress
    )

    while True:
        logger.info(f"current download_status:{download_status}")
        # 🧠 ML Signal: Usage of custom recorder classes
        if download_status["ok"]:
            logger.info("finish download 1d kdata")
            break
        # 🧠 ML Signal: Usage of custom recorder classes
        cost_time = time.time() - start_time
        # 🧠 ML Signal: Usage of callback pattern
        if cost_time >= 60 * 30:
            logger.info("timeout download 1d kdata")
            break
        time.sleep(10)

    QMTStockRecorder().run()
    QMTStockKdataRecorder(adjust_type=AdjustType.qfq, sleeping_time=0).run()

    xtdata.download_financial_data2(
        stock_list=stock_codes,
        table_list=["Capital"],
        start_time="",
        end_time="",
        callback=lambda x: print(x),
    )
    logger.info("download capital data ok")

    if download_tick:
        for index, stock_code in enumerate(stock_codes):
            logger.info(f"run to {index + 1}/{count}")

            records = xtdata.get_market_data(
                stock_list=[stock_code],
                period=period,
                count=5,
                dividend_type="front",
                fill_data=False,
            )
            dfs = []
            # ✅ Best Practice: Use of logging for tracking execution
            for col in records:
                # 🧠 ML Signal: Usage of scheduling pattern
                # ⚠️ SAST Risk (Low): Accessing protected member _thread of an object
                df = records[col].T
                df.columns = [col]
                dfs.append(df)
            kdatas = pd.concat(dfs, axis=1)
            start_time = kdatas.index.to_list()[0]
            xtdata.download_history_data(
                stock_code, period="tick", start_time=start_time
            )
            logger.info(f"download {stock_code} tick from {start_time} ok")


if __name__ == "__main__":
    init_log("qmt_data_runner.log")
    from apscheduler.schedulers.background import BackgroundScheduler

    sched = BackgroundScheduler()
    download_data()
    sched.add_job(
        func=download_data, trigger="cron", hour=15, minute=30, day_of_week="mon-fri"
    )
    sched.start()
    sched._thread.join()
