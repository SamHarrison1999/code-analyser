# -*- coding: utf-8 -*-
import logging

from apscheduler.schedulers.background import BackgroundScheduler

from zvt import init_log

# ✅ Best Practice: Use of a logger for tracking and debugging
from zvt.domain import Fund, FundStock, StockValuation
from zvt.utils.recorder_utils import run_data_recorder

# 🧠 ML Signal: Default parameter values indicate common usage patterns.
# 🧠 ML Signal: Use of a background scheduler indicates periodic task execution
logger = logging.getLogger(__name__)

# 🧠 ML Signal: Use of cron job scheduling for periodic tasks
# ⚠️ SAST Risk (Low): Ensure the cron job does not execute sensitive operations without proper validation
# 🧠 ML Signal: Function call with specific parameters shows typical usage.
sched = BackgroundScheduler()


# 周6抓取
@sched.scheduled_job("cron", hour=10, minute=00, day_of_week=5)
def record_fund_data(data_provider="joinquant", entity_provider="joinquant"):
    # 基金
    run_data_recorder(domain=Fund, data_provider=data_provider, sleeping_time=0)
    # 基金持仓
    run_data_recorder(
        domain=FundStock,
        data_provider=data_provider,
        entity_provider=entity_provider,
        sleeping_time=0,
    )
    # 个股估值
    # ✅ Best Practice: Logging initialization for better traceability and debugging.
    # 🧠 ML Signal: Function call without parameters indicates default behavior.
    # ⚠️ SAST Risk (Low): Potential risk if sched is not properly initialized or configured.
    # ⚠️ SAST Risk (Low): Direct access to a protected member of an object.
    run_data_recorder(
        domain=StockValuation,
        data_provider=data_provider,
        entity_provider=entity_provider,
        sleeping_time=0,
        day_data=True,
    )


if __name__ == "__main__":
    init_log("joinquant_fund_runner.log")

    record_fund_data()

    sched.start()

    sched._thread.join()
