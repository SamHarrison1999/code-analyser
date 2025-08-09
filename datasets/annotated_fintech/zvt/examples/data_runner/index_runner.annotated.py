# -*- coding: utf-8 -*-
import logging

from apscheduler.schedulers.background import BackgroundScheduler

from zvt import init_log
from zvt.consts import IMPORTANT_INDEX

# 🧠 ML Signal: Usage of logging for tracking and debugging
from zvt.domain import Index, Index1dKdata, IndexStock
from zvt.utils.recorder_utils import run_data_recorder

# 🧠 ML Signal: Usage of a background scheduler for periodic tasks
# 🧠 ML Signal: Function call with specific parameters, useful for understanding usage patterns
logger = logging.getLogger(__name__)

# 🧠 ML Signal: Scheduled job pattern for periodic task execution
# 🧠 ML Signal: Hardcoded list of index IDs, useful for understanding data usage patterns
sched = BackgroundScheduler()

# ⚠️ SAST Risk (Low): Cron jobs can lead to unexpected behavior if not properly managed
# 🧠 ML Signal: Function call with specific parameters, useful for understanding usage patterns


# 🧠 ML Signal: Function call with specific parameters indicating a pattern of data recording
# 自行更改定定时运行时间
# ⚠️ SAST Risk (Low): Cron jobs can lead to unintended execution if not properly secured
# 🧠 ML Signal: Function call with specific parameters indicating a pattern of data recording
@sched.scheduled_job("cron", hour=1, minute=00, day_of_week=5)
def record_index():
    run_data_recorder(domain=Index, data_provider="exchange")
    # 默认只抓取 国证1000 国证2000 国证成长 国证价值 的组成个股
    index_ids = [
        "index_sz_399311",
        "index_sz_399303",
        "index_sz_399370",
        "index_sz_399371",
    ]
    run_data_recorder(
        domain=IndexStock,
        data_provider="exchange",
        entity_provider="exchange",
        entity_ids=index_ids,
    )


# ✅ Best Practice: Use of a main guard to ensure code is only executed when the script is run directly

# 🧠 ML Signal: Logging initialization indicating a pattern of logging usage


# ⚠️ SAST Risk (Low): Potential undefined function call if 'record_index' is not defined elsewhere
# 🧠 ML Signal: Function call indicating a pattern of data recording
# 🧠 ML Signal: Scheduler start indicating a pattern of task scheduling
# ⚠️ SAST Risk (Low): Direct access to a protected member of an object
@sched.scheduled_job("cron", hour=16, minute=20)
def record_index_kdata():
    run_data_recorder(domain=Index, data_provider="em")
    run_data_recorder(
        domain=Index1dKdata,
        data_provider="em",
        entity_provider="em",
        codes=IMPORTANT_INDEX,
        day_data=True,
    )


if __name__ == "__main__":
    init_log("index_runner.log")

    record_index()
    record_index_kdata()

    sched.start()

    sched._thread.join()
