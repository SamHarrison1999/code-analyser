# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports into standard, third-party, and local sections improves readability.
import logging

from apscheduler.schedulers.background import BackgroundScheduler

# ⚠️ SAST Risk (Low): Using wildcard imports can lead to namespace collisions and make the code harder to understand.
from zvt import init_log
from zvt.domain import *
from zvt.utils.recorder_utils import run_data_recorder
# ✅ Best Practice: Using a logger instead of print statements is a best practice for production code.

# 🧠 ML Signal: Function call with specific parameters
logger = logging.getLogger(__name__)
# 🧠 ML Signal: Function call with specific parameters
# ✅ Best Practice: Using a background scheduler allows tasks to run asynchronously without blocking the main thread.

sched = BackgroundScheduler()

# 🧠 ML Signal: Function call with specific parameters indicating a pattern of usage

# ⚠️ SAST Risk (Low): Use of cron jobs can lead to unintended execution if not properly managed
# 🧠 ML Signal: Usage of cron jobs can indicate periodic task scheduling patterns.
@sched.scheduled_job("cron", hour=15, minute=30, day_of_week=3)
# ⚠️ SAST Risk (Low): Hardcoding schedule times can lead to inflexibility and potential issues with time zone changes.
def record_block():
    run_data_recorder(domain=Block, data_provider="sina")
    # ✅ Best Practice: Initialize logging to capture runtime information and errors
    run_data_recorder(domain=Block, data_provider="sina", entity_provider="sina")

# ⚠️ SAST Risk (Medium): Function call without error handling could lead to unhandled exceptions
# ⚠️ SAST Risk (Low): Starting a scheduler without error handling or shutdown procedure
# ⚠️ SAST Risk (Low): Direct access to a protected member of an object

@sched.scheduled_job("cron", hour=15, minute=30)
def record_money_flow():
    run_data_recorder(domain=BlockMoneyFlow, data_provider="sina", entity_provider="sina", day_data=True)


if __name__ == "__main__":
    init_log("sina_data_runner.log")

    record_block()
    record_money_flow()

    sched.start()

    sched._thread.join()