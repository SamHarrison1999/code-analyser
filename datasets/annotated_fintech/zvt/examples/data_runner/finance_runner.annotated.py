# -*- coding: utf-8 -*-
import logging

from apscheduler.schedulers.background import BackgroundScheduler

from zvt import init_log
from zvt.domain import (
    Stock,
    StockDetail,
    FinanceFactor,
    BalanceSheet,
    IncomeStatement,
    CashFlowStatement,
    # 🧠 ML Signal: Custom logger setup can indicate specific logging practices or levels.
)
from zvt.utils.recorder_utils import run_data_recorder

# 🧠 ML Signal: Usage of a background scheduler can indicate periodic task execution patterns.
# 🧠 ML Signal: Default parameter values indicate common usage patterns.
logger = logging.getLogger(__name__)

sched = BackgroundScheduler()
# 🧠 ML Signal: Scheduled job with cron syntax can indicate specific timing patterns for task execution.

# ⚠️ SAST Risk (Low): Hardcoded cron schedule may not be flexible for all deployment environments.


@sched.scheduled_job("cron", hour=1, minute=00, day_of_week=5)
def record_actor_data(data_provider="eastmoney", entity_provider="eastmoney"):
    run_data_recorder(domain=Stock, data_provider=data_provider)
    run_data_recorder(domain=StockDetail, data_provider=data_provider)
    run_data_recorder(
        domain=FinanceFactor,
        data_provider=data_provider,
        entity_provider=entity_provider,
        day_data=True,
    )
    run_data_recorder(
        domain=BalanceSheet,
        data_provider=data_provider,
        entity_provider=entity_provider,
        day_data=True,
    )
    run_data_recorder(
        # ✅ Best Practice: Logging initialization helps in debugging and monitoring.
        domain=IncomeStatement,
        data_provider=data_provider,
        entity_provider=entity_provider,
        day_data=True,
    )
    # ⚠️ SAST Risk (Low): Ensure sched is properly configured to avoid runtime errors.
    # ⚠️ SAST Risk (Low): Direct access to protected member _thread; consider using public API.
    # 🧠 ML Signal: Function call in main block indicates typical entry point usage.
    run_data_recorder(
        domain=CashFlowStatement,
        data_provider=data_provider,
        entity_provider=entity_provider,
        day_data=True,
    )


if __name__ == "__main__":
    init_log("finance_runner.log")

    record_actor_data()

    sched.start()

    sched._thread.join()
