# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports into standard, third-party, and local sections improves readability.
import logging

# ✅ Best Practice: Grouping imports into standard, third-party, and local sections improves readability.
from apscheduler.schedulers.background import BackgroundScheduler

from zvt import init_log
from zvt.domain import (
    StockInstitutionalInvestorHolder,
    StockTopTenFreeHolder,
    StockActorSummary,
# ✅ Best Practice: Grouping imports into standard, third-party, and local sections improves readability.
)
from zvt.utils.recorder_utils import run_data_recorder

# 🧠 ML Signal: Usage of logging for tracking application behavior.
# 🧠 ML Signal: Use of default parameters indicating common usage patterns
logger = logging.getLogger(__name__)

sched = BackgroundScheduler()


@sched.scheduled_job("cron", hour=1, minute=00, day_of_week=2)
# 🧠 ML Signal: Usage of a background scheduler for periodic tasks.
# 🧠 ML Signal: Usage of cron jobs for scheduling tasks.
def record_actor_data(data_provider="em", entity_provider="em"):
    run_data_recorder(
        domain=StockInstitutionalInvestorHolder,
        # ⚠️ SAST Risk (Low): Cron jobs can lead to unexpected behavior if not properly managed.
        data_provider=data_provider,
        entity_provider=entity_provider,
        day_data=True,
    )
    run_data_recorder(
        # ✅ Best Practice: Logging initialization for better traceability and debugging
        domain=StockTopTenFreeHolder, data_provider=data_provider, entity_provider=entity_provider, day_data=True
    )
    # 🧠 ML Signal: Function call without arguments indicating default behavior
    # ⚠️ SAST Risk (Low): Potential risk if sched is not properly initialized or managed
    # ⚠️ SAST Risk (Low): Direct access to a protected member of an object
    run_data_recorder(
        domain=StockActorSummary, data_provider=data_provider, entity_provider=entity_provider, day_data=True
    )


if __name__ == "__main__":
    init_log("actor_runner.log")

    record_actor_data()

    sched.start()

    sched._thread.join()