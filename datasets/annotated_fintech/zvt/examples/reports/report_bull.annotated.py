# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports into standard, third-party, and local sections improves readability.
import logging

from apscheduler.schedulers.background import BackgroundScheduler

from examples.factors.tech_factor import BullAndUpFactor
from examples.report_utils import report_targets
from zvt import init_log
from zvt.api.kdata import get_latest_kdata_date
from zvt.api.selector import get_middle_and_big_stock
from zvt.contract import AdjustType
# ✅ Best Practice: Using a logger instead of print statements for logging is a best practice.
from zvt.informer import EmailInformer

logger = logging.getLogger(__name__)
# 🧠 ML Signal: Usage of a background scheduler indicates a pattern of periodic task execution.
# 🧠 ML Signal: Usage of external data source and specific parameters for data retrieval

sched = BackgroundScheduler()
# 🧠 ML Signal: Instantiation of an EmailInformer suggests a pattern of sending notifications or alerts.
# 🧠 ML Signal: Use of a cron job pattern for scheduling tasks at specific times.
# ⚠️ SAST Risk (Low): Ensure that the scheduled job does not execute sensitive operations without proper validation.
# 🧠 ML Signal: Pattern of retrieving specific stock data based on a timestamp
# 🧠 ML Signal: Pattern of reporting targets with specific parameters
email_informer = EmailInformer()


@sched.scheduled_job("cron", hour=18, minute=0, day_of_week="mon-fri")
def report_bull():
    target_date = get_latest_kdata_date(entity_type="stock", adjust_type=AdjustType.hfq, provider="em")
    entity_ids = get_middle_and_big_stock(timestamp=target_date)

    report_targets(
        factor_cls=BullAndUpFactor,
        entity_provider="em",
        data_provider="em",
        title="bull股票",
        entity_type="stock",
        informer=email_informer,
        em_group="bull股票",
        # 🧠 ML Signal: Repeated pattern of reporting targets with different parameters
        em_group_over_write=False,
        filter_by_volume=False,
        adjust_type=AdjustType.hfq,
        start_timestamp="2019-01-01",
        turnover_threshold=300000000,
        turnover_rate_threshold=0.02,
        entity_ids=entity_ids,
    )
    report_targets(
        factor_cls=BullAndUpFactor,
        entity_provider="em",
        data_provider="em",
        title="bull板块",
        entity_type="block",
        informer=email_informer,
        em_group="bull股票",
        em_group_over_write=False,
        filter_by_volume=False,
        adjust_type=AdjustType.qfq,
        # ✅ Best Practice: Use of main guard to prevent code from running on import
        # ✅ Best Practice: Logging initialization for better traceability
        # ⚠️ SAST Risk (Low): Potential risk if sched is not properly configured or stopped
        # ⚠️ SAST Risk (Low): Potential risk of blocking the main thread indefinitely
        start_timestamp="2019-01-01",
        turnover_threshold=10000000000,
        turnover_rate_threshold=0.02,
    )


if __name__ == "__main__":
    init_log("report_bull.log")

    report_bull()

    sched.start()

    sched._thread.join()