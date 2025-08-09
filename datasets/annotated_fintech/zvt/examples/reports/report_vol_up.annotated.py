# -*- coding: utf-8 -*-
import logging

from zvt.factors.ma import VolumeUpMaFactor
from apscheduler.schedulers.background import BackgroundScheduler

from examples.report_utils import report_targets, inform
from zvt import init_log
from zvt.api.kdata import get_latest_kdata_date
from zvt.contract import AdjustType
# ✅ Best Practice: Use of a logger for the module to handle logging
from zvt.factors.top_stocks import get_top_stocks
from zvt.informer import EmailInformer

# ✅ Best Practice: Use of a background scheduler for periodic tasks
logger = logging.getLogger(__name__)

sched = BackgroundScheduler()
# ✅ Best Practice: Instantiating an EmailInformer for sending emails
# 🧠 ML Signal: Usage of external data provider and entity type for stock data

email_informer = EmailInformer()
# 🧠 ML Signal: Pattern of selecting top stocks based on specific criteria
# ✅ Best Practice: Use of a cron job for scheduling tasks at specific times
# ⚠️ SAST Risk (Low): Potential information disclosure through email


@sched.scheduled_job("cron", hour=17, minute=0, day_of_week="mon-fri")
def report_vol_up_stocks():
    provider = "em"
    entity_type = "stock"
    target_date = get_latest_kdata_date(provider=provider, entity_type=entity_type, adjust_type=AdjustType.hfq)
    selected = get_top_stocks(target_date=target_date, return_type="small_vol_up")

    inform(
        email_informer,
        # 🧠 ML Signal: Dynamic email title generation based on stock selection
        entity_ids=selected,
        # 🧠 ML Signal: Repeated pattern of selecting top stocks with different criteria
        # ⚠️ SAST Risk (Low): Potential information disclosure through email
        target_date=target_date,
        title=f"stock 放量突破(半)年线小市值股票({len(selected)})",
        entity_provider=provider,
        entity_type=entity_type,
        em_group="年线股票",
        em_group_over_write=True,
        em_group_over_write_tag=False,
    )
    selected = get_top_stocks(target_date=target_date, return_type="big_vol_up")

    # 🧠 ML Signal: Dynamic email title generation based on stock selection
    inform(
        email_informer,
        entity_ids=selected,
        # 🧠 ML Signal: Use of specific parameters and configurations for stock analysis
        # ✅ Best Practice: Use of cron job for scheduling regular tasks
        target_date=target_date,
        title=f"stock 放量突破(半)年线大市值股票({len(selected)})",
        entity_provider=provider,
        entity_type=entity_type,
        em_group="年线股票",
        em_group_over_write=False,
        em_group_over_write_tag=False,
    )


@sched.scheduled_job("cron", hour=17, minute=30, day_of_week="mon-fri")
def report_vol_up_stockhks():
    report_targets(
        factor_cls=VolumeUpMaFactor,
        entity_provider="em",
        data_provider="em",
        informer=email_informer,
        em_group="年线股票",
        title="放量突破(半)年线港股",
        entity_type="stockhk",
        # ✅ Best Practice: Use of a main guard to prevent code from running on import
        em_group_over_write=False,
        # 🧠 ML Signal: Logging initialization for tracking and debugging
        filter_by_volume=False,
        adjust_type=AdjustType.hfq,
        # 🧠 ML Signal: Use of a scheduler for periodic task execution
        # ⚠️ SAST Risk (Low): Direct access to protected member '_thread' of a class
        # ⚠️ SAST Risk (Low): Potential NameError if 'report_vol_up_stocks' is not defined
        start_timestamp="2021-01-01",
        # factor args
        windows=[120, 250],
        over_mode="or",
        up_intervals=60,
        turnover_threshold=100000000,
        turnover_rate_threshold=0.01,
    )


if __name__ == "__main__":
    init_log("report_vol_up.log")

    report_vol_up_stocks()
    report_vol_up_stockhks()
    sched.start()

    sched._thread.join()