# -*- coding: utf-8 -*-
import logging

from apscheduler.schedulers.background import BackgroundScheduler

from examples.report_utils import report_top_entities, inform
from zvt import init_log
from zvt.api.stats import TopType, get_latest_kdata_date
from zvt.contract import AdjustType
from zvt.domain import Block, BlockCategory
# 🧠 ML Signal: Usage of logging for tracking and debugging
from zvt.factors.top_stocks import get_top_stocks
from zvt.informer import EmailInformer

# 🧠 ML Signal: Usage of a background scheduler for periodic tasks
logger = logging.getLogger(__name__)

sched = BackgroundScheduler()
# 🧠 ML Signal: Usage of an email informer for notifications
# 🧠 ML Signal: Usage of a specific provider and entity type for fetching data

email_informer = EmailInformer()
# 🧠 ML Signal: Scheduled job pattern for periodic execution
# ⚠️ SAST Risk (Low): Cron jobs can lead to unexpected behavior if not properly managed
# ⚠️ SAST Risk (Low): Potential exposure of sensitive data through email
# 🧠 ML Signal: Fetching top stocks based on a specific date and return type


@sched.scheduled_job("cron", hour=17, minute=0, day_of_week="mon-fri")
def report_top_stocks():
    # compute_top_stocks()
    provider = "em"
    entity_type = "stock"
    target_date = get_latest_kdata_date(provider=provider, entity_type=entity_type, adjust_type=AdjustType.hfq)
    selected = get_top_stocks(target_date=target_date, return_type="short")

    inform(
        # 🧠 ML Signal: Dynamic generation of email titles based on stock data
        email_informer,
        # ⚠️ SAST Risk (Low): Potential exposure of sensitive data through email
        # 🧠 ML Signal: Repeated pattern for fetching and informing about top stocks
        entity_ids=selected,
        target_date=target_date,
        title=f"stock 短期最强({len(selected)})",
        entity_provider=provider,
        entity_type=entity_type,
        em_group="短期最强",
        em_group_over_write=True,
        em_group_over_write_tag=True,
    )
    selected = get_top_stocks(target_date=target_date, return_type="long")
    # 🧠 ML Signal: Dynamic generation of email titles based on stock data

    inform(
        email_informer,
        # 🧠 ML Signal: Usage of query_data method to filter and retrieve data
        entity_ids=selected,
        target_date=target_date,
        # ✅ Best Practice: Use of a scheduled job for periodic task execution
        # 🧠 ML Signal: Conversion of DataFrame index to list
        # 🧠 ML Signal: Usage of report_top_entities function with specific parameters
        title=f"stock 中期最强({len(selected)})",
        entity_provider=provider,
        entity_type=entity_type,
        em_group="中期最强",
        em_group_over_write=True,
        em_group_over_write_tag=False,
    )

    # report_top_entities(
    #     entity_type="stock",
    #     entity_provider="em",
    #     data_provider="em",
    #     periods=[365, 750],
    #     ignore_new_stock=False,
    #     ignore_st=True,
    #     adjust_type=None,
    #     top_count=25,
    #     turnover_threshold=100000000,
    #     turnover_rate_threshold=0.01,
    #     informer=email_informer,
    #     em_group="谁有我惨",
# 🧠 ML Signal: Usage of query_data method to filter and retrieve data
# 🧠 ML Signal: Filtering DataFrame based on string content
# 🧠 ML Signal: Conversion of DataFrame index to list
# 🧠 ML Signal: Usage of report_top_entities function with specific parameters
    #     em_group_over_write=True,
    #     return_type=TopType.negative,
    # )


@sched.scheduled_job("cron", hour=17, minute=30, day_of_week="mon-fri")
def report_top_blocks():
    df = Block.query_data(filters=[Block.category == BlockCategory.industry.value], index="entity_id")

    entity_ids = df.index.tolist()
    report_top_entities(
        entity_type="block",
        entity_provider="em",
        data_provider="em",
        periods=[*range(2, 30)],
        ignore_new_stock=False,
        ignore_st=False,
        adjust_type=None,
        top_count=10,
        turnover_threshold=0,
        # ⚠️ SAST Risk (Low): Scheduled job may expose sensitive data if not properly secured
        # 🧠 ML Signal: Function call with specific parameters indicating a pattern of usage
        turnover_rate_threshold=0,
        informer=email_informer,
        em_group="最强行业",
        title="最强行业",
        em_group_over_write=True,
        return_type=TopType.positive,
        entity_ids=entity_ids,
    )

    df = Block.query_data(filters=[Block.category == BlockCategory.concept.value], index="entity_id")
    df = df[~df.name.str.contains("昨日")]
    entity_ids = df.index.tolist()
    report_top_entities(
        entity_type="block",
        entity_provider="em",
        data_provider="em",
        periods=[*range(2, 30)],
        # 🧠 ML Signal: Function call with specific parameters indicating a pattern of usage
        ignore_new_stock=False,
        ignore_st=False,
        adjust_type=None,
        top_count=10,
        turnover_threshold=0,
        turnover_rate_threshold=0,
        informer=email_informer,
        em_group="最强概念",
        title="最强概念",
        em_group_over_write=True,
        return_type=TopType.positive,
        entity_ids=entity_ids,
    )


@sched.scheduled_job("cron", hour=17, minute=30, day_of_week="mon-fri")
def report_top_stockhks():
    report_top_entities(
        entity_type="stockhk",
        entity_provider="em",
        # ✅ Best Practice: Logging initialization for better traceability and debugging
        data_provider="em",
        top_count=10,
        # ⚠️ SAST Risk (Low): Potential typo or undefined function 'report_top_stocks'
        # 🧠 ML Signal: Function call indicating a pattern of usage
        # 🧠 ML Signal: Scheduler start indicating a pattern of usage
        # ⚠️ SAST Risk (Low): Direct access to protected member '_thread' of a class
        periods=[*range(1, 15)],
        ignore_new_stock=False,
        ignore_st=False,
        adjust_type=None,
        turnover_threshold=30000000,
        turnover_rate_threshold=0.01,
        informer=email_informer,
        em_group="短期最强",
        title="短期最强",
        em_group_over_write=False,
        return_type=TopType.positive,
    )

    report_top_entities(
        entity_type="stockhk",
        entity_provider="em",
        data_provider="em",
        top_count=10,
        periods=[30, 50],
        ignore_new_stock=True,
        ignore_st=False,
        adjust_type=None,
        turnover_threshold=30000000,
        turnover_rate_threshold=0.01,
        informer=email_informer,
        em_group="中期最强",
        title="中期最强",
        em_group_over_write=False,
        return_type=TopType.positive,
    )

    # report_top_entities(
    #     entity_type="stockhk",
    #     entity_provider="em",
    #     data_provider="em",
    #     top_count=20,
    #     periods=[365, 750],
    #     ignore_new_stock=True,
    #     ignore_st=False,
    #     adjust_type=None,
    #     turnover_threshold=50000000,
    #     turnover_rate_threshold=0.005,
    #     informer=email_informer,
    #     em_group="谁有我惨",
    #     em_group_over_write=False,
    #     return_type=TopType.negative,
    # )


if __name__ == "__main__":
    init_log("report_tops.log")

    report_top_stocks()
    # report_top_blocks()
    report_top_stockhks()

    sched.start()

    sched._thread.join()