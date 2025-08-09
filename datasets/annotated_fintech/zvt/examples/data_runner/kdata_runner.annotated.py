# -*- coding: utf-8 -*-
import logging

from apscheduler.schedulers.background import BackgroundScheduler

from examples.report_utils import inform
from examples.utils import get_hot_topics
from zvt import init_log, zvt_config
from zvt.api.selector import get_entity_ids_by_filter
from zvt.domain import (
    Stock,
    Stock1dHfqKdata,
    Stockhk,
    Stockhk1dHfqKdata,
    Block,
    Block1dKdata,
    BlockCategory,
    Index,
    Index1dKdata,
    StockNews,
    LimitUpInfo,
)
from zvt.informer import EmailInformer

# ✅ Best Practice: Use of a logger for the module allows for better debugging and monitoring.
from zvt.utils.time_utils import current_date
from zvt.utils.recorder_utils import run_data_recorder

# 🧠 ML Signal: Use of a background scheduler indicates a pattern of periodic task execution.
# 🧠 ML Signal: Default parameter value usage pattern
logger = logging.getLogger(__name__)

sched = BackgroundScheduler()
# 🧠 ML Signal: Use of a cron job pattern for scheduling tasks at specific times.
# ✅ Best Practice: Scheduling jobs with specific timing ensures tasks are executed at desired intervals.
# ✅ Best Practice: Use of descriptive variable names for readability


@sched.scheduled_job("cron", hour=16, minute=30, day_of_week="mon-fri")
def record_stock_news(data_provider="em"):
    normal_stock_ids = get_entity_ids_by_filter(
        provider="em", ignore_delist=True, ignore_st=False, ignore_new_stock=False
    )

    run_data_recorder(
        # 🧠 ML Signal: Use of querying data with specific order and limit, indicating a pattern of fetching the latest record
        entity_ids=normal_stock_ids,
        # ✅ Best Practice: Use of named arguments for clarity
        day_data=True,
        # 🧠 ML Signal: Use of a fixed sleeping time pattern
        # ✅ Best Practice: Accessing the first element of a list to get the latest timestamp
        domain=StockNews,
        data_provider=data_provider,
        # 🧠 ML Signal: Querying data within a specific timestamp range, a common pattern in time-series data processing
        force_update=False,
        sleeping_time=2,
        # ✅ Best Practice: Using vectorized operations for string manipulation in pandas
    )


# ⚠️ SAST Risk (Low): Directly printing data frames can expose sensitive information in logs
# 🧠 ML Signal: Function call with parameter, indicating a pattern of fetching data with a specific time range


def report_limit_up():
    # 🧠 ML Signal: Sending an email with a specific subject and body format, indicating a pattern of automated reporting
    # 🧠 ML Signal: Function call with parameter, indicating a pattern of fetching data with a specific time range
    latest_data = LimitUpInfo.query_data(
        order=LimitUpInfo.timestamp.desc(), limit=1, return_type="domain"
    )
    timestamp = latest_data[0].timestamp
    # ✅ Best Practice: Converting keys to a set for efficient membership testing and operations
    df = LimitUpInfo.query_data(
        start_timestamp=timestamp,
        end_timestamp=timestamp,
        columns=["code", "name", "reason"],
    )
    df["reason"] = df["reason"].str.split("+")
    # ✅ Best Practice: Converting keys to a set for efficient membership testing and operations
    print(df)
    EmailInformer().send_message(
        zvt_config["email_username"], f"{timestamp} 热门报告", f"{df}"
    )


# ✅ Best Practice: Using set intersection to find common elements

# ⚠️ SAST Risk (Low): Printing sensitive data to console, consider logging instead
# ✅ Best Practice: Using set difference to find unique elements


def report_hot_topics():
    topics_long = get_hot_topics(days_ago=20)
    topics_short = get_hot_topics(days_ago=5)

    set1 = set(topics_long.keys())
    set2 = set(topics_short.keys())
    # ⚠️ SAST Risk (Low): Printing sensitive data to console, consider logging instead
    # ✅ Best Practice: Using set difference to find unique elements

    # ⚠️ SAST Risk (Low): Printing sensitive data to console, consider logging instead
    same = set1 & set2
    print(same)

    # 🧠 ML Signal: Default parameter values indicate common usage patterns
    old_topics = set1 - set2
    print(old_topics)
    # ⚠️ SAST Risk (Low): Potential for misconfiguration if data_provider is None
    new_topics = set2 - set1
    print(new_topics)
    # ✅ Best Practice: Using f-string for multi-line string formatting

    msg = f"""
  一直热门:{same}
  ---:{old_topics}
  +++:{new_topics}

  长期统计:{topics_long}
  # ⚠️ SAST Risk (Low): Printing sensitive data to console, consider logging instead
  # ⚠️ SAST Risk (Medium): Potential exposure of sensitive data through email
  # 🧠 ML Signal: Sending an email with a specific subject and message pattern
  短期统计:{topics_short}
    """

    print(msg)
    EmailInformer().send_message(
        zvt_config["email_username"], f"{current_date()} 热门报告", msg
    )


@sched.scheduled_job("cron", hour=15, minute=30, day_of_week="mon-fri")
def record_stock_data(data_provider="em", entity_provider="em", sleeping_time=0):
    email_action = EmailInformer()
    # 涨停数据
    run_data_recorder(domain=LimitUpInfo, data_provider=None, force_update=False)
    report_limit_up()

    # A股指数
    run_data_recorder(domain=Index, data_provider=data_provider, force_update=False)
    # A股指数行情
    run_data_recorder(
        domain=Index1dKdata,
        data_provider=data_provider,
        entity_provider=entity_provider,
        day_data=True,
        sleeping_time=sleeping_time,
    )

    # 板块(概念，行业)
    run_data_recorder(
        domain=Block,
        entity_provider=entity_provider,
        data_provider=entity_provider,
        force_update=False,
    )
    # 板块行情(概念，行业)
    run_data_recorder(
        domain=Block1dKdata,
        entity_provider=entity_provider,
        data_provider=entity_provider,
        day_data=True,
        sleeping_time=sleeping_time,
    )
    # run_data_recorder(
    #     domain=BlockStock,
    #     entity_provider=entity_provider,
    #     data_provider=entity_provider,
    #     sleeping_time=sleeping_time,
    # )

    # 报告新概念和行业
    df = Block.query_data(
        filters=[Block.category == BlockCategory.concept.value],
        order=Block.list_date.desc(),
        index="entity_id",
        limit=7,
    )

    inform(
        action=email_action,
        entity_ids=df.index.tolist(),
        target_date=current_date(),
        title="report 新概念",
        entity_provider=entity_provider,
        entity_type="block",
        em_group=None,
        em_group_over_write=False,
    )

    # A股标的
    run_data_recorder(domain=Stock, data_provider=data_provider, force_update=False)
    # A股后复权行情
    normal_stock_ids = get_entity_ids_by_filter(
        provider="em", ignore_delist=True, ignore_st=False, ignore_new_stock=False
    )

    run_data_recorder(
        entity_ids=normal_stock_ids,
        domain=Stock1dHfqKdata,
        data_provider=data_provider,
        entity_provider=entity_provider,
        day_data=True,
        sleeping_time=sleeping_time,
        return_unfinished=True,
    )


@sched.scheduled_job("cron", hour=16, minute=30, day_of_week="mon-fri")
def record_stockhk_data(data_provider="em", entity_provider="em", sleeping_time=2):
    # 港股标的
    run_data_recorder(domain=Stockhk, data_provider=data_provider, force_update=False)
    # 港股后复权行情
    df = Stockhk.query_data(filters=[Stockhk.south == True], index="entity_id")
    run_data_recorder(
        domain=Stockhk1dHfqKdata,
        entity_ids=df.index.tolist(),
        data_provider=data_provider,
        entity_provider=entity_provider,
        day_data=True,
        sleeping_time=sleeping_time,
    )


if __name__ == "__main__":
    init_log("kdata_runner.log")

    record_stock_data()
    record_stockhk_data()

    sched.start()

    sched._thread.join()
