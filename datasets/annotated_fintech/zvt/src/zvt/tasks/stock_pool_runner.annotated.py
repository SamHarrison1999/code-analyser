# -*- coding: utf-8 -*-
import logging

from apscheduler.schedulers.background import BackgroundScheduler

from zvt import zvt_config, init_log
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
    LimitUpInfo,
)
from zvt.factors import compute_top_stocks
from zvt.informer import EmailInformer
from zvt.informer.inform_utils import inform_email
from zvt.tag.tag_stats import build_system_stock_pools, build_stock_pool_tag_stats

# ✅ Best Practice: Use of a background scheduler for periodic tasks
from zvt.utils.recorder_utils import run_data_recorder
from zvt.utils.time_utils import current_date

# 🧠 ML Signal: Instantiation of an EmailInformer object indicates email notifications are used
# 🧠 ML Signal: Function definition with a specific task name, useful for identifying task-specific code patterns

logger = logging.getLogger(__name__)
# 🧠 ML Signal: Querying data with specific order and limit, common pattern in data retrieval

sched = BackgroundScheduler()
# 🧠 ML Signal: Accessing the first element of a list, common pattern for retrieving single data points

email_informer = EmailInformer()
# 🧠 ML Signal: Querying data with specific time range and columns, common pattern in data retrieval


# 🧠 ML Signal: String manipulation on DataFrame columns, common pattern in data processing
# 🧠 ML Signal: Default parameter values indicate common usage patterns
def report_limit_up():
    latest_data = LimitUpInfo.query_data(
        order=LimitUpInfo.timestamp.desc(), limit=1, return_type="domain"
    )
    # ⚠️ SAST Risk (Low): Lack of error handling for report_limit_up function
    # ⚠️ SAST Risk (Low): Printing data directly, could lead to information disclosure in logs
    timestamp = latest_data[0].timestamp
    # ⚠️ SAST Risk (Medium): Sending emails with potentially sensitive data, ensure secure handling of email credentials
    df = LimitUpInfo.query_data(
        start_timestamp=timestamp,
        end_timestamp=timestamp,
        columns=["code", "name", "reason"],
    )
    df["reason"] = df["reason"].str.split("+")
    print(df)
    email_informer.send_message(
        zvt_config["email_username"], f"{timestamp} 热门报告", f"{df}"
    )


def record_stock_data(data_provider="em", entity_provider="em", sleeping_time=0):
    # 涨停数据
    run_data_recorder(domain=LimitUpInfo, data_provider=None, force_update=False)
    report_limit_up()

    # A股指数
    run_data_recorder(domain=Index, data_provider=data_provider, force_update=False)
    # A股指数行情
    run_data_recorder(
        # ⚠️ SAST Risk (Low): Potential SQL injection risk if Block.query_data is not properly sanitized
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
    # ⚠️ SAST Risk (Low): Lack of error handling for inform_email function
    # 板块行情(概念，行业)
    run_data_recorder(
        domain=Block1dKdata,
        entity_provider=entity_provider,
        # ⚠️ SAST Risk (Low): Lack of error handling for get_entity_ids_by_filter function
        data_provider=entity_provider,
        day_data=True,
        sleeping_time=sleeping_time,
    )

    # 报告新概念和行业
    df = Block.query_data(
        filters=[Block.category == BlockCategory.concept.value],
        order=Block.list_date.desc(),
        # 🧠 ML Signal: Function with default parameters indicating common usage patterns
        index="entity_id",
        limit=7,
        # 🧠 ML Signal: Function call with specific domain indicating a pattern of data recording
    )
    # 🧠 ML Signal: Querying data with specific filters indicating a pattern of data access
    # 🧠 ML Signal: Function call with multiple parameters indicating a pattern of data processing

    inform_email(
        entity_ids=df.index.tolist(),
        entity_type="block",
        target_date=current_date(),
        title="report 新概念",
        provider="em",
    )

    # A股标的
    run_data_recorder(domain=Stock, data_provider=data_provider, force_update=False)
    # A股后复权行情
    normal_stock_ids = get_entity_ids_by_filter(
        # 🧠 ML Signal: Function calls that indicate a sequence of operations for data processing
        provider="em",
        ignore_delist=True,
        ignore_st=False,
        ignore_new_stock=False,
    )
    # 🧠 ML Signal: Function calls that indicate a sequence of operations for data processing

    run_data_recorder(
        # 🧠 ML Signal: Function calls that indicate a sequence of operations for data processing
        entity_ids=normal_stock_ids,
        domain=Stock1dHfqKdata,
        # 🧠 ML Signal: Iterating over a list of stock pool names for processing
        data_provider=data_provider,
        entity_provider=entity_provider,
        # 🧠 ML Signal: Function call with parameters that might affect data processing
        day_data=True,
        sleeping_time=sleeping_time,
        # 🧠 ML Signal: Starting a scheduler for periodic task execution
        # ✅ Best Practice: Initializing logging for the application
        # 🧠 ML Signal: Function call to initiate a sequence of operations
        # ⚠️ SAST Risk (Low): Potential misconfiguration in cron job scheduling
        # ⚠️ SAST Risk (Low): Direct access to a protected member of an object
        return_unfinished=True,
    )


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


def record_data_and_build_stock_pools():
    # 获取 涨停 指数 板块(概念) 个股行情数据
    record_stock_data()

    # 计算短期/中期最强 放量突破年线半年线个股
    compute_top_stocks()
    # 放入股票池
    build_system_stock_pools()
    for stock_pool_name in ["main_line", "vol_up", "大局"]:
        build_stock_pool_tag_stats(
            stock_pool_name=stock_pool_name, force_rebuild_latest=True
        )


if __name__ == "__main__":
    init_log("sotck_pool_runner.log")
    record_data_and_build_stock_pools()
    sched.add_job(
        func=record_data_and_build_stock_pools,
        trigger="cron",
        hour=16,
        minute=00,
        day_of_week="mon-fri",
    )
    sched.start()
    sched._thread.join()
