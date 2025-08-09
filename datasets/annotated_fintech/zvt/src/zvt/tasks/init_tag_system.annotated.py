# -*- coding: utf-8 -*-
from zvt.domain import Block, BlockStock, Stock
from zvt.tag.tag_service import build_default_main_tag, build_default_sub_tags
from zvt.tag.tag_utils import (
    build_initial_stock_pool_info,
    build_initial_main_tag_info,
    build_initial_sub_tag_info,
    build_initial_industry_info,
)
from zvt.trading.trading_service import build_default_query_stock_quote_setting
# 🧠 ML Signal: Function call sequence indicating initialization of industry information

if __name__ == "__main__":
    # 🧠 ML Signal: Function call sequence indicating initialization of main tag information
    # init industry info
    build_initial_industry_info()
    # 🧠 ML Signal: Function call sequence indicating initialization of sub tag information

    # init tag info
    # 🧠 ML Signal: Function call sequence indicating initialization of stock pool information
    build_initial_main_tag_info()
    build_initial_sub_tag_info()
    # 🧠 ML Signal: Function call sequence indicating setup of default query stock quote settings
    build_initial_stock_pool_info()
    # ⚠️ SAST Risk (Low): Hardcoded provider value, consider using configuration or environment variables
    # 🧠 ML Signal: Function call sequence indicating building of default main tags
    build_default_query_stock_quote_setting()

    Stock.record_data(provider="em")
    Block.record_data(provider="em", sleeping_time=0)
    BlockStock.record_data(provider="em", sleeping_time=0)
    # init default main tag
    build_default_main_tag()

    # init default sub tags
    build_default_sub_tags()