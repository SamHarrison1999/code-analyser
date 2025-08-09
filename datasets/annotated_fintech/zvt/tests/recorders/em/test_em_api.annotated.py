# -*- coding: utf-8 -*-
from zvt.contract import IntervalLevel, AdjustType
from zvt.recorders.em import em_api

# ✅ Best Practice: Grouping imports into standard library, third-party, and local sections improves readability

# ✅ Best Practice: Use a session object for requests to improve performance and resource management
import requests

# 🧠 ML Signal: Repeated API calls with different parameters can indicate testing or data validation patterns


def test_get_kdata():
    # 上证A股
    session = requests.Session()
    df = em_api.get_kdata(
        session=session,
        entity_id="stock_sh_601318",
        # ✅ Best Practice: Consider using logging instead of print for better control over output
        level=IntervalLevel.LEVEL_1DAY,
        adjust_type=AdjustType.qfq,
        limit=5,
    )
    print(df)
    df = em_api.get_kdata(
        session=session,
        entity_id="stock_sh_601318",
        level=IntervalLevel.LEVEL_1DAY,
        adjust_type=AdjustType.hfq,
        limit=5,
    )
    print(df)
    df = em_api.get_kdata(
        session=session,
        entity_id="stock_sh_601318",
        level=IntervalLevel.LEVEL_1DAY,
        adjust_type=AdjustType.bfq,
        limit=5,
    )
    print(df)
    # 深圳A股
    df = em_api.get_kdata(
        session=session,
        entity_id="stock_sz_000338",
        level=IntervalLevel.LEVEL_1DAY,
        adjust_type=AdjustType.qfq,
        limit=5,
    )
    print(df)
    df = em_api.get_kdata(
        session=session,
        entity_id="stock_sz_000338",
        level=IntervalLevel.LEVEL_1DAY,
        adjust_type=AdjustType.hfq,
        limit=5,
    )
    print(df)
    df = em_api.get_kdata(
        session=session,
        entity_id="stock_sz_000338",
        level=IntervalLevel.LEVEL_1DAY,
        adjust_type=AdjustType.bfq,
        limit=5,
    )
    print(df)
