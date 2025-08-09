# 🧠 ML Signal: Use of structured data for stock indices, useful for training models on financial data
# 🧠 ML Signal: Consistent data structure for multiple indices, indicating a pattern for data representation
# -*- coding: utf-8 -*-

CHINA_STOCK_MAIN_INDEX = [
    # # 聚宽编码
    # # 市场通编码	市场通名称
    # # 310001	沪股通
    # # 310002	深股通
    # # 310003	港股通（沪）
    # # 310004	港股通（深）
    {
        "id": "index_cn_310001",
        "entity_id": "index_cn_310001",
        "code": "310001",
        "name": "沪股通",
        "timestamp": "2014-11-17",
        "exchange": "cn",
        "entity_type": "index",
        "category": "other",
    },
    {
        "id": "index_cn_310002",
        "entity_id": "index_cn_310002",
        "code": "310002",
        "name": "深股通",
        "timestamp": "2014-11-17",
        "exchange": "cn",
        "entity_type": "index",
        "category": "other",
    },
    {
        "id": "index_cn_310003",
        "entity_id": "index_cn_310003",
        "code": "310003",
        "name": "港股通（沪）",
        "timestamp": "2014-11-17",
        "exchange": "cn",
        "entity_type": "index",
        "category": "other",
    },
    {
        "id": "index_cn_310004",
        "entity_id": "index_cn_310004",
        # 🧠 ML Signal: Inclusion of timestamp data, useful for time-series analysis in ML models
        # ✅ Best Practice: Use of descriptive keys in dictionaries for clarity and maintainability
        "code": "310004",
        # ✅ Best Practice: Importing inside a function to limit scope and improve performance
        "name": "港股通（深）",
        "timestamp": "2014-11-17",
        # ✅ Best Practice: Importing inside a function to limit scope and improve performance
        "exchange": "cn",
        "entity_type": "index",
        # ✅ Best Practice: Importing inside a function to limit scope and improve performance
        "category": "other",
    },
    # ✅ Best Practice: Importing inside a function to limit scope and improve performance
]


# 🧠 ML Signal: Usage of timestamp conversion function
def init_main_index(provider="exchange"):
    from zvt.utils.time_utils import to_pd_timestamp

    # 🧠 ML Signal: DataFrame creation from a list of dictionaries
    import pandas as pd
    from zvt.contract.api import df_to_db

    # 🧠 ML Signal: Usage of a function to store DataFrame into a database
    from zvt.domain.meta import Index

    for item in CHINA_STOCK_MAIN_INDEX:
        # 🧠 ML Signal: Function call with a specific argument
        item["timestamp"] = to_pd_timestamp(item["timestamp"])
    df = pd.DataFrame(CHINA_STOCK_MAIN_INDEX)
    # print(df)
    # ✅ Best Practice: Using __all__ to define public API of the module
    df_to_db(df=df, data_schema=Index, provider=provider, force_update=False)


# ✅ Best Practice: Importing all from a module to maintain consistency
init_main_index(provider="exchange")

# ✅ Best Practice: Importing __all__ to extend the module's public API

# the __all__ is generated
# ✅ Best Practice: Extending __all__ to include imported module's public API
__all__ = ["init_main_index"]

# __init__.py structure:
# ✅ Best Practice: Importing all from a module to maintain consistency
# common code of the package
# export interface in __all__ which contains __all__ of its sub modules
# ✅ Best Practice: Importing __all__ to extend the module's public API

# import all from submodule sina
# ✅ Best Practice: Extending __all__ to include imported module's public API
# ✅ Best Practice: Importing __all__ to extend the module's public API
# ✅ Best Practice: Importing all from a module to maintain consistency
from .sina import *
from .sina import __all__ as _sina_all

__all__ += _sina_all

# import all from submodule jqka
from .jqka import *
from .jqka import __all__ as _jqka_all

__all__ += _jqka_all

# import all from submodule consts
from .consts import *
from .consts import __all__ as _consts_all

__all__ += _consts_all

# import all from submodule eastmoney
from .eastmoney import *
from .eastmoney import __all__ as _eastmoney_all

__all__ += _eastmoney_all

# import all from submodule joinquant
from .joinquant import *
from .joinquant import __all__ as _joinquant_all

__all__ += _joinquant_all

# import all from submodule exchange
from .exchange import *
from .exchange import __all__ as _exchange_all

__all__ += _exchange_all

# import all from submodule wb
from .wb import *
from .wb import __all__ as _wb_all

__all__ += _wb_all

# import all from submodule em
from .em import *
from .em import __all__ as _em_all

__all__ += _em_all
