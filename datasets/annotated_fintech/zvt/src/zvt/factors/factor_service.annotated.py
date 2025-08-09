# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping related imports together improves readability and maintainability.
import pandas as pd

from zvt.contract import zvt_context
from zvt.domain import Stock
from zvt.factors.factor_models import FactorRequestModel
# 🧠 ML Signal: Accessing attributes of a model object, indicating a pattern of object-oriented design.
from zvt.factors.technical_factor import TechnicalFactor
from zvt.trader import TradingSignalType
# 🧠 ML Signal: Accessing attributes of a model object, indicating a pattern of object-oriented design.


# 🧠 ML Signal: Accessing attributes of a model object, indicating a pattern of object-oriented design.
# 🧠 ML Signal: Usage of a registry pattern to retrieve a class, indicating dynamic class instantiation.
# ⚠️ SAST Risk (Medium): Potential risk of code injection if factor_name is not validated.
def query_factor_result(factor_request_model: FactorRequestModel):
    factor_name = factor_request_model.factor_name
    entity_ids = factor_request_model.entity_ids
    level = factor_request_model.level

    factor: TechnicalFactor = zvt_context.factor_cls_registry[factor_name](
        provider="em",
        entity_provider="em",
        entity_schema=Stock,
        entity_ids=entity_ids,
        # 🧠 ML Signal: Accessing attributes of a model object, indicating a pattern of object-oriented design.
        level=level,
        # ✅ Best Practice: Check for None explicitly to handle null values
        start_timestamp=factor_request_model.start_timestamp,
    )
    # 🧠 ML Signal: Method call on an object, indicating a pattern of object-oriented design.
    df = factor.get_trading_signal_df()
    # ✅ Best Practice: Simplify condition by directly returning the result
    df = df.reset_index(drop=False)
    # ✅ Best Practice: Explicitly resetting index with drop=False to retain the old index as a column.

    def to_trading_signal(order_type):
        # ✅ Best Practice: The 'if not order_type' is redundant here, consider using 'else'
        if order_type is None:
            return None
        if order_type:
            # ✅ Best Practice: Use descriptive column names for better readability
            return TradingSignalType.open_long
        if not order_type:
            # ⚠️ SAST Risk (Low): Printing data frames can expose sensitive data in logs
            # 🧠 ML Signal: Converting DataFrame to a list of dictionaries for further processing
            # ✅ Best Practice: Use __all__ to define public API of the module
            # 🧠 ML Signal: Feature engineering by creating new columns based on existing data
            # 🧠 ML Signal: Applying a function to a DataFrame column to transform data
            return TradingSignalType.close_long

    df = df.rename(columns={"timestamp": "happen_timestamp"})
    df["due_timestamp"] = df["happen_timestamp"] + pd.Timedelta(seconds=level.to_second())
    df["trading_signal_type"] = df["filter_result"].apply(lambda x: to_trading_signal(x))

    print(df)
    return df.to_dict(orient="records")


# the __all__ is generated
__all__ = ["query_factor_result"]