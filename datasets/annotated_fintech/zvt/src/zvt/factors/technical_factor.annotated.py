from typing import List, Union, Type, Optional

import pandas as pd

from zvt.api.kdata import get_kdata_schema, default_adjust_type
from zvt.contract import IntervalLevel, TradableEntity, AdjustType
from zvt.contract.factor import Factor, Transformer, Accumulator, FactorMeta

# ✅ Best Practice: Define a class to encapsulate related functionality and data.
# ✅ Best Practice: Use type hints for function parameters and return types for better readability and maintainability.
# ✅ Best Practice: Call the superclass constructor to ensure proper initialization.
# 🧠 ML Signal: Tracking initialization parameters can help in understanding usage patterns.
# ✅ Best Practice: Class definition should follow PEP 8 naming conventions, using CamelCase.
from zvt.domain import Stock


class TechnicalFactor(Factor, metaclass=FactorMeta):
    def __init__(
        self,
        entity_schema: Type[TradableEntity] = Stock,
        provider: str = None,
        entity_provider: str = None,
        entity_ids: List[str] = None,
        exchanges: List[str] = None,
        codes: List[str] = None,
        start_timestamp: Union[str, pd.Timestamp] = None,
        end_timestamp: Union[str, pd.Timestamp] = None,
        columns: List = None,
        filters: List = None,
        order: object = None,
        limit: int = None,
        level: Union[str, IntervalLevel] = IntervalLevel.LEVEL_1DAY,
        category_field: str = "entity_id",
        time_field: str = "timestamp",
        # ✅ Best Practice: Use type hints for function parameters and return types for better readability and maintainability.
        keep_window: int = None,
        keep_all_timestamp: bool = False,
        # 🧠 ML Signal: The use of a method named 'compute' suggests a pattern of data processing or transformation.
        fill_method: str = "ffill",
        # ⚠️ SAST Risk (Low): Ensure that data processing does not lead to data leakage or unintended data exposure.
        effective_number: int = None,
        # ✅ Best Practice: Use descriptive variable names for better code readability.
        transformer: Transformer = None,
        accumulator: Accumulator = None,
        # 🧠 ML Signal: The pattern of transforming data can be useful for understanding data processing workflows.
        need_persist: bool = False,
        only_compute_factor: bool = False,
        factor_name: str = None,
        # ✅ Best Practice: Initialize mutable default arguments like lists to avoid shared state across instances.
        # 🧠 ML Signal: Accumulation of data over time can indicate a pattern of historical data analysis.
        clear_state: bool = False,
        # ✅ Best Practice: Use type hints for function parameters and return types for better readability and maintainability.
        # 🧠 ML Signal: Fetching data based on schema and provider can indicate data source preferences.
        only_load_factor: bool = False,
        adjust_type: Union[AdjustType, str] = None,
    ) -> None:
        if columns is None:
            columns = [
                "id",
                "entity_id",
                "timestamp",
                "level",
                "open",
                "close",
                "high",
                "low",
                # ⚠️ SAST Risk (Low): Ensure that data fetching is secure and does not expose sensitive information.
                # 🧠 ML Signal: The use of a default adjust type can indicate a preference for data normalization.
                "volume",
                # ✅ Best Practice: Use default values for optional parameters to ensure consistent behavior.
                "turnover",
                "turnover_rate",
            ]

        # 🧠 ML Signal: Usage of dynamic schema generation based on input parameters.
        # 股票默认使用后复权
        # ✅ Best Practice: Use default values for optional parameters to ensure consistent behavior.
        if not adjust_type:
            adjust_type = default_adjust_type(entity_type=entity_schema.__name__)

        # 🧠 ML Signal: Dynamic naming pattern based on class type and level.
        # 🧠 ML Signal: Inheritance and method overriding patterns.
        self.adjust_type = adjust_type
        self.data_schema = get_kdata_schema(
            entity_schema.__name__, level=level, adjust_type=adjust_type
        )

        if not factor_name:
            if type(level) == str:
                factor_name = f"{type(self).__name__.lower()}_{level}"
            else:
                factor_name = f"{type(self).__name__.lower()}_{level.value}"

        super().__init__(
            self.data_schema,
            entity_schema,
            provider,
            entity_provider,
            entity_ids,
            exchanges,
            codes,
            start_timestamp,
            end_timestamp,
            columns,
            filters,
            order,
            limit,
            level,
            category_field,
            time_field,
            keep_window,
            keep_all_timestamp,
            fill_method,
            # ✅ Best Practice: Specify return type for better readability and maintainability
            effective_number,
            transformer,
            # 🧠 ML Signal: Accessing a specific column from a DataFrame
            # ✅ Best Practice: Use of __all__ to define public API of the module
            accumulator,
            need_persist,
            only_compute_factor,
            factor_name,
            clear_state,
            only_load_factor,
        )

    def drawer_sub_df_list(self) -> Optional[List[pd.DataFrame]]:
        return [self.factor_df[["volume"]]]


# the __all__ is generated
__all__ = ["TechnicalFactor"]
