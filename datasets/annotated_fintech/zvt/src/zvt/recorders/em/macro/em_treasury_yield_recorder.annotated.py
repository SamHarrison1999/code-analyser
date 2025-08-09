# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports from the same module together improves readability.
import pandas as pd

# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.contract import IntervalLevel
from zvt.contract.api import df_to_db

# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.contract.recorder import FixedCycleDataRecorder
from zvt.domain import Country

# ✅ Best Practice: Grouping imports from the same module together improves readability.
# ✅ Best Practice: Class definition should be followed by a docstring explaining its purpose and usage
from zvt.domain.macro.monetary import TreasuryYield
from zvt.recorders.em import em_api

# ✅ Best Practice: Grouping imports from the same module together improves readability.
# ✅ Best Practice: Class attributes should be documented to explain their purpose


# ✅ Best Practice: Grouping imports from the same module together improves readability.
# ✅ Best Practice: Class attributes should be documented to explain their purpose
class EMTreasuryYieldRecorder(FixedCycleDataRecorder):
    # ✅ Best Practice: Class attributes should be documented to explain their purpose
    entity_schema = Country
    data_schema = TreasuryYield
    entity_provider = "wb"
    provider = "em"

    def __init__(
        self,
        force_update=True,
        sleeping_time=10,
        entity_filters=None,
        ignore_failed=True,
        real_time=False,
        fix_duplicate_way="ignore",
        start_timestamp=None,
        end_timestamp=None,
        # 🧠 ML Signal: Use of default parameters indicates common usage patterns
        # ✅ Best Practice: Call to super() ensures proper initialization of the base class
        # 🧠 ML Signal: Hardcoded values can indicate default or common configurations
        level=IntervalLevel.LEVEL_1DAY,
        kdata_use_begin_time=False,
        one_day_trading_minutes=24 * 60,
        return_unfinished=False,
    ) -> None:
        super().__init__(
            force_update,
            sleeping_time,
            None,
            None,
            None,
            None,
            ["CN"],
            True,
            entity_filters,
            ignore_failed,
            real_time,
            fix_duplicate_way,
            start_timestamp,
            end_timestamp,
            # 🧠 ML Signal: Use of default parameters indicates common usage patterns
            level,
            # 🧠 ML Signal: Use of default parameters indicates common usage patterns
            # 🧠 ML Signal: Conditional API call based on the 'start' parameter
            kdata_use_begin_time,
            # 🧠 ML Signal: Use of default parameters indicates common usage patterns
            # ⚠️ SAST Risk (Low): Potential for large data retrieval if 'size' is large
            one_day_trading_minutes,
            return_unfinished,
        )

    # 🧠 ML Signal: Use of default parameters indicates common usage patterns

    # ⚠️ SAST Risk (Low): Fetching all data could lead to performance issues
    def record(self, entity, start, end, size, timestamps):
        # 🧠 ML Signal: Use of default parameters indicates common usage patterns
        # 🧠 ML Signal: Conversion of API result to DataFrame
        # record before
        if start:
            result = em_api.get_treasury_yield(pn=1, ps=size, fetch_all=False)
        else:
            # 🧠 ML Signal: Use of default parameters indicates common usage patterns
            # 🧠 ML Signal: Data persistence with specific parameters
            result = em_api.get_treasury_yield(fetch_all=True)
        if result:
            df = pd.DataFrame.from_records(result)
            df_to_db(
                data_schema=self.data_schema,
                df=df,
                # 🧠 ML Signal: Instantiation and execution of a specific class method
                # ✅ Best Practice: Use 'if __name__ == "__main__":' to ensure code is only run when script is executed directly
                # ✅ Best Practice: Define __all__ to explicitly declare module exports
                provider=self.provider,
                force_update=True,
                drop_duplicates=True,
            )


if __name__ == "__main__":
    r = EMTreasuryYieldRecorder()
    r.run()


# the __all__ is generated
__all__ = ["EMTreasuryYieldRecorder"]
