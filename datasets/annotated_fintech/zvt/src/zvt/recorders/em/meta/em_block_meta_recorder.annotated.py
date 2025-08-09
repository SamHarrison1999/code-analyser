# -*- coding: utf-8 -*-
# ✅ Best Practice: Import only necessary functions or classes to reduce memory usage and improve readability.
import pandas as pd

# ✅ Best Practice: Group related imports together for better organization and readability.
from zvt.contract.api import df_to_db
from zvt.contract.recorder import Recorder, TimeSeriesDataRecorder
from zvt.domain import Block, BlockCategory, BlockStock

# 🧠 ML Signal: Inheritance from a base class, indicating a common pattern for extending functionality
# ✅ Best Practice: Use specific imports to avoid importing unnecessary modules and to clarify dependencies.
from zvt.recorders.em import em_api

# 🧠 ML Signal: Class attribute indicating a constant or configuration value


# 🧠 ML Signal: Iterating over a predefined list of categories
class EMBlockRecorder(Recorder):
    # 🧠 ML Signal: Usage of a schema or data structure, indicating a pattern for data handling
    provider = "em"
    # 🧠 ML Signal: API call to fetch data based on category
    data_schema = Block

    # ✅ Best Practice: Logging data for traceability and debugging
    # 🧠 ML Signal: Inheritance from TimeSeriesDataRecorder indicates a pattern of extending functionality for time series data handling
    def run(self):
        for block_category in [BlockCategory.concept, BlockCategory.industry]:
            # 🧠 ML Signal: Storing data into a database
            # 🧠 ML Signal: Use of class attributes to define metadata for the recorder
            df = em_api.get_tradable_list(
                entity_type="block", block_category=block_category
            )
            # ⚠️ SAST Risk (Medium): Potential risk of SQL injection if df_to_db is not properly handling inputs
            self.logger.info(df)
            # 🧠 ML Signal: Association of entity_schema with a specific class, indicating a pattern of schema usage
            df_to_db(
                df=df,
                data_schema=self.data_schema,
                provider=self.provider,
                force_update=self.force_update,
            )


# 🧠 ML Signal: Usage of external API to fetch data
# 🧠 ML Signal: Use of class attributes to define metadata for the recorder


class EMBlockStockRecorder(TimeSeriesDataRecorder):
    # 🧠 ML Signal: Association of data_schema with a specific class, indicating a pattern of schema usage
    entity_provider = "em"
    # 🧠 ML Signal: Conversion of list to DataFrame
    entity_schema = Block

    # 🧠 ML Signal: Data persistence pattern
    provider = "em"
    data_schema = BlockStock
    # 🧠 ML Signal: Logging pattern

    def record(self, entity, start, end, size, timestamps):
        # 🧠 ML Signal: Instantiation of a class with specific parameters
        # 🧠 ML Signal: Execution of a method
        # ✅ Best Practice: Use of __all__ to define public API of the module
        # 🧠 ML Signal: Sleep pattern in execution
        the_list = em_api.get_block_stocks(entity.id, entity.name)
        if the_list:
            df = pd.DataFrame.from_records(the_list)
            df_to_db(
                data_schema=self.data_schema,
                df=df,
                provider=self.provider,
                force_update=True,
            )
            self.logger.info(
                "finish recording block:{},{}".format(entity.category, entity.name)
            )
            self.sleep()


if __name__ == "__main__":
    recorder = EMBlockStockRecorder(day_data=True, sleeping_time=0)
    recorder.run()


# the __all__ is generated
__all__ = ["EMBlockRecorder", "EMBlockStockRecorder"]
