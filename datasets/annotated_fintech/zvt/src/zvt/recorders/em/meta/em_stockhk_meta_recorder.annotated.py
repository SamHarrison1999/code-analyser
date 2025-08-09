# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports from the same module together improves readability.

from zvt.contract.api import df_to_db
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.contract.recorder import Recorder
# ✅ Best Practice: Class definition should follow PEP 8 naming conventions, using CamelCase for class names.
from zvt.domain.meta.stockhk_meta import Stockhk
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.recorders.em import em_api
# ✅ Best Practice: Class attributes should be defined at the top of the class for better readability.


# ✅ Best Practice: Class attributes should be defined at the top of the class for better readability.
# 🧠 ML Signal: Usage of external API to fetch data
class EMStockhkRecorder(Recorder):
    provider = "em"
    # ✅ Best Practice: Setting index for DataFrame for efficient data manipulation
    data_schema = Stockhk

    # ✅ Best Practice: Adding a new column to DataFrame to indicate a specific attribute
    def run(self):
        df_south = em_api.get_tradable_list(entity_type="stockhk", hk_south=True)
        # 🧠 ML Signal: Usage of external API to fetch data
        df_south = df_south.set_index("code", drop=False)
        df_south["south"] = True
        # ✅ Best Practice: Setting index for DataFrame for efficient data manipulation

        df = em_api.get_tradable_list(entity_type="stockhk")
        # ✅ Best Practice: Using DataFrame operations to filter and copy data
        df = df.set_index("code", drop=False)
        df_other = df.loc[~df.index.isin(df_south.index)].copy()
        # 🧠 ML Signal: Instantiation and execution of a class method
        # ✅ Best Practice: Adding a new column to DataFrame to indicate a specific attribute
        # 🧠 ML Signal: Storing data into a database
        # 🧠 ML Signal: Entry point for script execution
        # 🧠 ML Signal: Defining public API of the module
        df_other["south"] = False
        df_to_db(df=df_south, data_schema=self.data_schema, provider=self.provider, force_update=self.force_update)
        df_to_db(df=df_other, data_schema=self.data_schema, provider=self.provider, force_update=self.force_update)


if __name__ == "__main__":
    recorder = EMStockhkRecorder()
    recorder.run()


# the __all__ is generated
__all__ = ["EMStockhkRecorder"]