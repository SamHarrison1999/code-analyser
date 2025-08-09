# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports from the same module together improves readability and maintainability.

from zvt.contract.api import df_to_db
from zvt.contract.recorder import Recorder
from zvt.domain.meta.stockus_meta import Stockus
# 🧠 ML Signal: Inheritance pattern indicating a subclass relationship
from zvt.recorders.em import em_api

# 🧠 ML Signal: Class attribute assignment pattern

# 🧠 ML Signal: Usage of external API to fetch data
class EMStockusRecorder(Recorder):
    # 🧠 ML Signal: Class attribute assignment pattern
    provider = "em"
    # 🧠 ML Signal: Logging of data for monitoring or debugging
    data_schema = Stockus

    # 🧠 ML Signal: Data persistence pattern
    def run(self):
        df = em_api.get_tradable_list(entity_type="stockus")
        # 🧠 ML Signal: Instantiation and execution of a class method
        # ✅ Best Practice: Use of __all__ to define public interface of the module
        # ✅ Best Practice: Use of Python's entry point check
        self.logger.info(df)
        df_to_db(df=df, data_schema=self.data_schema, provider=self.provider, force_update=self.force_update)


if __name__ == "__main__":
    recorder = EMStockusRecorder()
    recorder.run()


# the __all__ is generated
__all__ = ["EMStockusRecorder"]