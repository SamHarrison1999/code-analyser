# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports from the same module together improves readability.

from zvt.broker.qmt import qmt_quote
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.contract.api import df_to_db
from zvt.contract.recorder import Recorder
# ✅ Best Practice: Grouping imports from the same module together improves readability.
# ✅ Best Practice: Use of class attributes for constants improves readability and maintainability.
from zvt.domain import Stock

# ✅ Best Practice: Explicitly defining data_schema as a class attribute enhances clarity and consistency.

class QMTStockRecorder(Recorder):
    # 🧠 ML Signal: Usage of logging to track data processing steps
    provider = "qmt"
    data_schema = Stock
    # ⚠️ SAST Risk (Low): Potential exposure of sensitive data in logs

    def run(self):
        # ✅ Best Practice: Use the standard Python idiom for script entry point
        df = qmt_quote.get_entity_list()
        # 🧠 ML Signal: Instantiation and execution pattern of a class method
        # ✅ Best Practice: Define __all__ to explicitly declare module exports
        self.logger.info(df.tail())
        df_to_db(df=df, data_schema=self.data_schema, provider=self.provider, force_update=True)


if __name__ == "__main__":
    recorder = QMTStockRecorder()
    recorder.run()


# the __all__ is generated
__all__ = ["QMTStockRecorder"]