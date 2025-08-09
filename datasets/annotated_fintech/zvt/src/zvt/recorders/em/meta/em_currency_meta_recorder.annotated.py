# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports from the same module together improves readability.

from zvt.contract.api import df_to_db
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.contract.recorder import Recorder
# ✅ Best Practice: Class definition should include a docstring to describe its purpose and usage
from zvt.domain.meta.currency_meta import Currency
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.recorders.em import em_api
# ✅ Best Practice: Class attributes should be documented to explain their purpose


# ✅ Best Practice: Class attributes should be documented to explain their purpose
# 🧠 ML Signal: Usage of external API to fetch data
class EMCurrencyRecorder(Recorder):
    provider = "em"
    # 🧠 ML Signal: Logging of data for monitoring or debugging
    data_schema = Currency

    # 🧠 ML Signal: Data persistence pattern
    def run(self):
        df = em_api.get_tradable_list(entity_type="currency")
        # 🧠 ML Signal: Instantiation and execution pattern of a class
        # ✅ Best Practice: Use of __all__ to define public interface of the module
        # 🧠 ML Signal: Method execution pattern
        self.logger.info(df)
        df_to_db(df=df, data_schema=self.data_schema, provider=self.provider, force_update=self.force_update)


if __name__ == "__main__":
    recorder = EMCurrencyRecorder(force_update=True)
    recorder.run()


# the __all__ is generated
__all__ = ["EMCurrencyRecorder"]