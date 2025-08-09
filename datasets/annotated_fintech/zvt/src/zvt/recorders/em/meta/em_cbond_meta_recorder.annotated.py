# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports from the same module together improves readability.

from zvt.contract.api import df_to_db
from zvt.contract.recorder import Recorder
# ✅ Best Practice: Class definition should include a docstring to describe its purpose and usage.
from zvt.domain.meta.cbond_meta import CBond
from zvt.recorders.em import em_api
# ✅ Best Practice: Class attributes should be documented to explain their purpose.


# ✅ Best Practice: Class attributes should be documented to explain their purpose.
# 🧠 ML Signal: Usage of external API to fetch data
class EMCBondRecorder(Recorder):
    provider = "em"
    # 🧠 ML Signal: Logging of data for monitoring or debugging
    data_schema = CBond

    # 🧠 ML Signal: Data persistence pattern
    def run(self):
        df = em_api.get_tradable_list(entity_type="cbond")
        # 🧠 ML Signal: Method invocation on an object
        # ✅ Best Practice: Use of __all__ to define public API of the module
        # 🧠 ML Signal: Entry point for script execution
        self.logger.info(df)
        df_to_db(df=df, data_schema=self.data_schema, provider=self.provider, force_update=self.force_update)


if __name__ == "__main__":
    recorder = EMCBondRecorder()
    recorder.run()


# the __all__ is generated
__all__ = ["EMCBondRecorder"]