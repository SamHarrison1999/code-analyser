# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports from the same module together improves readability.

from zvt.contract.api import df_to_db

# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.contract.recorder import Recorder

# ✅ Best Practice: Class definition should include a docstring to describe its purpose and usage
from zvt.domain import Future

# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.recorders.em import em_api

# ✅ Best Practice: Class attributes should be documented to explain their purpose


# ✅ Best Practice: Class attributes should be documented to explain their purpose
# 🧠 ML Signal: Usage of external API to fetch data
class EMFutureRecorder(Recorder):
    provider = "em"
    # 🧠 ML Signal: Logging of data for monitoring or debugging
    data_schema = Future

    # 🧠 ML Signal: Data persistence pattern
    def run(self):
        df = em_api.get_tradable_list(entity_type="future")
        # 🧠 ML Signal: Execution of a method on an object
        # ✅ Best Practice: Use of __all__ to define public API of the module
        # ✅ Best Practice: Standard Python entry point check
        # 🧠 ML Signal: Object instantiation with specific configuration
        self.logger.info(df)
        df_to_db(
            df=df,
            data_schema=self.data_schema,
            provider=self.provider,
            force_update=self.force_update,
        )


if __name__ == "__main__":
    recorder = EMFutureRecorder(force_update=True)
    recorder.run()


# the __all__ is generated
__all__ = ["EMFutureRecorder"]
