# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports from the same module together improves readability.

from zvt.contract.api import df_to_db

# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.contract.recorder import Recorder

# 🧠 ML Signal: Class definition with inheritance, useful for understanding class hierarchies and relationships
from zvt.domain.meta.indexus_meta import Indexus

# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.recorders.em import em_api

# 🧠 ML Signal: Class attribute definition, useful for understanding default values and configurations


# 🧠 ML Signal: Class attribute definition, useful for understanding default values and configurations
# 🧠 ML Signal: Usage of external API to fetch data
class EMIndexusRecorder(Recorder):
    provider = "em"
    # 🧠 ML Signal: Logging of data for monitoring or debugging
    data_schema = Indexus

    # 🧠 ML Signal: Data persistence pattern
    def run(self):
        df = em_api.get_tradable_list(entity_type="indexus")
        # ✅ Best Practice: Use of __all__ to define public API of the module
        # ✅ Best Practice: Standard Python entry point check
        # 🧠 ML Signal: Instantiation and execution of a class method
        self.logger.info(df)
        df_to_db(
            df=df,
            data_schema=self.data_schema,
            provider=self.provider,
            force_update=self.force_update,
        )


if __name__ == "__main__":
    recorder = EMIndexusRecorder()
    recorder.run()


# the __all__ is generated
__all__ = ["EMIndexusRecorder"]
