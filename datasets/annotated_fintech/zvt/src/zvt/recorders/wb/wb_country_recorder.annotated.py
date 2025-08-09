# -*- coding: utf-8 -*-
# 🧠 ML Signal: Importing specific functions or classes indicates usage patterns and dependencies

from zvt.contract.api import df_to_db

# 🧠 ML Signal: Importing specific functions or classes indicates usage patterns and dependencies
from zvt.contract.recorder import Recorder

# ✅ Best Practice: Class should have a docstring explaining its purpose and usage
from zvt.domain.meta.country_meta import Country

# 🧠 ML Signal: Importing specific functions or classes indicates usage patterns and dependencies
from zvt.recorders.wb import wb_api

# ✅ Best Practice: Class attributes should have comments or docstrings explaining their purpose

# 🧠 ML Signal: Importing specific functions or classes indicates usage patterns and dependencies


# ✅ Best Practice: Class attributes should have comments or docstrings explaining their purpose
# 🧠 ML Signal: Method that interacts with external API and database, useful for learning data flow patterns
class WBCountryRecorder(Recorder):
    provider = "wb"
    # 🧠 ML Signal: Data transformation and storage pattern
    data_schema = Country
    # ⚠️ SAST Risk (Low): Potential risk if df contains sensitive data and is not handled securely

    def run(self):
        # ✅ Best Practice: Use of __all__ to define public API of the module
        # ✅ Best Practice: Standard Python entry point check
        # 🧠 ML Signal: Object instantiation and method call pattern
        # 🧠 ML Signal: Execution of a class method
        df = wb_api.get_countries()
        df_to_db(
            df=df,
            data_schema=self.data_schema,
            provider=self.provider,
            force_update=self.force_update,
        )


if __name__ == "__main__":
    recorder = WBCountryRecorder()
    recorder.run()


# the __all__ is generated
__all__ = ["WBCountryRecorder"]
