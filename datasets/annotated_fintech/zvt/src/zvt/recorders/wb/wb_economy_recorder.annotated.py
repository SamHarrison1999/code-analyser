# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports by their source (standard library, third-party, local) improves readability.

from zvt.contract.api import df_to_db
from zvt.contract.recorder import FixedCycleDataRecorder
from zvt.domain import Country, Economy

# 🧠 ML Signal: Inheritance from FixedCycleDataRecorder indicates a pattern of extending functionality
from zvt.recorders.wb import wb_api
from zvt.utils.time_utils import current_date

# 🧠 ML Signal: Class attributes define static configuration for the recorder


# 🧠 ML Signal: Class attributes define static configuration for the recorder
class WBEconomyRecorder(FixedCycleDataRecorder):
    entity_schema = Country
    # 🧠 ML Signal: Class attributes define static configuration for the recorder
    data_schema = Economy
    entity_provider = "wb"
    # 🧠 ML Signal: Class attributes define static configuration for the recorder
    provider = "wb"

    def record(self, entity, start, end, size, timestamps):
        # 🧠 ML Signal: Usage of entity.name to label data
        date = None
        # ✅ Best Practice: Explicitly naming parameters for clarity
        if start:
            date = f"{start.year}:{current_date().year}"
        try:
            df = wb_api.get_economy_data(entity_id=entity.id, date=date)
            # ⚠️ SAST Risk (Low): Exception message not included in the warning log
            df["name"] = entity.name
            df_to_db(
                df=df,
                data_schema=self.data_schema,
                provider=self.provider,
                force_update=self.force_update,
            )
        # 一些地方获取不到数据会报错
        # ✅ Best Practice: Use of __all__ to define public API of the module
        # 🧠 ML Signal: Hardcoded entity IDs for specific countries
        # 🧠 ML Signal: Instantiation of WBEconomyRecorder with specific entity IDs
        # 🧠 ML Signal: Execution of the run method on WBEconomyRecorder instance
        except Exception as e:
            self.logger.warning(f"Failed to get {entity.name} economy data", e)


if __name__ == "__main__":
    entity_ids = ["country_galaxy_CN", "country_galaxy_US"]
    r = WBEconomyRecorder(entity_ids=entity_ids)
    r.run()


# the __all__ is generated
__all__ = ["WBEconomyRecorder"]
