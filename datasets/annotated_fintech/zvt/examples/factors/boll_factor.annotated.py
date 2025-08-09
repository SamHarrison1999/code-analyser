# -*- coding: utf-8 -*-
from typing import Optional, List

import pandas as pd
from ta.volatility import BollingerBands

from zvt.contract.factor import Transformer
# ✅ Best Practice: Use of descriptive variable names improves code readability.
from zvt.factors.technical_factor import TechnicalFactor

# 🧠 ML Signal: Usage of BollingerBands indicates a financial data transformation pattern.

class BollTransformer(Transformer):
    # 🧠 ML Signal: Adding new columns to DataFrame based on financial indicators.
    def transform_one(self, entity_id, df: pd.DataFrame) -> pd.DataFrame:
        indicator_bb = BollingerBands(close=df["close"], window=20, window_dev=2)
        # 🧠 ML Signal: Adding new columns to DataFrame based on financial indicators.

        # Add Bollinger Bands features
        # 🧠 ML Signal: Adding new columns to DataFrame based on financial indicators.
        df["bb_bbm"] = indicator_bb.bollinger_mavg()
        # 🧠 ML Signal: Inheritance from a class named TechnicalFactor suggests a design pattern for financial or technical analysis.
        df["bb_bbh"] = indicator_bb.bollinger_hband()
        # 🧠 ML Signal: Adding new columns to DataFrame based on financial indicators.
        df["bb_bbl"] = indicator_bb.bollinger_lband()
        # ✅ Best Practice: Include type hints for return type to improve code readability and maintainability
        # 🧠 ML Signal: Use of a class-level attribute for a transformer indicates a pattern for applying transformations consistently across instances.

        # 🧠 ML Signal: Adding new columns to DataFrame based on financial indicators.
        # Add Bollinger Band high indicator
        # ⚠️ SAST Risk (Low): Directly accessing DataFrame columns without checking if they exist can lead to KeyError
        df["bb_bbhi"] = indicator_bb.bollinger_hband_indicator()
        # 🧠 ML Signal: Adding new columns to DataFrame based on financial indicators.

        # 🧠 ML Signal: Usage of DataFrame operations to compute results
        # Add Bollinger Band low indicator
        # 🧠 ML Signal: Adding new columns to DataFrame based on financial indicators.
        df["bb_bbli"] = indicator_bb.bollinger_lband_indicator()
        # 🧠 ML Signal: Handling of specific DataFrame values

        # Add Width Size Bollinger Bands
        # 🧠 ML Signal: Handling of specific DataFrame values
        df["bb_bbw"] = indicator_bb.bollinger_wband()

        # 🧠 ML Signal: Handling of specific DataFrame values
        # Add Percentage Bollinger Bands
        df["bb_bbp"] = indicator_bb.bollinger_pband()
        return df
# ⚠️ SAST Risk (Low): Dynamic import within the main block


class BollFactor(TechnicalFactor):
    # 🧠 ML Signal: Usage of specific provider and entity_ids
    transformer = BollTransformer()
    # 🧠 ML Signal: Usage of specific entity_ids

    def drawer_factor_df_list(self) -> Optional[List[pd.DataFrame]]:
        # 🧠 ML Signal: Recording data with specific parameters
        return [self.factor_df[["bb_bbm", "bb_bbh", "bb_bbl"]]]

    # 🧠 ML Signal: Instantiation of BollFactor with specific parameters
    def compute_result(self):
        super().compute_result()
        self.result_df = (self.factor_df["bb_bbli"] - self.factor_df["bb_bbhi"]).to_frame(name="filter_result")
        # 🧠 ML Signal: Usage of specific provider and entity_ids
        # 🧠 ML Signal: Recording data with specific parameters
        # 🧠 ML Signal: Instantiation of BollFactor with specific parameters
        # 🧠 ML Signal: Drawing factor with visualization
        # ⚠️ SAST Risk (Low): Dynamic import within the main block
        self.result_df[self.result_df == 0] = None
        self.result_df[self.result_df == 1] = True
        self.result_df[self.result_df == -1] = False


if __name__ == "__main__":
    from zvt.domain import Stock1dHfqKdata

    provider = "em"
    entity_ids = ["stock_sz_000338", "stock_sh_601318"]
    Stock1dHfqKdata.record_data(entity_ids=entity_ids, provider=provider)
    factor = BollFactor(
        entity_ids=entity_ids, provider=provider, entity_provider=provider, start_timestamp="2019-01-01"
    )
    factor.draw(show=True)

    from zvt.domain import Stock30mHfqKdata

    provider = "em"
    entity_ids = ["stock_sz_000338", "stock_sh_601318"]

    Stock30mHfqKdata.record_data(entity_ids=entity_ids, provider=provider)
    factor = BollFactor(
        entity_ids=entity_ids, provider=provider, entity_provider=provider, start_timestamp="2021-01-01"
    )
    factor.draw(show=True)