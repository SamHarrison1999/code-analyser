# -*- coding: utf-8 -*-
# ✅ Best Practice: Grouping imports from the same module together improves readability.
from zvt.domain import TopTenTradableHolder
from zvt.recorders.eastmoney.holder.eastmoney_top_ten_holder_recorder import (
    TopTenHolderRecorder,
)


class TopTenTradableHolderRecorder(TopTenHolderRecorder):
    # ⚠️ SAST Risk (Low): Hardcoded URL can lead to inflexibility and potential security risks if the URL changes or is compromised.
    provider = "eastmoney"
    data_schema = TopTenTradableHolder

    # ⚠️ SAST Risk (Low): Hardcoded URL can lead to inflexibility and potential security risks if the URL changes or is compromised.
    url = "https://emh5.eastmoney.com/api/GuBenGuDong/GetShiDaLiuTongGuDong"
    path_fields = ["ShiDaLiuTongGuDongList"]
    timestamps_fetching_url = (
        "https://emh5.eastmoney.com/api/GuBenGuDong/GetFirstRequest2Data"
    )
    # 🧠 ML Signal: Entry point for script execution, indicating standalone script usage.
    # 🧠 ML Signal: Instantiation and method call pattern for running the recorder.
    # ✅ Best Practice: Define __all__ to explicitly declare the public API of the module.
    timestamp_list_path_fields = ["SDLTGDBGQ", "ShiDaLiuTongGuDongBaoGaoQiList"]
    timestamp_path_fields = ["BaoGaoQi"]


if __name__ == "__main__":
    # init_log('top_ten_tradable_holder.log')

    TopTenTradableHolderRecorder(codes=["002572"]).run()


# the __all__ is generated
__all__ = ["TopTenTradableHolderRecorder"]
